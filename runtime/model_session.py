"""Synchronous real-model inference + explicit feedback memory transactions.

No optimizer or learning steps. Token IDs are supplied by caller; output protocol
commits a JSON text representation of IDs (character offsets, not token streaming).
One sample/session; captures E keys at observation time. Model parameters must stay
fixed; use a different session/version after parameter training.
"""
import copy
import hashlib
import json

import torch
from .session import RuntimeSession, TransitionError


def state_hash(model, *, memory=False):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('mem.') != memory:
            continue
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def inference_config(model):
    names = ('vocab', 'k', 'depth', 'd', 'heads', 'act_x', 'e_n', 'emo_dim',
             'field_scale', 'use_mem', 'trainable_target', 'simple_dec', 'dense',
             'use_flow', 'direct_only', 'use_sample_masks', 'refine')
    module_attributes = {}
    for module_name, module in model.named_modules():
        attributes = {name: getattr(module, name) for name in
                      ('depth', 'heads', 'hd', 'self_causal', 'res_scale', 'steps', 'fdim', 'nb', 'eps')
                      if hasattr(module, name)}
        if attributes:
            module_attributes[module_name] = attributes
    return {'class': type(model).__module__ + '.' + type(model).__qualname__,
            'attributes': {name: getattr(model, name) for name in names},
            'module_attributes': module_attributes,
            'flow_steps': model.flow.steps, 'flow_fdim': model.flow.fdim,
            'dtype': str(model.tok.weight.dtype)}


def model_fingerprint(model):
    h = hashlib.sha256(state_hash(model).encode())
    h.update(json.dumps(inference_config(model), sort_keys=True).encode())
    return h.hexdigest()


class CausalModelSession:
    SCHEMA = 'pbslm.causal_model_session.v1'

    def __init__(self, model, session_id='model-session'):
        if model.training:
            raise ValueError('model must be in eval mode')
        self.model = model
        self.protocol = RuntimeSession(session_id)
        self.model_version = model_fingerprint(model)
        self.memory_hash = state_hash(model, memory=True)
        self.observations = {}
        self.pending = {}
        self.applied = []
        self.receipts = {}
        self.current = None

    def _guard(self):
        if self.model.training or model_fingerprint(self.model) != self.model_version:
            raise ValueError('model version/mode changed; start a new session')
        if state_hash(self.model, memory=True) != self.memory_hash:
            raise ValueError('memory was modified outside the transaction controller')

    def _ids(self, values):
        if not isinstance(values, list) or not values or any(type(x) is not int for x in values):
            raise ValueError('nonempty list of integer token IDs required')
        if any(x < 0 or x >= self.model.vocab for x in values):
            raise ValueError('token ID outside vocabulary')
        return torch.tensor([values], device=self.model.tok.weight.device, dtype=torch.long)

    def observe(self, ids, *, event_time):
        self._guard()
        self._ids(ids)
        revision = self.protocol.observe(json.dumps(ids), event_time=event_time)
        self.current = f'observation-{revision}'
        self.observations[self.current] = {'ids': list(ids), 'observed_at': event_time, 'features': None}
        return self.current

    @torch.no_grad()
    def advance(self, m=1):
        self._guard()
        if self.current is None:
            raise TransitionError('observe a model input first')
        if type(m) is not int or not 1 <= m <= self.model.q_emb.shape[0]:
            raise ValueError('invalid output length')
        item = self.observations[self.current]
        # Infer first; a failed model call must not advance protocol state.
        result = self.model(self._ids(item['ids']), m, sample_noise=False)
        if not torch.isfinite(result['z_a']).all() or not torch.isfinite(result['z_c']).all():
            raise FloatingPointError('model prediction nonfinite')
        self.protocol.advance()
        item['features'] = result['e_feats'].detach().clone()
        decoded = {side: self.model.decode_nearest(result['z_' + side]).argmax(-1)[0].tolist()
                   for side in ('a', 'c')}
        return {'observation_id': self.current, 'tokens_a': decoded['a'], 'tokens_c': decoded['c']}

    def propose(self, token_ids):
        self._guard()
        self._ids(token_ids)
        return self.protocol.propose(json.dumps(token_ids))

    def commit(self, proposal_id, *, expected_offset, count):
        self._guard()
        return self.protocol.commit(proposal_id, expected_offset=expected_offset, count=count)

    def interrupt(self, reason='input_changed'):
        self._guard()
        return self.protocol.interrupt(reason)

    def resume(self):
        self._guard()
        return self.protocol.resume()

    def stage_feedback(self, observation_id, label_ids, *, event_time):
        """Feedback arrival is a new observed event; no label is used before arrival."""
        self._guard()
        self._ids(label_ids)
        if observation_id not in self.observations:
            raise ValueError('unknown observation')
        source = self.observations[observation_id]
        if source['features'] is None:
            raise ValueError('observation has not been processed')
        if event_time < source['observed_at']:
            raise ValueError('feedback precedes observation')
        revision = self.protocol.observe('feedback:' + json.dumps(label_ids), event_time=event_time)
        self.current = None
        candidate = self.protocol.propose_write(
            {'observation_id': observation_id, 'label_ids': list(label_ids), 'feedback_revision': revision},
            provenance='feedback', available_at=event_time)
        self.pending[candidate] = {'features': source['features'].detach().clone(),
                                   'label_ids': list(label_ids)}
        return candidate

    @torch.no_grad()
    def confirm_memory(self, candidate_id, *, lr=.05):
        self._guard()
        if candidate_id in self.applied:
            if lr != self.receipts[candidate_id]['lr']:
                raise ValueError('confirmation retry conflicts with original lr')
            return {**self.receipts[candidate_id], 'replayed': True}
        candidate = self.protocol.candidates.get(candidate_id)
        if candidate is None or candidate['status'] != 'pending' or candidate_id not in self.pending:
            raise TransitionError('pending captured feedback required')
        if not isinstance(lr, (int, float)) or not 0 < lr <= 1:
            raise ValueError('lr must be in (0,1]')
        old_mem = [value.detach().clone() for value in self.model.mem]
        old_protocol = self.protocol.snapshot()
        payload = self.pending[candidate_id]
        try:
            values = self.model.target_embed(self._ids(payload['label_ids']))
            self.model.write_memory(payload['features'], values, lr=lr)
            if any(not torch.isfinite(value).all() for value in self.model.mem):
                raise FloatingPointError('nonfinite memory transaction')
            record = self.protocol.confirm_write(candidate_id, confirmation_provenance='feedback',
                                                  available_at=candidate['available_at'])
        except Exception:
            for destination, original in zip(self.model.mem, old_mem):
                destination.copy_(original)
            self.protocol = RuntimeSession.restore(old_protocol)
            raise
        before_hash = self.memory_hash
        self.applied.append(candidate_id)
        self.memory_hash = state_hash(self.model, memory=True)
        feature_hash = hashlib.sha256(payload['features'].detach().cpu().numpy().tobytes()).hexdigest()
        self.receipts[candidate_id] = {'version': record['version'], 'lr': lr,
                                      'memory_before': before_hash, 'memory_after': self.memory_hash,
                                      'feature_sha256': feature_hash,
                                      'observation_id': candidate['value']['observation_id']}
        return {**self.receipts[candidate_id], 'replayed': False}

    def reject_memory(self, candidate_id, reason='rejected_feedback'):
        self._guard()
        self.protocol.reject_write(candidate_id, reason)

    def snapshot(self):
        self._guard()
        observations = copy.deepcopy(self.observations)
        pending = copy.deepcopy(self.pending)
        for item in observations.values():
            if item['features'] is not None:
                item['features'] = item['features'].detach().cpu().tolist()
        for item in pending.values():
            item['features'] = item['features'].detach().cpu().tolist()
        return {'schema': self.SCHEMA, 'model_version': self.model_version,
                'inference_config': inference_config(self.model),
                'protocol': self.protocol.snapshot(), 'memory': [x.detach().cpu().tolist() for x in self.model.mem],
                'observations': observations, 'pending': pending, 'applied': list(self.applied),
                'receipts': copy.deepcopy(self.receipts), 'current': self.current}

    @classmethod
    @torch.no_grad()
    def restore(cls, model, snapshot):
        """Atomic local restore, including M; no output is re-emitted and no write replayed."""
        value = json.loads(json.dumps(snapshot, allow_nan=False))
        expected_keys = {'schema', 'model_version', 'inference_config', 'protocol', 'memory',
                         'observations', 'pending', 'applied', 'receipts', 'current'}
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise ValueError('snapshot fields mismatch')
        if value['schema'] != cls.SCHEMA or value['model_version'] != model_fingerprint(model) or value['inference_config'] != inference_config(model):
            raise ValueError('snapshot model/schema mismatch')
        protocol = RuntimeSession.restore(value['protocol'])
        if value['applied'] != [record['candidate_id'] for record in protocol.confirmed_writes]:
            raise ValueError('applied memory/confirmation history mismatch')
        if len(value['memory']) != len(model.mem):
            raise ValueError('memory count mismatch')
        device, dtype = model.tok.weight.device, model.tok.weight.dtype
        memories = [torch.tensor(data, device=device, dtype=dst.dtype)
                    for data, dst in zip(value['memory'], model.mem)]
        if any(a.shape != b.shape or not torch.isfinite(a).all() for a, b in zip(memories, model.mem)):
            raise ValueError('invalid numerical memory')
        for group in ('observations', 'pending'):
            for item in value[group].values():
                if item['features'] is not None:
                    item['features'] = torch.tensor(item['features'], device=device, dtype=dtype)
                    if item['features'].dim() != 4 or item['features'].shape[0] != model.e_n or item['features'].shape[1] != 1 or item['features'].shape[-1] != model.d or not torch.isfinite(item['features']).all():
                        raise ValueError('invalid captured E features')
        if value['current'] is not None and value['current'] not in value['observations']:
            raise ValueError('unknown current observation')
        if set(value['receipts']) != set(value['applied']):
            raise ValueError('memory receipt history mismatch')
        for version, candidate_id in enumerate(value['applied'], 1):
            receipt = value['receipts'][candidate_id]
            if receipt.get('version') != version or not 0 < receipt.get('lr', 0) <= 1:
                raise ValueError('invalid memory receipt')
        for candidate_id, payload in value['pending'].items():
            candidate = protocol.candidates.get(candidate_id)
            if candidate is None or payload['features'] is None or payload['label_ids'] != candidate['value']['label_ids']:
                raise ValueError('pending evidence mismatch')
            source = value['observations'].get(candidate['value']['observation_id'])
            if source is None or source['features'] is None or not torch.equal(source['features'], payload['features']):
                raise ValueError('captured observation features mismatch')
        session = cls(model, protocol.session_id)
        session.protocol = protocol
        session.observations, session.pending = value['observations'], value['pending']
        session.applied, session.current = value['applied'], value['current']
        session.receipts = value['receipts']
        for destination, source in zip(model.mem, memories):
            destination.copy_(source)
        session.memory_hash = state_hash(model, memory=True)
        return session
