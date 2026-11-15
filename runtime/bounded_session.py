"""Explicit Runtime confirmation wired to an external bounded slot ledger.

The caller must supply a semantic slot (e.g. entity/color); no entity extraction or
truth verification is claimed. Rebuild cost and external state are visible. Session
parameters remain fixed. Separate class keeps append-only historical controller
and its snapshots unchanged.
"""
import copy
import json

import torch
from .bounded_memory import BoundedFeedbackMemory
from .model_session import CausalModelSession, state_hash
from .session import RuntimeSession, TransitionError


class BoundedCausalModelSession(CausalModelSession):
    SCHEMA = 'pbslm.bounded_causal_model_session.v1'

    def __init__(self, model, session_id='bounded-model-session', *, capacity=8, policy='lru', lr=.05):
        super().__init__(model, session_id)
        self.controller = BoundedFeedbackMemory(model, capacity, policy=policy, lr=lr)

    def stage_feedback(self, observation_id, label_ids, *, event_time, slot):
        if not isinstance(slot, str) or not slot:
            raise ValueError('explicit nonempty semantic slot required')
        candidate = super().stage_feedback(observation_id, label_ids, event_time=event_time)
        self.pending[candidate]['slot'] = slot
        # Bind slot into the protocol evidence, rather than a controller-only alias.
        self.protocol.candidates[candidate]['value']['slot'] = slot
        return candidate

    @torch.no_grad()
    def confirm_memory(self, candidate_id, *, lr=.05):
        self._guard()
        if lr != self.controller.lr:
            raise ValueError('bounded rebuild uses fixed configured lr')
        if candidate_id in self.applied:
            return {**copy.deepcopy(self.receipts[candidate_id]), 'replayed': True}
        candidate = self.protocol.candidates.get(candidate_id)
        if candidate is None or candidate['status'] != 'pending' or candidate_id not in self.pending:
            raise TransitionError('pending captured feedback required')
        payload = self.pending[candidate_id]
        if payload.get('slot') != candidate['value'].get('slot'):
            raise ValueError('semantic slot evidence mismatch')
        old_protocol = self.protocol.snapshot()
        old_controller = self.controller.snapshot()
        try:
            transaction = self.controller.put_confirmed(candidate_id, payload['slot'],
                                                        payload['features'], payload['label_ids'])
            confirmation = self.protocol.confirm_write(candidate_id, confirmation_provenance='feedback',
                                                       available_at=candidate['available_at'])
        except Exception:
            self.controller = BoundedFeedbackMemory.restore(self.model, old_controller)
            self.protocol = RuntimeSession.restore(old_protocol)
            self.memory_hash = state_hash(self.model, memory=True)
            raise
        self.applied.append(candidate_id)
        self.memory_hash = state_hash(self.model, memory=True)
        self.receipts[candidate_id] = {**transaction, 'version': confirmation['version'],
                                       'observation_id': candidate['value']['observation_id']}
        return {**copy.deepcopy(self.receipts[candidate_id]), 'replayed': False}

    def snapshot(self):
        base = super().snapshot()
        base['schema'] = CausalModelSession.SCHEMA
        return {'schema': self.SCHEMA, 'base': base, 'controller': self.controller.snapshot()}

    @classmethod
    @torch.no_grad()
    def restore(cls, model, snapshot):
        value = json.loads(json.dumps(snapshot, allow_nan=False))
        if not isinstance(value, dict) or set(value) != {'schema','base','controller'} or value['schema'] != cls.SCHEMA:
            raise ValueError('bounded session snapshot schema mismatch')
        old = [mem.detach().clone() for mem in model.mem]
        try:
            # Validate both histories and numerical representations, no external emissions.
            base = CausalModelSession.restore(model, value['base'])
            controller = BoundedFeedbackMemory.restore(model, value['controller'])
            if state_hash(model, memory=True) != base.memory_hash:
                raise ValueError('base and ledger memory snapshots differ')
            if base.applied != [r['event_id'] for r in controller.receipts]:
                raise ValueError('confirmation and bounded ledger histories differ')
            for cid, payload in base.pending.items():
                if payload.get('slot') != base.protocol.candidates[cid]['value'].get('slot'):
                    raise ValueError('pending semantic slot mismatch')
            obj = cls.__new__(cls)
            obj.__dict__ = base.__dict__
            obj.controller = controller
            return obj
        except Exception:
            for dst, original in zip(model.mem, old):
                dst.copy_(original)
            raise
