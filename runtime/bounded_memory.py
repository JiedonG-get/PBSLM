"""External bounded latest-confirmed-feedback ledger rebuilding existing numerical M.

Caller supplies semantic slots and explicitly confirmed feedback, never hidden truth.
This is a system strategy, NOT intrinsic matrix overwrite, factual verification or
learning. Rebuild costs O(capacity) writes. LRU means update-recency (reads do not
refresh); FIFO preserves a slot's original insertion position on overwrite. Ledger
is bounded; audit/idempotency history is append-only and currently unbounded.
Trusted local snapshots include captured E features and numerical M, not weights.
No optimizer, training, threads, crash persistence, or CausalModelSession integration.
"""
from collections import OrderedDict
import copy
import hashlib
import json
import math

import torch
from .model_session import model_fingerprint, state_hash


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


class BoundedFeedbackMemory:
    SCHEMA = 'pbslm.bounded_feedback_memory.v1'

    def __init__(self, model, capacity=8, *, policy='lru', lr=.05):
        if model.training:
            raise ValueError('eval model required')
        if type(capacity) is not int or capacity < 1 or policy not in {'lru', 'fifo'}:
            raise ValueError('positive capacity and lru/fifo policy required')
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or not 0 < lr <= 1:
            raise ValueError('lr must be finite in (0,1]')
        if any(not torch.isfinite(m).all() or m.count_nonzero() for m in model.mem):
            raise ValueError('new ledger requires zero memory; restore for existing state')
        self.model, self.capacity, self.policy, self.lr = model, capacity, policy, lr
        self.version = model_fingerprint(model)
        self.memory_hash = state_hash(model, memory=True)
        self._ledger, self._events, self._audit = OrderedDict(), {}, []

    @property
    def slots(self):
        return tuple(self._ledger)

    @property
    def receipts(self):
        return copy.deepcopy(self._audit)

    def _guard(self):
        if self.model.training or model_fingerprint(self.model) != self.version:
            raise ValueError('model weights/configuration changed')
        if state_hash(self.model, memory=True) != self.memory_hash:
            raise ValueError('external memory mutation')

    @staticmethod
    def _text(value, name):
        if not isinstance(value, str) or not value:
            raise ValueError(f'nonempty {name} required')

    def _payload(self, features, label_ids):
        tensor = torch.as_tensor(features, device=self.model.tok.weight.device,
                                 dtype=self.model.tok.weight.dtype).detach().clone()
        if (tensor.dim() != 4 or tensor.shape[0] != self.model.e_n or tensor.shape[1] != 1
                or tensor.shape[2] < 1 or tensor.shape[3] != self.model.d or not torch.isfinite(tensor).all()):
            raise ValueError('E features must be finite (E,1,T,D), T>=1')
        if (not isinstance(label_ids, list) or not label_ids
                or any(type(i) is not int or not 0 <= i < self.model.vocab for i in label_ids)):
            raise ValueError('nonempty valid integer label IDs required')
        return {'features': tensor.cpu().tolist(), 'label_ids': list(label_ids)}

    @torch.no_grad()
    def _rebuild(self, ledger):
        for memory in self.model.mem:
            memory.zero_()
        for payload in ledger.values():
            features = torch.tensor(payload['features'], device=self.model.tok.weight.device,
                                    dtype=self.model.tok.weight.dtype)
            ids = torch.tensor([payload['label_ids']], device=features.device, dtype=torch.long)
            self.model.write_memory(features, self.model.target_embed(ids), lr=self.lr)
        if any(not torch.isfinite(m).all() for m in self.model.mem):
            raise FloatingPointError('nonfinite rebuilt memory')

    @torch.no_grad()
    def _apply(self, event_id, request):
        self._guard()
        self._text(event_id, 'event_id')
        self._text(request['slot'], 'slot')
        if event_id in self._events:
            saved = self._events[event_id]
            if saved['request'] != request:
                raise ValueError('event ID conflicts with original payload')
            return copy.deepcopy(saved['receipt'])
        ledger = copy.deepcopy(self._ledger)
        slot = request['slot']
        evicted = None
        if request['operation'] == 'put':
            if self.policy == 'lru':
                ledger.pop(slot, None)
            ledger[slot] = request['payload']
            if len(ledger) > self.capacity:
                evicted, _ = ledger.popitem(last=False)
        else:
            ledger.pop(slot, None)
        original = [m.detach().clone() for m in self.model.mem]
        before = self.memory_hash
        try:
            self._rebuild(ledger)
            after = state_hash(self.model, memory=True)
        except Exception:
            for dst, src in zip(self.model.mem, original):
                dst.copy_(src)
            raise
        receipt = {'event_id': event_id, 'version': len(self._audit) + 1,
                   'operation': request['operation'], 'slot': slot, 'evicted': evicted,
                   'retained_slots': list(ledger), 'write_calls': len(ledger), 'lr': self.lr,
                   'payload_sha256': hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
                   'memory_before': before, 'memory_after': after,
                   'provenance': 'caller_asserted_confirmed_feedback'}
        self._ledger, self.memory_hash = ledger, after
        self._events[event_id] = {'request': copy.deepcopy(request), 'receipt': copy.deepcopy(receipt)}
        self._audit.append(copy.deepcopy(receipt))
        return copy.deepcopy(receipt)

    def put_confirmed(self, event_id, slot, features, label_ids):
        """Capture E keys and labels now; caller, not this controller, confirms truth."""
        payload = self._payload(features, label_ids)
        return self._apply(event_id, {'operation': 'put', 'slot': slot, 'payload': payload})

    def remove(self, event_id, slot):
        return self._apply(event_id, {'operation': 'remove', 'slot': slot})

    def snapshot(self):
        self._guard()
        return _json({'schema': self.SCHEMA, 'model_fingerprint': self.version,
                      'capacity': self.capacity, 'policy': self.policy, 'lr': self.lr,
                      'ledger': list(self._ledger.items()), 'events': self._events, 'audit': self._audit,
                      'memory': [m.detach().cpu().tolist() for m in self.model.mem]})

    @classmethod
    @torch.no_grad()
    def restore(cls, model, snapshot):
        """Verify by temporary reconstruction; failed verification restores original M.

        Floating equality uses atol=1e-6, rtol=1e-5. Successful restore installs the
        verified reconstruction, not unchecked stored numbers. Snapshot is trusted
        local data, not an authentication/security format.
        """
        value = _json(snapshot)
        keys = {'schema', 'model_fingerprint', 'capacity', 'policy', 'lr', 'ledger', 'events', 'audit', 'memory'}
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError('snapshot fields mismatch')
        if value['schema'] != cls.SCHEMA or value['model_fingerprint'] != model_fingerprint(model):
            raise ValueError('snapshot model/schema mismatch')
        if len(value['memory']) != len(model.mem):
            raise ValueError('memory count mismatch')
        memories = [torch.tensor(item, device=dst.device, dtype=dst.dtype)
                    for item, dst in zip(value['memory'], model.mem)]
        if any(a.shape != b.shape or not torch.isfinite(a).all() for a, b in zip(memories, model.mem)):
            raise ValueError('invalid numerical memory')
        original = [m.detach().clone() for m in model.mem]
        try:
            for m in model.mem:
                m.zero_()
            obj = cls(model, value['capacity'], policy=value['policy'], lr=value['lr'])
            for slot, payload in value['ledger']:
                obj._text(slot, 'slot')
                if slot in obj._ledger:
                    raise ValueError('duplicate retained slot')
                obj._ledger[slot] = obj._payload(payload['features'], payload['label_ids'])
            if len(obj._ledger) > obj.capacity:
                raise ValueError('ledger exceeds capacity')
            if not isinstance(value['events'], dict) or not isinstance(value['audit'], list):
                raise ValueError('invalid audit history')
            if len(value['events']) != len(value['audit']):
                raise ValueError('event/receipt count mismatch')
            shadow = OrderedDict()
            seen_events = set()
            previous_hash = obj.memory_hash
            for version, receipt in enumerate(value['audit'], 1):
                saved = value['events'].get(receipt['event_id'])
                if receipt['event_id'] in seen_events:
                    raise ValueError('duplicate audit event')
                seen_events.add(receipt['event_id'])
                if receipt['version'] != version or saved is None or saved['receipt'] != receipt:
                    raise ValueError('event/receipt mismatch')
                request = saved['request']
                request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
                if request_hash != receipt['payload_sha256']:
                    raise ValueError('event payload hash mismatch')
                slot = request['slot']
                obj._text(slot, 'event slot')
                evicted = None
                if request['operation'] == 'put':
                    payload = obj._payload(request['payload']['features'], request['payload']['label_ids'])
                    if obj.policy == 'lru':
                        shadow.pop(slot, None)
                    shadow[slot] = payload
                    if len(shadow) > obj.capacity:
                        evicted, _ = shadow.popitem(last=False)
                elif request['operation'] == 'remove':
                    shadow.pop(slot, None)
                else:
                    raise ValueError('unknown ledger operation')
                if (receipt['operation'] != request['operation'] or receipt['slot'] != slot
                        or receipt['evicted'] != evicted or receipt['retained_slots'] != list(shadow)
                        or receipt['write_calls'] != len(shadow) or receipt['lr'] != obj.lr
                        or receipt['memory_before'] != previous_hash):
                    raise ValueError('ledger receipt semantics mismatch')
                previous_hash = receipt['memory_after']
            if list(shadow.items()) != list(obj._ledger.items()):
                raise ValueError('retained ledger disagrees with event history')
            obj._rebuild(obj._ledger)
            if any(not torch.allclose(a, b, atol=1e-6, rtol=1e-5) for a, b in zip(model.mem, memories)):
                raise ValueError('ledger does not reconstruct snapshot numerical memory')
            candidate = [m.detach().clone() for m in model.mem]
            obj._events, obj._audit = value['events'], value['audit']
        finally:
            for dst, src in zip(model.mem, original):
                dst.copy_(src)
        for dst, src in zip(model.mem, candidate):
            dst.copy_(src)
        obj.memory_hash = state_hash(model, memory=True)
        return obj
