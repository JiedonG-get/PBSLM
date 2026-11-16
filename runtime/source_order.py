"""Explicit per-source ordering reference for continuous observation state.

Caller-asserted source/sequence, no authentication or clock synchronization. Latest
sequence only orders one source's one slot. Cross-source conflicts remain visible.
No neural understanding, truth verification or numerical M writes.
"""
import copy


class SourceOrderedState:
    def __init__(self,*,slot_limit=64,event_limit=1024):
        if any(type(v) is not int or v<1 for v in (slot_limit,event_limit)):
            raise ValueError('positive state and event capacities required')
        self.slot_limit=slot_limit;self.event_limit=event_limit
        self._slots={};self._events={}

    def apply(self,event_id,*,source,slot,sequence,value):
        if any(not isinstance(v,str) or not v for v in (event_id,source,slot)):
            raise ValueError('nonempty event id, source and slot required')
        if type(sequence) is not int or sequence<0:
            raise ValueError('nonnegative source sequence required')
        payload={'source':source,'slot':slot,'sequence':sequence,'value':copy.deepcopy(value)}
        if event_id in self._events:
            prior=self._events[event_id]
            if prior['payload']!=payload:raise ValueError('event id payload conflict')
            return copy.deepcopy(prior['receipt'])
        key=(source,slot);old=self._slots.get(key)
        if len(self._events)>=self.event_limit:
            raise OverflowError('event history capacity exhausted; explicit checkpoint/rotation required')
        if old is None and len(self._slots)>=self.slot_limit:
            raise OverflowError('source-slot capacity exhausted; no silent eviction of order watermark')
        if old is not None and sequence==old['sequence'] and value!=old['value']:
            raise ValueError('same source sequence has conflicting value')
        accepted=old is None or sequence>old['sequence']
        reason='applied' if accepted else ('duplicate_sequence' if sequence==old['sequence'] else 'late_ignored')
        receipt={'event_id':event_id,**payload,'applied':accepted,'reason':reason,
                 'prior_sequence':None if old is None else old['sequence']}
        if accepted:self._slots[key]={'sequence':sequence,'value':copy.deepcopy(value),'event_id':event_id}
        self._events[event_id]={'payload':payload,'receipt':receipt}
        return copy.deepcopy(receipt)

    def view(self,slot):
        candidates={source:copy.deepcopy(state) for (source,name),state in self._slots.items() if name==slot}
        values=[item['value'] for item in candidates.values()]
        conflict=bool(values) and any(value!=values[0] for value in values[1:])
        return {'slot':slot,'sources':candidates,'cross_source_conflict':conflict,
                'resolved_value':copy.deepcopy(values[0]) if values and not conflict else None,
                'has_value':bool(values) and not conflict,
                'warning':'agreement is not factual verification; source sequences are caller asserted'}
