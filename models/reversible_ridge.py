"""Bounded reversible ridge constraints with explicit caller slots.

Stores actual feature/target matrices, not parameter-only consolidation. Latest
caller-confirmed update replaces old slot; solve rebuilds retained constraints.
No semantic slot extraction, truth verification, persistent audit or concurrency.
"""
from collections import OrderedDict
import torch
from .cumulative_ridge import CumulativeRidge


class ReversibleRidge:
    def __init__(self,features,outputs,*,capacity=16,regularization=.01,max_rows_per_slot=8):
        if type(max_rows_per_slot) is not int or max_rows_per_slot<1:raise ValueError('positive per-slot row limit required')
        self.max_rows_per_slot=max_rows_per_slot
        if type(capacity) is not int or capacity<1:raise ValueError('positive capacity required')
        self.capacity=capacity
        self.model=CumulativeRidge(features,outputs,regularization=regularization)
        self._slots=OrderedDict();self.version=0

    def _rebuild(self,slots):
        candidate=CumulativeRidge(self.model.features,self.model.outputs,regularization=self.model.regularization)
        for x,y in slots.values():candidate.add(x,y)
        candidate.solve()
        return candidate

    def put(self,slot,x,y):
        if not isinstance(slot,str) or not slot:raise ValueError('explicit nonempty slot required')
        x=torch.as_tensor(x).detach().cpu().double().clone()
        y=torch.as_tensor(y).detach().cpu().double().clone()
        if x.dim()!=2 or x.shape[0]>self.max_rows_per_slot:
            raise ValueError('feature rows exceed per-slot contract')
        # Validate before mutation with the same numerical contract as accumulation.
        validator=CumulativeRidge(self.model.features,self.model.outputs,regularization=self.model.regularization)
        validator.add(x,y)
        slots=OrderedDict(self._slots);slots.pop(slot,None);slots[slot]=(x,y)
        evicted=None
        if len(slots)>self.capacity:evicted,_=slots.popitem(last=False)
        candidate=self._rebuild(slots)
        self.model,self._slots=candidate,slots;self.version+=1
        return {'version':self.version,'slot':slot,'evicted':evicted,'retained_slots':list(slots),
                'retained_constraints':candidate.samples,'policy':'latest confirmed update-recency, full rebuild'}

    def remove(self,slot):
        if not isinstance(slot,str) or not slot:raise ValueError('explicit nonempty slot required')
        slots=OrderedDict(self._slots);existed=slot in slots;slots.pop(slot,None)
        candidate=self._rebuild(slots)
        self.model,self._slots=candidate,slots;self.version+=1
        return {'version':self.version,'removed':existed,'retained_slots':list(slots)}

    @property
    def slots(self):return tuple(self._slots)

    def predict(self,x):return self.model.predict(x)

    def storage(self):
        state=self.model.storage()
        state.pop('raw_examples_stored',None)
        state['retained_feature_target_rows']=sum(x.shape[0] for x,y in self._slots.values())
        state['max_rows_per_slot']=self.max_rows_per_slot
        state['slot_tensor_bytes']=sum(t.numel()*t.element_size() for pair in self._slots.values() for t in pair)
        state['slot_count']=len(self._slots)
        state['note']='includes ridge tensors and retained slot tensor payload; excludes Python/solve workspace/base model'
        return state
