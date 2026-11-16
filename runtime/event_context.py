"""Explicit fixed-window event/history/output packing for continuous inference.

No trained role embeddings are invented. Segments use existing BPE values, no role
markers; structured spans are evidence only. Packing is NOT semantic understanding.
Unknown/UNK token IDs fail explicitly. Latest input wins limited window allocation.
"""
from dataclasses import dataclass
from .text_codec import InputContractError


@dataclass(frozen=True)
class PackedContext:
    compact_ids: tuple
    spans: tuple
    truncation: tuple
    latest_revision: int


class EventContextPacker:
    def __init__(self,codec,*,history_tokens=3,prefix_tokens=3,min_latest_tokens=2):
        for value in (history_tokens,prefix_tokens,min_latest_tokens):
            if type(value) is not int or value<0:raise ValueError('nonnegative integer budgets required')
        if min_latest_tokens<1 or min_latest_tokens>codec.input_length:
            raise ValueError('latest reserve must fit context')
        self.codec=codec;self.history_tokens=history_tokens;self.prefix_tokens=prefix_tokens
        self.min_latest_tokens=min_latest_tokens

    def _encode(self,text):
        ids=self.codec.tokenizer.encode(text).ids
        if any(token==0 or token not in self.codec.mapping for token in ids):
            raise InputContractError('oov_context_segment',{'original_bpe_ids':ids})
        return [self.codec.mapping[x] for x in ids]

    def pack(self,request):
        if not request.inputs:raise ValueError('at least one input required')
        latest_revision,latest_text=request.inputs[-1]
        latest=self._encode(latest_text)
        if not latest:raise InputContractError('empty_latest_context',{})
        length=self.codec.input_length
        # Reserve suffix of latest observation before history/output allocation.
        reserve=min(len(latest),max(self.min_latest_tokens,length-self.history_tokens-self.prefix_tokens))
        remaining=length-reserve
        prefix=self._encode(request.committed_prefix) if request.committed_prefix and self.prefix_tokens and remaining else []
        prefix_keep=prefix[-min(len(prefix),self.prefix_tokens,remaining):] if self.prefix_tokens and remaining else []
        remaining-=len(prefix_keep)
        history=[(rev,self._encode(text)) for rev,text in request.inputs[:-1]] if self.history_tokens and remaining else []
        history_budget=min(self.history_tokens,remaining)
        selected=[]
        for rev,ids in reversed(history):
            if history_budget<=0:break
            if not ids:continue
            keep=ids[-history_budget:]
            selected.append((rev,keep,len(ids)))
            history_budget-=len(keep)
        selected.reverse()
        used=sum(len(ids) for _,ids,_ in selected)+len(prefix_keep)
        latest_keep=latest[-(length-used):]
        # Model sees past input, committed output, then newest input suffix.
        rows=[];spans=[]
        for rev,ids,total in selected:
            start=len(rows);rows.extend(ids);spans.append({'role':'prior_input','revision':rev,'start':start,'end':len(rows),'source_tokens':total})
        if prefix_keep:
            start=len(rows);rows.extend(prefix_keep);spans.append({'role':'committed_output','start':start,'end':len(rows),'source_tokens':len(prefix)})
        start=len(rows);rows.extend(latest_keep);spans.append({'role':'latest_input','revision':latest_revision,'start':start,'end':len(rows),'source_tokens':len(latest)})
        pad=length-len(rows)
        shifted=tuple({**span,'start':span['start']+pad,'end':span['end']+pad} for span in spans)
        dropped=[{'role':'latest_input','tokens':len(latest)-len(latest_keep)},
                 {'role':'committed_output','tokens':len(prefix)-len(prefix_keep)},
                 {'role':'prior_input','tokens':sum(len(ids) for _,ids in history)-sum(len(ids) for _,ids,_ in selected)}]
        return PackedContext(tuple([self.codec.pad_id]*pad+rows),shifted,tuple(dropped),latest_revision)
