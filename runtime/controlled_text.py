"""Controlled Chinese text session backed by real model + explicit bounded slots.

Accepted queries are the checkpoint task's exact 96 configured surfaces. Feedback
is one of eight configured colors and slot is supplied explicitly. No free-form
fact extraction, language understanding claim or parameter updates.
"""
from .bounded_session import BoundedCausalModelSession
from .model_session import CausalModelSession


class ControlledTextSession:
    def __init__(self, model, task, *, capacity=4, session_id='controlled-text'):
        self.task = task
        self.inner = (CausalModelSession(model, session_id) if capacity is None else
                      BoundedCausalModelSession(model, session_id, capacity=capacity))
        self.allowed = {text: ids for text, ids in task.original_bpe_by_text.items()
                        if text not in set(task.color_tokens.values())}
        self.colors = {color:token for token,color in task.color_tokens.items()}

    def encode_question(self, text):
        if text not in self.allowed:
            raise ValueError('query outside explicitly configured template inventory')
        row = [self.task.mapping[token] for token in self.allowed[text]]
        if len(row) > self.task.length:
            raise ValueError('query longer than checkpoint input contract')
        return [self.task.pad_id]*(self.task.length-len(row)) + row

    def ask(self, text, *, event_time):
        observation = self.inner.observe(self.encode_question(text), event_time=event_time)
        result = self.inner.advance()
        return {**result, 'observation_id':observation, 'question':text,
                'answer_a':self.render(result['tokens_a']), 'answer_c':self.render(result['tokens_c'])}

    def render(self, ids):
        return ''.join(self.task.color_tokens.get(token,f'<compact-token:{token}>') for token in ids)

    def stage_feedback(self, observation_id, color, *, slot, event_time):
        if color not in self.colors:
            raise ValueError('feedback must be exactly one configured color label')
        if not isinstance(slot,str) or not slot:
            raise ValueError('explicit nonempty semantic slot required')
        arguments = {'slot':slot} if isinstance(self.inner,BoundedCausalModelSession) else {}
        return self.inner.stage_feedback(observation_id,[self.colors[color]],event_time=event_time,**arguments)

    def confirm(self, candidate_id):
        return self.inner.confirm_memory(candidate_id)

    def snapshot(self):
        return {'mode':'bounded' if isinstance(self.inner,BoundedCausalModelSession) else 'append',
                'task_tokenizer_sha256':self.task.tokenizer_hash,
                'task_original_ids':self.task.original_ids,
                'task_length':self.task.length,'inner':self.inner.snapshot()}

    @classmethod
    def restore(cls, model, task, snapshot):
        if (snapshot.get('task_tokenizer_sha256')!=task.tokenizer_hash
                or snapshot.get('task_original_ids')!=task.original_ids
                or snapshot.get('task_length')!=task.length):
            raise ValueError('text snapshot codec mismatch')
        mode=snapshot.get('mode')
        if mode not in ('bounded','append'): raise ValueError('unknown text session mode')
        session_class=BoundedCausalModelSession if mode=='bounded' else CausalModelSession
        inner=session_class.restore(model,snapshot['inner'])
        obj=cls.__new__(cls)
        obj.task=task; obj.inner=inner
        obj.allowed={text:ids for text,ids in task.original_bpe_by_text.items()
                     if text not in set(task.color_tokens.values())}
        obj.colors={color:token for token,color in task.color_tokens.items()}
        return obj
