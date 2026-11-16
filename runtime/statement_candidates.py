"""Learned bounded-relation extraction proposes; explicit caller confirmation writes M.

No automatic fact verification. A false positive becomes memory pollution if a caller
confirms it. Reader candidate history is ephemeral and not part of model snapshots.
"""
import copy
from evaluation.text_facts import TEMPLATES
from .fact_reader import infer
from .model_session import state_hash
from .session import RuntimeSession


class StatementCandidates:
    def __init__(self,text_session,reader,tok,*,threshold=.9,margin=.2):
        if reader.training:raise ValueError('reader must be eval')
        self.text_session=text_session;self.reader=reader;self.tok=tok
        self.threshold=threshold;self.margin=margin;self._candidates={}

    def propose(self,text,*,event_time):
        self.text_session.inner._guard()
        candidate=infer(self.reader,[text],self.tok,next(self.reader.parameters()).device,
                        self.threshold,self.margin)[0]
        revision=self.text_session.inner.protocol.observe('statement:'+text,event_time=event_time)
        self.text_session.inner.current=None
        ident=f'statement-{revision}'
        self._candidates[ident]={**candidate,'available_at':event_time,'status':'pending','receipt':None}
        return {'candidate_id':ident,**copy.deepcopy(self._candidates[ident])}

    def reject(self,candidate_id,reason='external_rejection'):
        self.text_session.inner._guard()
        value=self._candidates[candidate_id]
        if value['status']!='pending':raise ValueError('pending statement required')
        value['status']='rejected';value['reason']=reason

    def confirm(self,candidate_id,*,event_time):
        self.text_session.inner._guard()
        value=self._candidates[candidate_id]
        if value['status']=='confirmed':return copy.deepcopy(value['receipt'])
        if value['status']!='pending' or not value['accepted']:
            raise ValueError('reader-accepted pending candidate and external confirmation required')
        if event_time<value['available_at']:raise ValueError('confirmation before statement availability')
        inner=self.text_session.inner
        protocol_snapshot=inner.protocol.snapshot()
        logical={key:copy.deepcopy(item) for key,item in inner.__dict__.items()
                 if key not in ('model','controller','protocol')}
        try:
            question=TEMPLATES[0].format(entity=value['entity_name'])
            obs=self.text_session.ask(question,event_time=event_time)
            candidate=self.text_session.stage_feedback(obs['observation_id'],value['color_name'],
                              slot=f"entity:{value['entity']}/color",event_time=event_time)
            protocol=self.text_session.inner.protocol
            protocol.candidates[candidate]['value']['source_text_sha256']=value['source_sha256']
            protocol.candidates[candidate]['value']['source_statement']=value['source_text']
            receipt=self.text_session.confirm(candidate)
        except Exception:
            # Numerical confirm rolls back its own M/controller on failure. Restore
            # staged protocol/capture directly, never re-enter the failing writer.
            inner.protocol=RuntimeSession.restore(protocol_snapshot)
            inner.__dict__.update(logical)
            raise
        value['status']='confirmed';value['receipt']={**receipt,'source_sha256':value['source_sha256'],
                                                    'warning':'caller confirmed extraction, not independently verified fact'}
        return copy.deepcopy(value['receipt'])
