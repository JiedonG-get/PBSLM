import unittest
from unittest.mock import patch
import torch
from runtime.fact_reader import CandidateReader,tokenizer
from runtime.controlled_text import ControlledTextSession
from runtime.statement_candidates import StatementCandidates
from runtime.model_session import state_hash
from evaluation.text_facts import TextFacts,model_config
from models.model_v2 import PBSLMv2


class StatementCandidateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4);self.task=TextFacts('cpu')
        self.model=PBSLMv2(**model_config(self.task)).eval()
        self.session=ControlledTextSession(self.model,self.task,capacity=4)
        tok,_=tokenizer();reader=CandidateReader(tok.get_vocab_size()).eval()
        self.pipeline=StatementCandidates(self.session,reader,tok)

    def proposed(self,accepted=True):
        return {'source_text':'甲号设备的颜色是红。','source_sha256':'source-hash',
                'accepted':accepted,'entity':0,'color':0,'entity_name':'甲号设备','color_name':'红',
                'valid_score':.99,'entity_margin':.9,'color_margin':.9}

    def test_propose_then_reject_never_writes(self):
        before=state_hash(self.model,memory=True)
        with patch('runtime.statement_candidates.infer',return_value=[self.proposed()]):
            item=self.pipeline.propose('甲号设备的颜色是红。',event_time=1)
        self.assertEqual(before,state_hash(self.model,memory=True))
        self.pipeline.reject(item['candidate_id'])
        with self.assertRaises(ValueError):self.pipeline.confirm(item['candidate_id'],event_time=2)
        self.assertEqual(before,state_hash(self.model,memory=True))

    def test_explicit_confirm_binds_source_and_is_idempotent(self):
        with patch('runtime.statement_candidates.infer',return_value=[self.proposed()]):
            item=self.pipeline.propose('甲号设备的颜色是红。',event_time=1)
        with self.assertRaises(ValueError):self.pipeline.confirm(item['candidate_id'],event_time=0)
        receipt=self.pipeline.confirm(item['candidate_id'],event_time=2)
        memory=state_hash(self.model,memory=True)
        self.assertEqual(receipt['source_sha256'],'source-hash')
        self.assertEqual(receipt,self.pipeline.confirm(item['candidate_id'],event_time=2))
        self.assertEqual(memory,state_hash(self.model,memory=True))
        confirmed=self.session.inner.protocol.confirmed_writes[-1]
        self.assertEqual(confirmed['value']['source_statement'],'甲号设备的颜色是红。')

    def test_failed_write_leaves_statement_retryable_without_replay(self):
        with patch('runtime.statement_candidates.infer',return_value=[self.proposed()]):
            item=self.pipeline.propose('甲号设备的颜色是红。',event_time=1)
        before=self.session.snapshot()
        with patch.object(self.model,'write_memory',side_effect=RuntimeError('writer fault')):
            with self.assertRaisesRegex(RuntimeError,'writer fault'):
                self.pipeline.confirm(item['candidate_id'],event_time=2)
        self.assertEqual(before,self.session.snapshot())
        self.pipeline.confirm(item['candidate_id'],event_time=2)

    def test_reader_rejection_cannot_be_auto_confirmed(self):
        with patch('runtime.statement_candidates.infer',return_value=[self.proposed(False)]):
            item=self.pipeline.propose('甲号设备不是红色。',event_time=1)
        with self.assertRaises(ValueError):self.pipeline.confirm(item['candidate_id'],event_time=2)


if __name__=='__main__':unittest.main()
