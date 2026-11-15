"""Controlled Chinese interface contracts, random weights, no training."""
import json
import unittest
import torch
from evaluation.text_facts import TextFacts, model_config, ENTITIES, TEMPLATES
from models.model_v2 import PBSLMv2
from runtime.controlled_text import ControlledTextSession
from runtime.model_session import state_hash


class ControlledTextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(15)
        self.task=TextFacts('cpu')
        self.model=PBSLMv2(**model_config(self.task)).eval()
        self.session=ControlledTextSession(self.model,self.task,capacity=2)

    def test_text_matches_original_tokenization_and_rejects_open_input(self):
        question=TEMPLATES[4].format(entity=ENTITIES[0])
        ids=self.session.encode_question(question)
        self.assertEqual(ids,self.task.inputs[0,4].tolist())
        with self.assertRaisesRegex(ValueError,'outside'):
            self.session.ask('请随便聊聊电影',event_time=1)
        self.assertEqual(self.session.inner.protocol.revision,0)

    def test_color_confirmation_and_text_snapshot_roundtrip(self):
        question=TEMPLATES[0].format(entity=ENTITIES[0])
        result=self.session.ask(question,event_time=1)
        before=state_hash(self.model,memory=True)
        candidate=self.session.stage_feedback(result['observation_id'],'红',slot='device0/color',event_time=2)
        self.assertEqual(before,state_hash(self.model,memory=True))
        receipt=self.session.confirm(candidate)
        self.assertEqual(receipt['slot'],'device0/color')
        snap=json.loads(json.dumps(self.session.snapshot()))
        torch.manual_seed(15)
        model=PBSLMv2(**model_config(self.task)).eval()
        restored=ControlledTextSession.restore(model,self.task,snap)
        self.assertEqual(restored.snapshot(),self.session.snapshot())
        with self.assertRaisesRegex(ValueError,'codec'):
            ControlledTextSession.restore(model,self.task,{**snap,'task_length':999})

    def test_unsupported_feedback_does_not_change_protocol(self):
        obs=self.session.ask(TEMPLATES[0].format(entity=ENTITIES[0]),event_time=1)
        before=self.session.snapshot()
        with self.assertRaises(ValueError):
            self.session.stage_feedback(obs['observation_id'],'红色也可能蓝色',slot='slot',event_time=2)
        self.assertEqual(before,self.session.snapshot())


if __name__=='__main__': unittest.main()
