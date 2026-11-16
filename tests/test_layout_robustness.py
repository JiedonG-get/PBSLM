import unittest
import torch
from evaluation.text_facts import TextFacts
from evaluation.layout_robustness_control import layout_tokens


class LayoutRobustnessTests(unittest.TestCase):
    def test_paired_layout_identity_and_no_truncation(self):
        task=TextFacts('cpu');x=task.inputs[2,3];distractor=task.inputs[3,3]
        self.assertTrue(torch.equal(layout_tokens(task,2,3,'original'),x))
        self.assertTrue(torch.equal(layout_tokens(task,2,3,'repeat'),torch.cat((x,x))))
        self.assertTrue(torch.equal(layout_tokens(task,2,3,'prefix'),torch.cat((distractor,x))))
        self.assertTrue(torch.equal(layout_tokens(task,2,3,'suffix'),torch.cat((x,distractor))))
        with self.assertRaises(ValueError):layout_tokens(task,2,3,'bad')


if __name__=='__main__':unittest.main()
