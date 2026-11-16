import unittest
import torch
from evaluation.consolidation_soft_control import distillation_loss,soft_steps
from evaluation.text_facts import TextFacts,model_config
from models.model_v2 import PBSLMv2


class SoftConsolidationContracts(unittest.TestCase):
    def test_teacher_detached_student_grad_and_zero_when_matching(self):
        student=torch.tensor([[1.,2.,3.]],requires_grad=True)
        teacher=student.detach().softmax(-1).requires_grad_()
        loss=distillation_loss(student,teacher)
        self.assertAlmostEqual(float(loss.detach()),0.,places=6)
        loss.backward();self.assertIsNone(teacher.grad)
        changed=torch.zeros(1,3,requires_grad=True)
        distillation_loss(changed,teacher).backward()
        self.assertGreater(float(changed.grad.abs().sum()),0.)

    def test_kl_direction_matches_manual_teacher_to_student(self):
        teacher=torch.tensor([[.8,.2]])
        scores=torch.tensor([[0.,0.]])
        expected=(teacher*(teacher.log()-scores.log_softmax(-1))).sum()
        self.assertAlmostEqual(float(distillation_loss(scores,teacher)),float(expected),places=6)

    def test_invalid_distribution_rejected(self):
        for invalid in (torch.tensor([[.2,.2]]),torch.tensor([[-.1,1.1]]),torch.tensor([[float('nan'),0.]])):
            with self.assertRaises(ValueError):distillation_loss(torch.zeros(1,2),invalid)

    def test_unfrozen_student_rejected(self):
        task=TextFacts('cpu');model=PBSLMv2(**model_config(task)).eval();model.use_mem=False
        probs=torch.full((4,2,task.vocab),1/task.vocab)
        with self.assertRaisesRegex(ValueError,'freeze contract'):soft_steps(model,task,probs,1,0)


if __name__=='__main__':unittest.main()
