import unittest

import torch

from evaluation.bounded_online_memory import (
    ARMS, DEFAULT_LABELS, SEED0_WEIGHTS_SHA256, make_memory, parse_observation, parse_query,
)


def observation(key, value, status='确认'):
    return f'{status}：{key}的存放位置是{value}。'


def query(key):
    return f'请问{key}的当前存放位置？'


class BoundedOnlineMemoryTests(unittest.TestCase):
    def test_confirmed_corrections_and_ignored_observations(self):
        old, new, rumor = DEFAULT_LABELS[:3]
        for arm in ARMS:
            with self.subTest(arm=arm):
                memory = make_memory(arm, 2)
                self.assertIsNone(memory.answer(query('甲样件')))
                self.assertTrue(memory.observe(observation('甲样件', old)))
                self.assertFalse(memory.observe(observation('甲样件', rumor, '传闻')))
                self.assertFalse(memory.observe('背景：广播响起。'))
                self.assertEqual(memory.answer(query('甲样件')), old)
                self.assertTrue(memory.observe(observation('甲样件', new)))
                self.assertEqual(memory.answer(query('甲样件')), new)
                before = memory.metrics()
                self.assertEqual(memory.peek(query('甲样件')), new)
                self.assertEqual(memory.metrics(), before)
                self.assertEqual((before['reads'], before['writes'], before['evictions']), (3, 2, 0))

    def test_bounded_eviction_follows_writes_independently_of_reads(self):
        first, second, third, correction = DEFAULT_LABELS[:4]
        for arm in ARMS:
            with self.subTest(arm=arm):
                memory = make_memory(arm, 2)
                memory.observe(observation('甲样件', first))
                memory.observe(observation('乙样件', second))
                memory.observe(observation('甲样件', correction))
                self.assertEqual(memory.answer(query('乙样件')), second)
                memory.observe(observation('丙样件', third))
                self.assertEqual(memory.answer(query('甲样件')), correction)
                self.assertEqual(memory.answer(query('丙样件')), third)
                metrics = memory.metrics()
                unlimited = arm == 'exact_unbounded'
                self.assertEqual(memory.answer(query('乙样件')), second if unlimited else None)
                self.assertEqual(metrics['evictions'], 0 if unlimited else 1)
                self.assertEqual(metrics['resident_entries'], 3 if unlimited else 2)
                self.assertEqual(metrics['persistent_payload_bytes'], 208 if arm == 'pbs_slots' else (
                    metrics['resident_entries'] * (len('甲样件'.encode('utf-8')) + 1)
                ))

    def test_malformed_text_rejected_before_state_change(self):
        memory = make_memory('exact_bounded', 2)
        before = memory.metrics()
        for text in ('确认：。', '背景：', '确认：甲样件的存放位置是镇政府。', query('甲样件'), None):
            with self.subTest(text=text), self.assertRaises(ValueError):
                memory.observe(text)
        for text in ('请问甲样件在哪儿？', '请问的当前存放位置？', None):
            with self.subTest(text=text), self.assertRaises(ValueError):
                memory.answer(text)
        self.assertEqual(memory.metrics(), before)
        self.assertEqual(parse_observation(observation('校样间甲样件', DEFAULT_LABELS[3])),
                         ('校样间甲样件', DEFAULT_LABELS[3]))
        self.assertEqual(parse_query(query('校样间甲样件')), '校样间甲样件')

    def test_frozen_reader_provenance_and_all_value_classes(self):
        rng_before = torch.random.get_rng_state()
        memory = make_memory('pbs_slots')
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        for label in DEFAULT_LABELS:
            memory.observe(observation('甲样件', label))
            self.assertEqual(memory.answer(query('甲样件')), label)
        self.assertIsNone(memory.answer(query('乙样件')))
        metrics = memory.metrics()
        self.assertEqual(metrics['reader']['weights_sha256'], SEED0_WEIGHTS_SHA256)
        self.assertEqual(metrics['fixed_reader_bytes'], 3752)
        self.assertEqual(metrics['peak_persistent_payload_bytes'], 16 * 26 * 4)


if __name__ == '__main__':
    unittest.main()
