"""Deterministic persistent-memory protocol checks, no parameter updates."""
import unittest
import torch
from evaluation.persistent_memory import stream, evaluate_arm
import tests.test_episodic_protocol as episodic_tests


class PersistentMemoryProtocolTests(unittest.TestCase):
    def test_corrupt_feedback_has_following_correction(self):
        sequence = stream(0, 32)
        self.assertEqual(len(sequence), 34)
        for i, item in enumerate(sequence):
            if item['is_corrupt']:
                self.assertNotEqual(item['feedback'], item['truth'])
                correction = sequence[i + 1]
                self.assertTrue(correction['is_correction'])
                self.assertEqual(item['key'], correction['key'])
                self.assertEqual(item['truth'], correction['feedback'])

    def test_memory_only_stream_invariants(self):
        model = episodic_tests.EpisodicProtocolTests().model()
        result = evaluate_arm(model, stream(0, 16), 'rebuild8', 'cpu')
        self.assertTrue(result['parameters_unchanged'])
        self.assertTrue(result['c_bitwise_stable'])
        self.assertTrue(result['external_ledger'])
        self.assertTrue(all(x['ledger_items'] <= 8 for x in result['events']))
        self.assertEqual(len(result['events']), 17)
        self.assertGreater(result['write_calls'], len(result['events']))


if __name__ == '__main__':
    unittest.main()
