import unittest

from evaluation.alias_retrieval import rank_context


def observation(tick, text):
    return {'event_id': f'e{tick}', 'source_tick': tick, 'deliver_after_ms': tick,
            'kind': 'observation', 'text': text}


class AliasRetrievalTests(unittest.TestCase):
    def test_chinese_latin_and_digits_normalize_and_preserve_chronology(self):
        first = observation(1, '甲库的ＡＢ－１２文件放在白柜。')
        distractor = observation(2, '乙库的AB-13文件放在白柜。')
        correction = observation(3, '甲库的ab 12文件放在木柜。')
        recent_noise = observation(4, '乙库换下了窗帘。')
        rows = [first, distractor, correction, recent_noise]

        self.assertEqual(rank_context(rows, '甲库 AB12', limit=1), [correction])
        self.assertEqual(rank_context(rows, '甲库 AB12', limit=2), [first, correction])
        self.assertEqual(rank_context(rows, '没有交集', limit=2), [correction, recent_noise])

    def test_complete_prefix_omits_private_fields_and_duplicate_ids(self):
        first = observation(1, '蓝柜有样品。')
        duplicate = {**first, 'text': '催收记录。'}
        last = observation(2, '样品撤出。')
        rows = [{**first, 'oracle': {'target': 'PRIVATE LABEL'}}, duplicate, last]

        self.assertEqual(rank_context(rows, '样品', limit=None), [first, last])
        self.assertEqual(rank_context(rows, '样品', limit=0), [])
        self.assertEqual(rank_context(rows, '催收', limit=1), [last])


if __name__ == '__main__':
    unittest.main()
