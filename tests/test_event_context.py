"""Token-identity assertions for mixed event context; no model or training."""
import unittest
from types import SimpleNamespace
from runtime.continuous import GenerationRequest
from runtime.event_context import EventContextPacker
from runtime.text_codec import InputContractError


class FakeTokenizer:
    def encode(self,text):return SimpleNamespace(ids=[ord(c)-96 for c in text])


class EventContextTests(unittest.TestCase):
    def codec(self,length=10):
        return SimpleNamespace(tokenizer=FakeTokenizer(),mapping={i:i for i in range(27)},pad_id=0,input_length=length)

    def request(self,inputs=((1,'abc'),(2,'defghi')),prefix='jk'):
        return GenerationRequest(2,inputs,prefix,0.)

    def test_history_prefix_latest_all_present_with_exact_spans(self):
        p=EventContextPacker(self.codec(),history_tokens=2,prefix_tokens=2,min_latest_tokens=2)
        packed=p.pack(self.request())
        self.assertEqual(packed.compact_ids,(2,3,10,11,4,5,6,7,8,9))
        self.assertEqual([s['role'] for s in packed.spans],['prior_input','committed_output','latest_input'])
        for s in packed.spans:self.assertGreater(s['end'],s['start'])
        self.assertEqual(packed.latest_revision,2)
        self.assertEqual(sum(x['tokens'] for x in packed.truncation),1)

    def test_latest_input_suffix_wins_small_budget(self):
        p=EventContextPacker(self.codec(4),history_tokens=3,prefix_tokens=3,min_latest_tokens=3)
        packed=p.pack(self.request())
        self.assertEqual(packed.compact_ids,(11,7,8,9))
        self.assertEqual(packed.spans[-1]['role'],'latest_input')
        self.assertEqual(packed.compact_ids[-3:],(7,8,9))
        self.assertTrue(any(x['tokens']>0 for x in packed.truncation))

    def test_no_silent_unknown_in_history_or_prefix(self):
        p=EventContextPacker(self.codec())
        with self.assertRaises(InputContractError):p.pack(self.request(prefix='!'))
        with self.assertRaises(InputContractError):p.pack(self.request(inputs=((1,'!'),(2,'def'))))

    def test_zero_budget_segments_do_not_block_newest_input(self):
        p=EventContextPacker(self.codec(),history_tokens=0,prefix_tokens=0)
        request=GenerationRequest(9,((1,'!'),(2,'abc')),'!',0.)
        packed=p.pack(request)
        self.assertEqual(packed.latest_revision,2)
        self.assertEqual(request.revision,9)
        self.assertEqual(packed.compact_ids[-3:],(1,2,3))

    def test_no_history_or_prefix_and_left_padding(self):
        p=EventContextPacker(self.codec(),history_tokens=0,prefix_tokens=0)
        packed=p.pack(self.request(inputs=((1,'abc'),),prefix='xyz'))
        self.assertEqual(packed.compact_ids,(0,0,0,0,0,0,0,1,2,3))
        self.assertEqual(packed.spans[0]['start'],7)
        self.assertEqual(len(packed.spans),1)


if __name__=='__main__':unittest.main()
