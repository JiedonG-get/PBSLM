import unittest
from pathlib import Path
from evaluation.text_facts import TextFacts
from runtime.text_codec import CompactTextCodec,InputContractError

ROOT=Path(__file__).resolve().parents[1]


class TextCodecTests(unittest.TestCase):
    def setUp(self):
        self.task=TextFacts('cpu')
        self.codec=CompactTextCodec(self.task.original_ids,self.task.length,
                         ROOT/'tokenizer/bpe16k_tokenizer.json',self.task.tokenizer_hash)

    def test_known_surface_exact_mapping(self):
        for text,ids in self.task.original_bpe_by_text.items():
            if text in self.task.color_tokens.values():continue
            encoded=self.codec.inspect(text)
            self.assertEqual(encoded['original_bpe_ids'],ids)
            self.assertEqual(encoded['compact_ids'],[self.task.pad_id]*(self.task.length-len(ids))+
                             [self.task.mapping[x] for x in ids])

    def test_no_silent_unknown_or_truncation(self):
        with self.assertRaises(InputContractError):self.codec.encode('unseen English xylophone')
        with self.assertRaises(InputContractError) as cm:self.codec.encode('颜色'*100)
        self.assertEqual(cm.exception.reason,'length_overflow')
        with self.assertRaises(InputContractError):self.codec.encode('')

    def test_hash_and_contract_validation(self):
        with self.assertRaises(ValueError):CompactTextCodec(self.task.original_ids,self.task.length,
                             ROOT/'tokenizer/bpe16k_tokenizer.json','incorrect')


if __name__=='__main__':unittest.main()
