"""No-truncation BPE codec for arbitrary strings within a fixed checkpoint vocabulary.

Representability is not semantic validity/confidence. Rejection means unknown BPE
IDs or length overflow, never an assertion that a fact or question is wrong.
"""
import hashlib
from pathlib import Path

from tokenizers import Tokenizer


class InputContractError(ValueError):
    def __init__(self, reason, details):
        self.reason, self.details = reason, details
        super().__init__(f'{reason}: {details}')


class CompactTextCodec:
    def __init__(self, original_ids, input_length, tokenizer_path, expected_tokenizer_sha256):
        path=Path(tokenizer_path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_tokenizer_sha256:
            raise ValueError('tokenizer hash mismatch')
        if type(input_length) is not int or input_length<1 or len(set(original_ids))!=len(original_ids) or 0 not in original_ids:
            raise ValueError('invalid compact text contract')
        self.tokenizer=Tokenizer.from_file(str(path))
        self.original_ids=list(original_ids)
        self.mapping={token:i for i,token in enumerate(original_ids)}
        self.input_length=input_length
        self.pad_id=self.mapping[0]

    def inspect(self,text):
        if not isinstance(text,str) or not text.strip():
            raise InputContractError('empty_input',{})
        ids=self.tokenizer.encode(text).ids
        # id0 from tokenizer means actual unknown text, distinct from deliberate pad0.
        unknown=sorted({token for token in ids if token==0 or token not in self.mapping})
        if unknown:
            raise InputContractError('oov_bpe_ids',{'ids':ids,'unknown':unknown})
        if len(ids)>self.input_length:
            raise InputContractError('length_overflow',{'tokens':len(ids),'limit':self.input_length})
        compact=[self.mapping[token] for token in ids]
        return {'original_bpe_ids':ids,'compact_ids':[self.pad_id]*(self.input_length-len(ids))+compact,
                'padding':self.input_length-len(ids)}

    def encode(self,text):
        return self.inspect(text)['compact_ids']
