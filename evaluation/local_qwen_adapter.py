"""Load a fixed local Qwen3 checkpoint and generate direct answers."""
import hashlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalQwenAdapter:
    def __init__(self, path, device='cuda', dtype='bfloat16'):
        self.path = Path(path).resolve(strict=True)
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False,
            torch_dtype=self.dtype, low_cpu_mem_usage=True, attn_implementation='sdpa',
        ).to(device).eval()
        self.model.requires_grad_(False)
        if self.model.config.model_type != 'qwen3':
            raise ValueError('only local Qwen3 supported')
        if not self.tokenizer.chat_template:
            raise ValueError('checkpoint chat template required')
        self.limit = min(int(self.model.config.max_position_embeddings), 8192)
        self.eos_ids = self.model.generation_config.eos_token_id
        if isinstance(self.eos_ids, int):
            self.eos_ids = [self.eos_ids]
        if not self.eos_ids:
            raise ValueError('checkpoint EOS configuration required')
        self.think_start = self.tokenizer.convert_tokens_to_ids('<think>')
        self.think_end = self.tokenizer.convert_tokens_to_ids('</think>')
        self.file_hashes = self.checkpoint_hashes()

    def checkpoint_hashes(self):
        hashes = {}
        for path in sorted(self.path.iterdir()):
            if path.is_file():
                with path.open('rb') as stream:
                    hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return hashes

    def render_prompt(self, messages):
        if not isinstance(messages, (tuple, list)) or not messages:
            raise ValueError('nonempty chat messages required')
        for message in messages:
            if (not isinstance(message, dict)
                    or message.get('role') not in {'system', 'user', 'assistant'}
                    or not isinstance(message.get('content'), str)
                    or not message['content'].strip()):
                raise ValueError('chat messages require a role and nonempty content')
        if messages[-1]['role'] != 'user':
            raise ValueError('last chat message must have the user role')
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        if not prompt.endswith('<think>\n\n</think>\n\n'):
            raise ValueError('Qwen3 checkpoint template did not disable thinking')
        return prompt

    @torch.inference_mode()
    def generate(self, messages, *, max_new_tokens=96):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
            raise ValueError('generation token budget must be an integer in 1..256')
        prompt = self.render_prompt(messages)
        batch = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(self.device)
        input_tokens = batch['input_ids'].shape[1]
        if input_tokens + max_new_tokens > self.limit:
            raise ValueError('context plus generation budget too long')
        output = self.model.generate(
            **batch, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_k=None, top_p=None, eos_token_id=self.eos_ids,
        )
        generated = output[0, input_tokens:].tolist()
        answer_ids = generated
        thinking_detected = self.think_start in generated or self.think_end in generated
        if self.think_end in generated:
            last_think_end = len(generated) - 1 - generated[::-1].index(self.think_end)
            answer_ids = generated[last_think_end + 1:]
        thinking_incomplete = self.think_start in answer_ids
        text = '' if thinking_incomplete else self.tokenizer.decode(
            answer_ids, skip_special_tokens=True,
        ).strip()
        return text, {
            'input_tokens': input_tokens,
            'output_tokens': len(generated),
            'max_new_tokens': max_new_tokens,
            'finished_by_eos': bool(generated and generated[-1] in self.eos_ids),
            'enable_thinking': False,
            'thinking_detected': thinking_detected,
            'thinking_incomplete': thinking_incomplete,
            'checkpoint_type': str(self.dtype).removeprefix('torch.'),
            'device': self.device,
            'local_files_only': True,
        }
