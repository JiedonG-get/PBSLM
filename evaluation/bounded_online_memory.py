"""Shared parser and state-only memory arms for the controlled Chinese fixture."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re
import struct
import time

import torch

from evaluation.continuous_key_control import state_hash
from models.content_fast_state import ContentFastState
from models.dynamic_content_slots import DynamicContentSlots
from runtime.fast_state_online import load_fast_model


ARMS = ('pbs_slots', 'exact_unbounded', 'exact_bounded')
DEFAULT_CAPACITY = 16
DEFAULT_LABELS = ('松间厅', '雨燕厅', '石桥厅', '东练功房', '竹溪站', '江柳门', '月湾站', '澄光柜')
DEFAULT_READER_PATH = Path(__file__).with_name('fixtures') / 'bounded_online_delta_seed0.json'
READER_FORMAT = 'pbslm.content_fast_state.fp32_json.v1'
SEED0_CHECKPOINT_SHA256 = '7070a5148fbd45fb7e58f38f9ca53c89e1e9ca33646c5f16263b1002264ba028'
SEED0_WEIGHTS_SHA256 = '49312a6c239efec016af1bcc995d7b136c1c6c06af98d270be774fec72613b3c'
SLOT_THRESHOLD = .9

_TERM = r'[^\s。？！：]+'
_LABEL = re.compile(_TERM)
_OBSERVATION = re.compile(rf'(确认|传闻)：({_TERM}?)的存放位置是({_TERM})。')
_BACKGROUND = re.compile(r'背景：[^\r\n]+')
_QUERY = re.compile(rf'请问({_TERM})的当前存放位置？')


def parse_observation(text, labels=DEFAULT_LABELS):
    """Return a confirmed pair, or None for a valid non-writing observation."""
    match = _OBSERVATION.fullmatch(text) if isinstance(text, str) else None
    if match is not None:
        kind, key, value = match.groups()
        if value not in labels:
            raise ValueError('observation value outside the location vocabulary')
        return (key, value) if kind == '确认' else None
    if isinstance(text, str) and _BACKGROUND.fullmatch(text) and text[3:].strip():
        return None
    raise ValueError('malformed observation')


def parse_query(text):
    match = _QUERY.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        raise ValueError('malformed current-location query')
    return match.group(1)


def _key_vector(key):
    digest = hashlib.sha256(b'pbslm.bounded_online.key.v1\x00' + key.encode('utf-8')).digest()
    return torch.tensor([struct.unpack('<16h', digest)], dtype=torch.float32, device='cpu')


def _load_reader(checkpoint):
    if checkpoint is not None:
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            reader, source = load_fast_model(checkpoint)
        reader = reader.float()
        return reader, {'format': 'content_fast_state_checkpoint',
                        'weights_sha256': state_hash(reader), **source}

    artifact = json.loads(DEFAULT_READER_PATH.read_text(encoding='utf-8'))
    if (artifact['format'] != READER_FORMAT or artifact['rule'] != 'delta'
            or artifact['seed'] != 0 or artifact['config'] != {'key_dim': 16, 'values': 8}
            or artifact['checkpoint_file_sha256'] != SEED0_CHECKPOINT_SHA256
            or artifact['weights_sha256'] != SEED0_WEIGHTS_SHA256):
        raise ValueError('bundled reader metadata differs from the published seed-0 checkpoint')

    # The checkpoint supplies all weights. Preserve the caller's RNG state across construction.
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        reader = ContentFastState().float().eval()
    reader.load_state_dict({name: torch.tensor(value, dtype=torch.float32, device='cpu')
                            for name, value in artifact['state_dict'].items()}, strict=True)
    if state_hash(reader) != SEED0_WEIGHTS_SHA256:
        raise ValueError('bundled reader weights differ from the published seed-0 checkpoint')
    return reader, {name: artifact[name] for name in (
        'format', 'seed', 'weights_sha256', 'checkpoint_file_sha256',
    )}


class _Memory:
    def __init__(self, capacity, labels):
        self.capacity = capacity
        self.labels = labels
        self._label_indices = {value: index for index, value in enumerate(labels)}
        self.reads = 0
        self.writes = 0
        self.evictions = 0
        self.resident_entries = 0
        self.persistent_payload_bytes = 0
        self.peak_persistent_payload_bytes = 0
        self.fixed_reader_bytes = 0
        self.reader = None
        self._cpu_ns = 0
        self._wall_ns = 0

    def observe(self, text):
        cpu_start = time.process_time_ns()
        wall_start = time.perf_counter_ns()
        event = parse_observation(text, self.labels)
        if event is not None:
            key, value = event
            self._write(key, self._label_indices[value])
            self.writes += 1
            self.peak_persistent_payload_bytes = max(
                self.peak_persistent_payload_bytes, self.persistent_payload_bytes,
            )
        self._wall_ns += time.perf_counter_ns() - wall_start
        self._cpu_ns += time.process_time_ns() - cpu_start
        return event is not None

    def answer(self, query_text):
        cpu_start = time.process_time_ns()
        wall_start = time.perf_counter_ns()
        answer = self.peek(query_text)
        self.reads += 1
        self._wall_ns += time.perf_counter_ns() - wall_start
        self._cpu_ns += time.process_time_ns() - cpu_start
        return answer

    def peek(self, query_text):
        """Check an active query without changing resident memory or counters."""
        index = self._read(parse_query(query_text))
        return None if index is None else self.labels[index]

    def metrics(self):
        return {
            'capacity': self.capacity,
            'reads': self.reads,
            'writes': self.writes,
            'evictions': self.evictions,
            'resident_entries': self.resident_entries,
            'persistent_payload_bytes': self.persistent_payload_bytes,
            'peak_persistent_payload_bytes': self.peak_persistent_payload_bytes,
            'payload_accounting': self.payload_accounting,
            'fixed_reader_bytes': self.fixed_reader_bytes,
            'reader': self.reader,
            'operation_cpu_seconds': self._cpu_ns / 1e9,
            'operation_wall_seconds': self._wall_ns / 1e9,
        }


class _ExactMemory(_Memory):
    payload_accounting = ('UTF-8 key bytes plus one byte per class ID, excluding Python objects, '
                          'ordered-dictionary overhead, instrumentation, and the fixed location vocabulary')

    def __init__(self, capacity, labels):
        super().__init__(capacity, labels)
        self._entries = OrderedDict()

    def _write(self, key, value):
        if key in self._entries:
            self._entries.move_to_end(key)
        else:
            if len(self._entries) == self.capacity:
                evicted, _ = self._entries.popitem(last=False)
                self.persistent_payload_bytes -= len(evicted.encode('utf-8')) + 1
                self.evictions += 1
            self.persistent_payload_bytes += len(key.encode('utf-8')) + 1
        self._entries[key] = value
        self.resident_entries = len(self._entries)

    def _read(self, key):
        return self._entries.get(key)


class _SlotMemory(_Memory):
    payload_accounting = ('allocated fp32 state tensor, excluding temporary tensors, '
                          'runtime overhead, instrumentation, and the fixed reader and location vocabulary')

    def __init__(self, capacity, labels, checkpoint):
        super().__init__(capacity, labels)
        reader, self.reader = _load_reader(checkpoint)
        self._slots = DynamicContentSlots(reader, capacity, threshold=SLOT_THRESHOLD)
        self._state = self._slots.initial(1)
        self.fixed_reader_bytes = sum(p.numel() * p.element_size() for p in reader.parameters())
        self.persistent_payload_bytes = self._state.numel() * self._state.element_size()
        self.peak_persistent_payload_bytes = self.persistent_payload_bytes

    @torch.inference_mode()
    def _write(self, key, value):
        vector = _key_vector(key)
        # The slot API returns state only. At full occupancy, count unmatched writes as evictions.
        evicted = False
        if self.resident_entries == self.capacity:
            _, score, _ = self._slots._match(self._state, vector)
            evicted = score.item() < self._slots.threshold
        self._state = self._slots.observe(self._state, vector, torch.tensor([value], dtype=torch.int64, device='cpu'))
        self.evictions += int(evicted)
        if self.resident_entries < self.capacity:
            self.resident_entries = int(torch.count_nonzero(self._state[0, :, 24]))

    @torch.inference_mode()
    def _read(self, key):
        index = int(self._slots.query(self._state, _key_vector(key)).argmax())
        return None if index == len(self.labels) else index


def make_memory(arm, capacity=DEFAULT_CAPACITY, labels=DEFAULT_LABELS, checkpoint=None):
    if arm not in ARMS:
        raise ValueError(f'unknown bounded-memory arm {arm!r}')
    if type(capacity) is not int or capacity < 1:
        raise ValueError('positive integer capacity required')
    labels = tuple(labels)
    if (len(labels) != 8 or any(not isinstance(label, str) or not _LABEL.fullmatch(label) for label in labels)
            or len(set(labels)) != 8):
        raise ValueError('eight distinct location labels required')
    if arm == 'pbs_slots':
        return _SlotMemory(capacity, labels, checkpoint)
    return _ExactMemory(None if arm == 'exact_unbounded' else capacity, labels)
