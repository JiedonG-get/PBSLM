"""Regenerate the frozen synthetic Chinese bounded-state benchmark."""

import argparse
import json
from pathlib import Path
from random import Random

from evaluation.continuous_event_protocol import FORMAT, validate


THEMES = (
    '织造间', '复印室', '画框铺', '社区展厅', '制版室', '陶土坊',
    '录音棚', '装订间', '校样室', '借阅站', '标本室', '工艺社',
)
ITEMS = tuple(f'{series}{number}号样件' for series in '甲乙丙丁戊己庚辛壬癸' for number in '一二三')
LOCATIONS = (
    '松间厅', '雨燕厅', '石桥厅', '东练功房', '竹溪站', '江柳门', '月湾站', '澄光柜',
)
BACKGROUND = (
    '窗台上的温度计已经换过电池。', '走廊的灯刚亮起。', '楼梯口的时钟快了两分钟。',
    '门口新增了一张访客登记表。', '桌上有一杯冷掉的茶。', '值班人员打开了排风扇。',
    '有人把空信封放在了窗边。', '广播提示午休即将结束。', '墙上的月历翻到了下一页。',
)
SEED = 20261118


def generate_stream(number, theme):
    random = Random(SEED + number)
    items = list(ITEMS)
    notices = list(BACKGROUND)
    random.shuffle(items)
    random.shuffle(notices)
    keys = [f'{theme}{item}' for item in items]
    confirmed = {}
    events = []
    clock = 0

    def append(kind, text, oracle=None):
        nonlocal clock
        tick = len(events) + 1
        if events:
            clock += random.randint(0, 35)
        event = {'event_id': f'e{tick:02}', 'source_tick': tick,
                 'deliver_after_ms': clock, 'kind': kind, 'text': text}
        if oracle is not None:
            event['oracle'] = oracle
        events.append(event)

    def confirm(index):
        key = keys[index]
        choices = [location for location in LOCATIONS if location != confirmed.get(key)]
        value = random.choice(choices)
        confirmed[key] = value
        append('observation', f'确认：{key}的存放位置是{value}。')

    def rumor(index):
        key = keys[index]
        value = random.choice([location for location in LOCATIONS if location != confirmed.get(key)])
        append('observation', f'传闻：{key}的存放位置是{value}。')

    def background():
        append('observation', f'背景：{notices.pop()}')

    def query(index, case):
        key = keys[index]
        append('query', f'请问{key}的当前存放位置？',
               {'target': confirmed.get(key, '不确定'), 'case': case})

    confirm(0)
    query(0, 'initial')
    background()
    rumor(0)
    for index in range(1, 14):
        confirm(index)
        if index in (3, 7, 11):
            background()
    query(1, 'retention')
    rumor(2)
    query(2, 'rumor')
    confirm(0)
    query(0, 'corrected')
    rumor(24)
    query(24, 'unknown')

    for index in range(14, 24):
        confirm(index)
        if index in (15, 18, 21):
            background()
        if index in (16, 17, 20):
            rumor(index - 12)
    query(3, 'retention')
    query(8, 'retention')
    confirm(1)
    query(1, 'corrected')
    rumor(1)
    background()
    query(1, 'rumor')

    for index in (0, 22, 0, 22, 0):
        confirm(index)
    query(0, 'corrected')
    rumor(25)
    background()
    query(5, 'retention')
    query(26, 'unknown')

    return validate({'format': FORMAT, 'stream_id': f'bounded-online-{number:02}',
                     'split': 'primary' if number <= 9 else 'replica', 'events': events})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        for number, theme in enumerate(THEMES, 1):
            stream = generate_stream(number, theme)
            handle.write(json.dumps(stream, ensure_ascii=False, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
