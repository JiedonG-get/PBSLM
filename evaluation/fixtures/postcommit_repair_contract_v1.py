"""Paired Chinese post-commit fixture, with labels reserved for offline scoring."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    identity: str
    text: str


@dataclass(frozen=True)
class Case:
    identity: str
    events: tuple[Event, ...]
    expectations: tuple[tuple[str, str | None, str | None], ...]


def scenario(identity, key, steps):
    events = []
    expectations = []
    for step, value, score in steps:
        event_id = f'e{len(events) + 1:02}'
        if step == 'query':
            text = f'请问{key}的当前存放位置？'
        elif step == 'confirmed':
            text = f'确认：{key}的存放位置是{value}。'
        elif step == 'rumor':
            text = f'传闻：{key}的存放位置是{value}。'
        elif step == 'unrelated':
            text = f'确认：{key}的邻近样件的存放位置是{value}。'
        elif step == 'background':
            text = f'背景：{value}'
        elif step == 'duplicate':
            previous = events[-1]
            event_id, text = previous.identity, previous.text
        else:
            raise ValueError(f'unknown fixture step {step!r}')
        events.append(Event(event_id, text))
        expectations.append(score)
    return Case(identity, tuple(events), tuple(expectations))


SILENCE = ('silent', None, None)


def initial(value):
    return ('initial', None, value)


def repair(old, new):
    return ('withdraw_and_replace', old, new)


def resolution(value):
    return ('resolve_unknown', None, value)


def pair(identity, key, old, new):
    return scenario(identity, key, (
        ('confirmed', old, SILENCE), ('query', None, initial(old)), ('confirmed', new, repair(old, new)),
    ))


DEVELOPMENT = (
    pair('dev_change', '玻璃工坊甲三号样件', '月湾站', '石桥厅'),
    scenario('dev_unknown', '档案室甲七号样件', (
        ('query', None, initial(None)), ('rumor', '澄光柜', SILENCE), ('confirmed', '松间厅', resolution('松间厅')),
    )),
)


FORMAL = (
    pair('straight_a', '天文角辛四号样件', '石桥厅', '江柳门'),
    pair('straight_b', '制伞坊丁六号样件', '松间厅', '雨燕厅'),
    pair('straight_c', '木模组癸二号样件', '澄光柜', '竹溪站'),
    pair('straight_d', '地质组乙五号样件', '东练功房', '月湾站'),
    scenario('idempotent_delivery', '校对室戊四号样件', (
        ('confirmed', '雨燕厅', SILENCE), ('query', None, initial('雨燕厅')),
        ('confirmed', '江柳门', repair('雨燕厅', '江柳门')), ('duplicate', None, SILENCE),
        ('confirmed', '江柳门', SILENCE), ('background', '门口刚收下新的快递。', SILENCE),
    )),
    scenario('unrelated_and_unchanged', '竹编社壬二号样件', (
        ('confirmed', '月湾站', SILENCE), ('query', None, initial('月湾站')),
        ('unrelated', '东练功房', SILENCE), ('confirmed', '月湾站', SILENCE),
        ('confirmed', '松间厅', repair('月湾站', '松间厅')),
    )),
    scenario('unconfirmed_then_confirmed', '雕刻室丙六号样件', (
        ('confirmed', '石桥厅', SILENCE), ('query', None, initial('石桥厅')),
        ('rumor', '澄光柜', SILENCE), ('unrelated', '竹溪站', SILENCE),
        ('confirmed', '东练功房', repair('石桥厅', '东练功房')),
    )),
    scenario('new_evidence', '微缩馆庚五号样件', (
        ('rumor', '松间厅', SILENCE), ('query', None, initial(None)),
        ('background', '落地钟刚刚校过时间。', SILENCE), ('rumor', '月湾站', SILENCE),
        ('confirmed', '江柳门', resolution('江柳门')),
        ('confirmed', '雨燕厅', repair('江柳门', '雨燕厅')),
    )),
    scenario('reversal', '模型房乙七号样件', (
        ('confirmed', '江柳门', SILENCE), ('query', None, initial('江柳门')),
        ('confirmed', '月湾站', repair('江柳门', '月湾站')), ('duplicate', None, SILENCE),
        ('confirmed', '江柳门', repair('月湾站', '江柳门')),
    )),
    scenario('successive_corrections', '拓印坊己六号样件', (
        ('confirmed', '松间厅', SILENCE), ('query', None, initial('松间厅')),
        ('confirmed', '石桥厅', repair('松间厅', '石桥厅')),
        ('rumor', '雨燕厅', SILENCE), ('confirmed', '东练功房', repair('石桥厅', '东练功房')),
    )),
)
