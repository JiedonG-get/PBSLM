"""Structured location memory driven by a frozen text extractor."""

from collections import defaultdict
import json
import re
import unicodedata


UNKNOWN = '不确定'
EXTRACT_TOKENS = 112
LINK_TOKENS = 32
EXTRACT_PROMPT = (
    '仅输出 JSON 数组。物件名、地点和别名照抄记录中最短的完整叫法，不带引号。'
    '只记录确认的物件存放地点、明确作废的存放记录和明确的别名关系。'
    '输出格式：[ ["set",物件,地点], ["clear",物件,""], ["alias",原叫法,新叫法] ]。'
    '在别名句中，原叫法在前，新叫法在后，即使原文先提新叫法。'
    '样例：维修班说的“小八”就是设备D8。 → [["alias","设备D8","小八"]]\n'
    '样例：设备D8已放进档案柜。 → [["set","设备D8","档案柜"]]\n'
    '样例：还没把设备D8移到器具室，档案柜的记录仍有效。 → [["set","设备D8","档案柜"]]\n'
    '样例：注销档案柜内设备D8的存放记录，去向待查。 → [["clear","设备D8",""]]\n'
    '样例：明天拟把设备D8搬到器具室，尚未实施。 → []\n'
    '样例：档案柜旁的指示灯已经更换。 → []\n'
    '只否认某处也输出 []，除非这条记录明确注销一个确认地点。'
)
LINK_PROMPT = (
    '从候选表识别当前问题里的物件。表中每项依次是[选项序号,物件全名,别名列表]。'
    '只输出选项序号，找不到唯一匹配则输出0。核对物件编号的每个字符，忽略语序。'
    '如候选为[[1,"导管R7",[]],[2,"导管R71",[]]]，“R71这根导管”选2，“R17的导管”选0。'
    '明确问“刚问的那个”时，匹配上个问题已识别出的物件。新简称需在别名列表中。'
    '同一别名指多个物件且缺少区分信息时输出0。'
)


def normalize_text(text):
    return unicodedata.normalize('NFKC', text).casefold()


def contains_mention(text, mention):
    literal = re.escape(mention)
    start = r'(?<![a-z0-9])' if re.match(r'[a-z0-9]', mention) else ''
    end = r'(?![a-z0-9])' if re.search(r'[a-z0-9]$', mention) else ''
    return re.search(start + literal + end, text) is not None


def messages(system, content):
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]


def parse_updates(raw, source):
    """Reject malformed or nonliteral operations individually."""
    cleaned = re.sub(r'\A```(?:json)?\s*|\s*```\Z', '', raw.strip())
    try:
        rows = json.loads(cleaned)
    except (ValueError, TypeError):
        return [], ['invalid JSON']
    if not isinstance(rows, list):
        return [], ['not an array']
    accepted = []
    rejected = []
    normalized = normalize_text(source)
    for row in rows:
        if not isinstance(row, list) or len(row) != 3 or any(not isinstance(s, str) for s in row):
            rejected.append('invalid operation shape')
            continue
        operation, entity, value = (s.strip() for s in row)
        if (operation not in {'set', 'alias', 'clear'} or not entity
                or (operation == 'clear') != (not value)):
            rejected.append('invalid operation or arguments')
            continue
        if any(not contains_mention(normalized, normalize_text(s)) for s in (entity, value) if s):
            rejected.append('argument absent from current observation')
            continue
        accepted.append((operation, entity, value))
    return accepted, rejected


class AliasState:
    def __init__(self):
        self.entities = {}
        self.aliases = defaultdict(dict)
        self.facts = {}
        self.previous_question = None
        self.previous_entity = None

    def apply(self, rows, event_id):
        applied = []
        rejected = []
        for operation, surface, value in rows:
            name = normalize_text(surface)
            aliases = self.aliases.get(name, {})
            if name in self.entities:
                key = name
                dependencies = ()
            elif len(aliases) == 1:
                key, dependencies = next(iter(aliases.items()))
            elif aliases:
                rejected.append('operation refers to an ambiguous alias')
                continue
            else:
                key = name
                dependencies = ()
                self.entities[key] = surface
            provenance = tuple(dict.fromkeys((*dependencies, event_id)))
            if operation == 'alias':
                alias = normalize_text(value)
                if alias != key:
                    self.aliases[alias][key] = provenance
            else:
                self.facts[key] = (value if operation == 'set' else None, provenance)
            applied.append([operation, surface, value])
        return applied, rejected

    def _direct(self, text):
        normalized = normalize_text(text)
        names = {name: {name: ()} for name in self.entities}
        for alias, targets in self.aliases.items():
            names.setdefault(alias, {}).update(targets)
        matching = [name for name in names if contains_mention(normalized, name)]
        longest = [name for name in matching if not any(name != other and name in other for other in matching)]
        matches = {}
        for name in longest:
            matches.update(names[name])
        return matches

    def lookup(self, question, call):
        matches = self._direct(question)
        key = None
        provenance = ()
        method = 'no_candidates'
        if len(matches) == 1:
            key, provenance = next(iter(matches.items()))
            method = 'literal'
        elif self.entities:
            candidates = list(matches or self.entities)
            catalog = []
            for index, candidate in enumerate(candidates, 1):
                aliases = [alias for alias, targets in self.aliases.items() if candidate in targets]
                catalog.append([index, self.entities[candidate], aliases])
            previous = {'question': self.previous_question, 'entity': self.previous_entity}
            content = json.dumps({'candidates': catalog, 'previous': previous, 'question': question},
                                 ensure_ascii=False)
            raw = call('link', messages(LINK_PROMPT, content), LINK_TOKENS).strip()
            method = 'model'
            if re.fullmatch(r'[1-9][0-9]*', raw) and int(raw) <= len(candidates):
                key = candidates[int(raw) - 1]
                provenance = matches.get(key, ())
        self.previous_question = question
        self.previous_entity = self.entities.get(key)
        readout, fact_events = self.facts.get(key, (None, ()))
        return self.previous_entity, readout, list(dict.fromkeys((*provenance, *fact_events))), method

    def payload_bytes(self):
        value = [self.entities, dict(self.aliases), self.facts, self.previous_question, self.previous_entity]
        return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
