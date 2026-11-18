"""Deterministic entity, scoped-alias, and tri-state location memory."""

import json
import re
import unicodedata


STATUS_LOCATED = 'located'
STATUS_NO_LOCATION = 'no_location'
STATUS_MISSING = 'missing'

OPERATIONS = {'set', 'clear', 'register', 'bind', 'revoke', 'rebind'}
CARRYOVER_PHRASES = ('刚问的', '刚才问的', '上个问题', '上一次问的', '刚问那件',
                     '刚才那件', '上一条问的', '接着上一条问的', '接着问那')


def normalize_text(text):
    return unicodedata.normalize('NFKC', text).casefold().strip()


def contains_mention(text, mention):
    literal = re.escape(mention)
    start = r'(?<![a-z0-9])' if re.match(r'[a-z0-9]', mention) else ''
    end = r'(?![a-z0-9])' if re.search(r'[a-z0-9]$', mention) else ''
    return re.search(start + literal + end, text) is not None


def identifiers(text):
    return set(re.findall(r'(?<![a-z0-9])(?:[a-z]+\d+[a-z0-9]*|\d+[a-z]+\d*)(?![a-z0-9])',
                          normalize_text(text)))


class ScopedState:
    """Store exact names and scoped aliases with explicit absence semantics.

    Rows are (operation, surface, value, scope). For bind/rebind, ``surface``
    is the registered canonical name and ``value`` is the alias. Revoke accepts
    either an alias in ``surface`` or a canonical name in ``surface`` and alias
    in ``value``. A null/empty scope denotes a global binding.
    """

    def __init__(self):
        self.entities = {}
        self.entity_events = {}
        self.aliases = {}
        self.facts = {}
        self.previous_entity = None

    def _entity_key(self, surface):
        return normalize_text(surface)

    def _identifier_surface(self, surface):
        normalized = self._entity_key(surface)
        candidate_ids = identifiers(surface)
        if len(candidate_ids) != 1:
            return None
        identifier = next(iter(candidate_ids))
        if normalized not in {identifier, identifier + '号'}:
            return None
        return identifier

    def _canonical_id_matches(self, identifier):
        return [entity for entity in self.entities if identifier in identifiers(entity)]

    def _resolve_canonical(self, surface):
        key = self._entity_key(surface)
        if key in self.entities:
            return key
        identifier = self._identifier_surface(surface)
        if identifier is None:
            return None
        matches = self._canonical_id_matches(identifier)
        return matches[0] if len(matches) == 1 else None

    def _ensure_entity(self, surface, event_id):
        key = self._entity_key(surface)
        if key not in self.entities:
            self.entities[key] = surface.strip()
            self.entity_events[key] = [event_id]
            self.facts[key] = {
                'status': STATUS_MISSING,
                'value': None,
                'events': [event_id],
            }
        return key

    def _record_entity_event(self, key, event_id):
        if event_id not in self.entity_events[key]:
            self.entity_events[key].append(event_id)
        if event_id not in self.facts[key]['events']:
            self.facts[key]['events'].append(event_id)

    @staticmethod
    def _clean_scope(scope):
        if scope is None:
            return None
        if not isinstance(scope, str):
            return None
        cleaned = normalize_text(scope)
        return cleaned or None

    def _active_bindings(self, alias):
        return [binding for binding in self.aliases.get(alias, ()) if binding['active']]

    def _resolve_surface(self, surface, scope):
        canonical = self._resolve_canonical(surface)
        if canonical is not None:
            return canonical
        key = self._entity_key(surface)
        bindings = self._active_bindings(key)
        requested_scope = self._clean_scope(scope)
        if requested_scope is not None:
            scoped = [binding for binding in bindings if binding['scope'] == requested_scope]
            global_bindings = [binding for binding in bindings if binding['scope'] is None]
            applicable = scoped or global_bindings
            targets = {binding['entity'] for binding in applicable}
            return next(iter(targets)) if len(targets) == 1 else None
        if len(bindings) > 1:
            targets = {binding['entity'] for binding in bindings}
            if len(targets) > 1:
                return None
        global_bindings = [binding for binding in bindings if binding['scope'] is None]
        targets = {binding['entity'] for binding in global_bindings}
        return next(iter(targets)) if len(targets) == 1 else None

    def apply(self, rows, event_id):
        applied = []
        rejected = []
        if not isinstance(event_id, str) or not event_id:
            raise ValueError('event_id must be a nonempty string')

        def reject(row, reason):
            rejected.append({'row': list(row) if isinstance(row, (tuple, list)) else row,
                             'reason': reason})

        for row in rows:
            if (not isinstance(row, (tuple, list)) or len(row) != 4
                    or not isinstance(row[0], str) or not isinstance(row[1], str)
                    or not isinstance(row[2], str)
                    or (row[3] is not None and not isinstance(row[3], str))):
                reject(row, 'invalid operation shape')
                continue
            operation, surface, value, raw_scope = row
            surface, value = surface.strip(), value.strip()
            scope = self._clean_scope(raw_scope)
            valid_value = (operation in {'set', 'bind', 'rebind'} and bool(value)
                           or operation == 'register' and not value
                           or operation == 'clear'
                           or operation == 'revoke')
            if operation not in OPERATIONS or not surface or not valid_value:
                reject(row, 'invalid operation or arguments')
                continue

            if operation == 'register':
                key = self._resolve_canonical(surface)
                identifier = self._identifier_surface(surface)
                if key is None and identifier is not None and self._canonical_id_matches(identifier):
                    reject(row, 'short identifier is ambiguous')
                    continue
                key = key or self._ensure_entity(surface, event_id)
                self._record_entity_event(key, event_id)
            elif operation in {'set', 'clear'}:
                key = self._resolve_surface(surface, scope)
                if key is None:
                    reject(row, 'entity surface is unknown or ambiguous')
                    continue
                fact = self.facts[key]
                if operation == 'clear' and value and (
                        fact['status'] != STATUS_LOCATED or fact['value'] != value):
                    reject(row, 'clear guard does not match current location')
                    continue
                fact['status'] = STATUS_LOCATED if operation == 'set' else STATUS_NO_LOCATION
                fact['value'] = value if operation == 'set' else None
                fact['events'] = list(dict.fromkeys((*fact['events'], event_id)))
            elif operation in {'bind', 'rebind'}:
                key = self._resolve_canonical(surface)
                if key is None:
                    reject(row, 'canonical entity is not registered')
                    continue
                alias = self._entity_key(value)
                if alias == key:
                    reject(row, 'alias duplicates canonical name')
                    continue
                bindings = self.aliases.setdefault(alias, [])
                if operation == 'rebind':
                    for binding in bindings:
                        if binding['active'] and binding['scope'] == scope:
                            binding['active'] = False
                            binding['revoked_by'] = event_id
                else:
                    conflicts = [binding for binding in bindings
                                 if binding['active'] and binding['scope'] == scope
                                 and binding['entity'] != key]
                    if conflicts:
                        reject(row, 'alias already bound in this scope; use rebind')
                        continue
                if not any(binding['active'] and binding['entity'] == key
                           and binding['scope'] == scope for binding in bindings):
                    bindings.append({'entity': key, 'scope': scope, 'active': True,
                                     'event_id': event_id, 'revoked_by': None})
            else:  # revoke
                if value:
                    canonical = self._resolve_canonical(surface)
                    if canonical is None:
                        reject(row, 'canonical entity is not registered')
                        continue
                    alias = self._entity_key(value)
                    matches = [binding for binding in self._active_bindings(alias)
                               if binding['entity'] == canonical
                               and binding['scope'] == scope]
                else:
                    alias = self._entity_key(surface)
                    matches = [binding for binding in self._active_bindings(alias)
                               if binding['scope'] == scope]
                if not matches:
                    reject(row, 'no active alias binding to revoke')
                    continue
                for binding in matches:
                    binding['active'] = False
                    binding['revoked_by'] = event_id
            applied.append([operation, surface, value, scope])
        return applied, rejected

    def _scope_from_question(self, question):
        normalized = normalize_text(question)
        matching = [scope for bindings in self.aliases.values() for binding in bindings
                    if binding['active'] and binding['scope'] is not None
                    for scope in (binding['scope'],) if contains_mention(normalized, scope)]
        return set(matching)

    def _name_targets(self, question):
        normalized = normalize_text(question)
        scopes = self._scope_from_question(question)
        candidates = {}
        ambiguous_alias = False
        ambiguous_alias_events = set()
        known_ids = {}
        for key in self.entities:
            for identifier in identifiers(key):
                known_ids.setdefault(identifier, set()).add(key)
        explicit_ids = identifiers(question)
        explicit_targets = set().union(*(known_ids.get(item, set()) for item in explicit_ids))
        explicit_unknown = explicit_ids - known_ids.keys()

        for key in self.entities:
            if contains_mention(normalized, key):
                candidates.setdefault(key, {'targets': set(), 'events': set(), 'surface': key})
                candidates[key]['targets'].add(key)
                candidates[key]['events'].update(self.entity_events[key])
            if key in explicit_targets:
                candidates.setdefault(key, {'targets': set(), 'events': set(), 'surface': key})
                candidates[key]['targets'].add(key)
                candidates[key]['events'].update(self.entity_events[key])

        for alias, bindings in self.aliases.items():
            active = [binding for binding in bindings if binding['active']]
            if not contains_mention(normalized, alias):
                continue
            scoped = [binding for binding in active if binding['scope'] in scopes]
            if scoped:
                selected = scoped
            else:
                selected = [binding for binding in active if binding['scope'] is None]
            if not selected:
                if len({binding['entity'] for binding in active}) > 1:
                    ambiguous_alias = True
                    ambiguous_alias_events.update(binding['event_id'] for binding in active)
                continue
            entry = candidates.setdefault(alias, {'targets': set(), 'events': set(), 'surface': alias})
            entry['targets'].update(binding['entity'] for binding in selected)
            entry['events'].update(binding['event_id'] for binding in selected)

        matched_names = list(candidates)
        longest = [name for name in matched_names
                   if not any(name != other and name in other for other in matched_names)]
        target_events = {}
        for name in longest:
            entry = candidates[name]
            for target in entry['targets']:
                target_events.setdefault(target, set()).update(entry['events'])
        return (target_events, scopes, explicit_targets, explicit_unknown,
                ambiguous_alias, ambiguous_alias_events)

    def lookup(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('question must be a nonempty string')
        (targets, scopes, explicit_targets, explicit_unknown, ambiguous_alias,
         ambiguous_alias_events) = self._name_targets(question)
        normalized = normalize_text(question)
        carryover = any(phrase in normalized for phrase in CARRYOVER_PHRASES)
        conflict = (len(explicit_targets) > 1 or bool(explicit_unknown)
                    or (explicit_targets and set(targets) - explicit_targets)
                    or (ambiguous_alias and not explicit_targets))
        if conflict:
            key = None
            method = 'ambiguous' if explicit_targets or targets or ambiguous_alias else 'unknown'
            evidence = set().union(*targets.values()) if targets else set()
            evidence.update(ambiguous_alias_events)
            entity_status = 'ambiguous' if method == 'ambiguous' else 'unknown'
        elif len(targets) == 1:
            key = next(iter(targets))
            method = 'scoped_alias' if any(
                contains_mention(normalized, alias)
                and binding['active'] and binding['entity'] == key
                and binding['scope'] is not None and binding['scope'] in scopes
                for alias, rows in self.aliases.items() for binding in rows) else 'literal'
            evidence = set(targets[key])
            entity_status = 'resolved'
        elif not targets and not ambiguous_alias and carryover and self.previous_entity in self.entities:
            key = self.previous_entity
            method = 'carryover'
            evidence = set()
            entity_status = 'resolved'
        else:
            key = None
            method = 'ambiguous' if targets or ambiguous_alias else 'unknown'
            evidence = set().union(*targets.values()) if targets else set()
            entity_status = method

        if key is None:
            result = {'entity': None, 'readout': None, 'status': STATUS_MISSING,
                      'location_status': None, 'entity_status': entity_status,
                      'evidence_event_ids': sorted(evidence), 'link_method': method}
        else:
            fact = self.facts[key]
            evidence.update(fact['events'])
            result = {'entity': self.entities[key], 'readout': fact['value'],
                      'status': fact['status'], 'location_status': fact['status'],
                      'entity_status': 'resolved', 'evidence_event_ids': sorted(evidence),
                      'link_method': method}
        self.previous_entity = key
        return result

    def payload_bytes(self):
        value = {'entities': self.entities, 'entity_events': self.entity_events,
                 'aliases': self.aliases, 'facts': self.facts,
                 'previous_entity': self.previous_entity}
        return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
