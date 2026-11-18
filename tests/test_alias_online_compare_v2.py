import unittest
import json

from evaluation.alias_online_compare_v2 import (
    _grounded_location,
    maintenance_needed,
    parse_operations,
    replay_scoped,
)
from evaluation.alias_scoped_state import ScopedState, normalize_text


def _observations(*rows):
    return [{'event_id': event_id, 'text': text} for event_id, text in rows]


class AliasOnlineV2ParsingTests(unittest.TestCase):
    def test_a_named_object_without_an_identifier_can_gain_a_location(self):
        stream = {'stream_id': 'unnumbered', 'events': [
            {'event_id': 'e1', 'kind': 'observation', 'text': '票据夹夹在器材柜内侧。'},
            {'event_id': 'q1', 'kind': 'query', 'text': '票据夹在哪？'},
        ]}

        def generate(messages, *, max_new_tokens):
            return '[["e1","set","票据夹","器材柜内侧",""]]', {'input_tokens': 1, 'output_tokens': 1}

        predictions, _, trace = replay_scoped(generate, stream)
        self.assertEqual(trace['calls'][0]['observation_event_ids'], ['e1'])
        self.assertEqual(predictions[0]['answer'], '器材柜内侧')

    def test_a_global_alias_in_the_same_event_is_not_registered_as_an_entity(self):
        stream = {'stream_id': 'global-alias', 'events': [
            {'event_id': 'e1', 'kind': 'observation',
             'text': '工具箱A14在各班组通用的称呼为小扳手，小扳手已放入北柜。'},
            {'event_id': 'q1', 'kind': 'query', 'text': '小扳手在哪？'},
        ]}

        def generate(messages, *, max_new_tokens):
            return ('[["e1","bind","工具箱A14","小扳手",""],'
                    '["e1","set","小扳手","北柜",""]]'), {
                        'input_tokens': 1, 'output_tokens': 1,
                    }

        predictions, _, trace = replay_scoped(generate, stream)
        self.assertEqual(predictions[0]['answer'], '北柜')
        self.assertEqual(trace['ingress'][0]['rejected_state'], [])

    def test_accepts_literal_updates_and_gates_scoped_aliases(self):
        source = '工具箱A14无存放地址，早班把“小扳手”作为工具箱A14的称呼。'
        raw = ('[["e1","register","工具箱A14","",""],'
               '["e1","bind","工具箱A14","小扳手","早班"]]')
        accepted, rejected = parse_operations(raw, _observations(('e1', source)))
        self.assertEqual(accepted, {'e1': [
            ('register', '工具箱A14', '', None),
            ('bind', '工具箱A14', '小扳手', '早班'),
        ]})
        self.assertEqual(rejected, [])

    def test_rejects_unscoped_alias_and_nonliteral_arguments(self):
        source = '小扳手是工具箱A14的别名。'
        raw = ('[["e1","bind","工具箱A14","小扳手",""],'
               '["e1","set","工具箱A14","北墙柜",""]]')
        accepted, rejected = parse_operations(raw, _observations(('e1', source)))
        self.assertFalse(any(accepted.values()))
        self.assertEqual(len(rejected), 2)
        self.assertEqual(rejected[0]['reason'], 'alias binding requires an explicit source scope')
        self.assertEqual(rejected[1]['reason'], 'argument absent from current observation')

    def test_rejects_unordered_contradictory_location_updates(self):
        source = '工具箱A14放在北墙柜，工具箱A14放在南窗柜。'
        raw = ('[["e1","set","工具箱A14","北墙柜",""],'
               '["e1","set","工具箱A14","南窗柜",""]]')
        accepted, rejected = parse_operations(raw, _observations(('e1', source)))
        self.assertFalse(any(accepted.values()))
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all(row['reason'].startswith('contradictory') for row in rejected))

    def test_clear_rejects_invalidation_cue_from_another_clause(self):
        source = '管理员取消物件E42的西楼梯箱存放登记，物件A14仍在北墙柜。'
        raw = '[["e1","clear","物件A14","北墙柜",""]]'
        accepted, rejected = parse_operations(raw, _observations(('e1', source)))
        self.assertEqual(accepted, {})
        self.assertEqual(rejected[0]['reason'], 'clear lacks a target-linked invalidation cue')

    def test_clear_accepts_location_before_entity_in_same_clause(self):
        source = '管理员取消西楼梯箱内物件E42的存放登记。'
        raw = '[["e1","clear","物件E42","西楼梯箱",""]]'
        accepted, rejected = parse_operations(raw, _observations(('e1', source)))
        self.assertEqual(accepted, {'e1': [('clear', '物件E42', '西楼梯箱', None)]})
        self.assertEqual(rejected, [])

    def test_requires_a_known_pending_event_id(self):
        observations = _observations(('e1', '工具箱A14已放入东侧柜。'))
        raw = '[ ["future","set","工具箱A14","东侧柜",""] ]'
        accepted, rejected = parse_operations(raw, observations)
        self.assertEqual(accepted, {})
        self.assertEqual(rejected[0]['reason'], 'event ID is outside the pending prefix')

    def test_grounding_accepts_confirmed_location_and_rejects_polarity_or_speculation(self):
        self.assertTrue(_grounded_location(normalize_text('工具箱A14已放入东侧柜。'),
                                          '工具箱A14', '东侧柜'))
        self.assertFalse(_grounded_location(normalize_text('工具箱A14没有放在东侧柜。'),
                                           '工具箱A14', '东侧柜'))
        self.assertFalse(_grounded_location(normalize_text('计划把工具箱A14放入东侧柜。'),
                                           '工具箱A14', '东侧柜'))

    def test_batches_only_observations_that_need_maintenance(self):
        stream = {
            'stream_id': 'batching',
            'events': [
                {'event_id': 'e1', 'kind': 'observation', 'text': '工具箱A14放在东侧柜。'},
                {'event_id': 'e2', 'kind': 'observation', 'text': '今天天气晴朗。'},
                {'event_id': 'e3', 'kind': 'observation', 'text': '工具箱A14后来放到南柜。'},
                {'event_id': 'q1', 'kind': 'query', 'text': '工具箱A14在哪里？'},
            ],
        }
        pending_observations = stream['events'][:3]
        memory = ScopedState()
        self.assertTrue(maintenance_needed(pending_observations[0]['text'], memory))
        self.assertFalse(maintenance_needed(pending_observations[1]['text'], memory))
        self.assertTrue(maintenance_needed(pending_observations[2]['text'], memory))
        batches = []

        def generate(messages, *, max_new_tokens):
            batch = json.loads(messages[-1]['content'])
            batches.append(batch)
            operations = []
            for row in batch:
                location = '东侧柜' if '东侧柜' in row['text'] else '南柜'
                operations.append([row['event_id'], 'set', '工具箱A14', location, ''])
            return json.dumps(operations, ensure_ascii=False), {
                'input_tokens': 1, 'output_tokens': 1,
            }

        predictions, costs, trace = replay_scoped(generate, stream)
        self.assertEqual(len(batches), 1)
        self.assertEqual([row['event_id'] for row in batches[0]], ['e1', 'e3'])
        self.assertEqual(costs['maintenance_calls'], 1)
        ingress = {row['event_id']: row for row in trace['ingress']}
        self.assertFalse(ingress['e1']['maintenance_skipped'])
        self.assertTrue(ingress['e2']['maintenance_skipped'])
        self.assertFalse(ingress['e3']['maintenance_skipped'])
        self.assertEqual(ingress['e1']['validated_operations'][0][0], 'register')
        self.assertEqual(ingress['e3']['validated_operations'][-1][2], '南柜')
        self.assertEqual(predictions[0]['readout'], '南柜')


if __name__ == '__main__':
    unittest.main()
