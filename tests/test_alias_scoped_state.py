import unittest

from evaluation.alias_scoped_state import (
    STATUS_LOCATED,
    STATUS_MISSING,
    STATUS_NO_LOCATION,
    ScopedState,
)


class ScopedStateTests(unittest.TestCase):
    def test_distinguishes_missing_no_location_and_located(self):
        state = ScopedState()
        state.apply([('register', '工具箱A14', '', None)], 'e1')
        initial = state.lookup('工具箱A14在哪里？')
        self.assertEqual(initial['status'], STATUS_MISSING)
        self.assertEqual(initial['location_status'], STATUS_MISSING)
        self.assertEqual(initial['entity_status'], 'resolved')
        state.apply([('clear', '工具箱A14', '', None)], 'e2')
        absent = state.lookup('工具箱A14在哪里？')
        self.assertEqual(absent['status'], STATUS_NO_LOCATION)
        self.assertEqual(absent['evidence_event_ids'], ['e1', 'e2'])
        state.apply([('set', '工具箱A14', '南墙柜', None)], 'e3')
        located = state.lookup('工具箱A14在哪里？')
        self.assertEqual((located['status'], located['readout']), (STATUS_LOCATED, '南墙柜'))
        self.assertEqual(located['evidence_event_ids'], ['e1', 'e2', 'e3'])

    def test_scoped_alias_requires_and_uses_explicit_scope(self):
        state = ScopedState()
        state.apply([('register', '设备R17', '', None), ('register', '设备R71', '', None)], 'e1')
        state.apply([('set', '设备R17', '北柜', None), ('set', '设备R71', '南柜', None)], 'e2')
        state.apply([('bind', '设备R17', '小扳手', '早班'),
                     ('bind', '设备R71', '小扳手', '晚班')], 'e3')
        self.assertEqual(state.lookup('早班的小扳手在哪？')['entity'], '设备R17')
        self.assertEqual(state.lookup('晚班的小扳手在哪？')['entity'], '设备R71')
        self.assertEqual(state.lookup('小扳手在哪？')['link_method'], 'ambiguous')

    def test_revoke_and_rebind_remove_old_owner(self):
        state = ScopedState()
        state.apply([('register', '绢录盒J03', '', None), ('register', '绢录盒J30', '', None)], 'e1')
        state.apply([('set', '绢录盒J03', '描页柜', None), ('set', '绢录盒J30', '签帖匣', None)], 'e2')
        state.apply([('bind', '绢录盒J30', '压题盒', '修编组')], 'e3')
        state.apply([('rebind', '绢录盒J03', '压题盒', '修编组')], 'e4')
        self.assertEqual(state.lookup('修编组的压题盒在哪？')['entity'], '绢录盒J03')
        self.assertEqual(state.lookup('绢录盒J30的地点？')['readout'], '签帖匣')

    def test_rebind_preserves_old_owner_under_other_scope(self):
        state = ScopedState()
        state.apply([('register', '物件A1', '', None), ('register', '物件A2', '', None)], 'e1')
        state.apply([('bind', '物件A1', '白盒', '甲组'), ('bind', '物件A2', '白盒', '乙组')], 'e2')
        state.apply([('rebind', '物件A2', '白盒', '甲组')], 'e3')
        self.assertEqual(state.lookup('甲组的白盒')['entity'], '物件A2')
        self.assertEqual(state.lookup('乙组的白盒')['entity'], '物件A2')

    def test_near_identifier_does_not_match_or_resolve_by_carryover(self):
        state = ScopedState()
        state.apply([('register', '量规H31', '', None)], 'e1')
        state.apply([('set', '量规H31', '幕架底柜', None)], 'e2')
        self.assertEqual(state.lookup('H13号量规在哪？')['link_method'], 'unknown')
        state.lookup('H31号量规在哪？')
        self.assertEqual(state.lookup('刚问的那件放在哪？')['link_method'], 'carryover')

    def test_explicit_unknown_and_global_revoke(self):
        state = ScopedState()
        state.apply([('register', '缆扣B4', '', None)], 'e1')
        state.apply([('bind', '缆扣B4', '四号扣', None)], 'e2')
        state.apply([('revoke', '四号扣', '', None)], 'e3')
        result = state.lookup('四号扣在哪？')
        self.assertEqual((result['entity'], result['status']), (None, STATUS_MISSING))

    def test_global_revoke_leaves_scoped_binding_intact(self):
        state = ScopedState()
        state.apply([('register', '缆扣B4', '', None),
                     ('register', '缆扣B5', '', None)], 'e1')
        state.apply([('bind', '缆扣B4', '四号扣', None),
                     ('bind', '缆扣B5', '四号扣', '晚班')], 'e2')
        state.apply([('revoke', '四号扣', '', None)], 'e3')
        self.assertEqual(state.lookup('晚班的四号扣在哪？')['entity'], '缆扣B5')
        self.assertEqual(state.lookup('四号扣在哪？')['entity_status'], 'unknown')

    def test_rejects_conflicting_bind_without_mutation(self):
        state = ScopedState()
        state.apply([('register', '封箱器C1', '', None), ('register', '封箱器C2', '', None)], 'e1')
        state.apply([('bind', '封箱器C1', '小蓝', '白班')], 'e2')
        applied, rejected = state.apply([('bind', '封箱器C2', '小蓝', '白班')], 'e3')
        self.assertEqual(applied, [])
        self.assertEqual(rejected[0]['reason'], 'alias already bound in this scope; use rebind')
        self.assertEqual(rejected[0]['row'], ['bind', '封箱器C2', '小蓝', '白班'])
        self.assertEqual(state.lookup('白班的小蓝')['entity'], '封箱器C1')

    def test_literal_name_wins_over_near_id_and_payload_is_stable(self):
        state = ScopedState()
        state.apply([('register', '导管R7', '', None), ('register', '导管R71', '', None)], 'e1')
        state.apply([('set', '导管R7', '旧架', None), ('set', '导管R71', '新架', None)], 'e2')
        self.assertEqual(state.lookup('R71号导管在哪？')['readout'], '新架')
        self.assertGreater(state.payload_bytes(), 0)
        self.assertEqual(state.payload_bytes(), state.payload_bytes())

    def test_reordered_identifier_resolves_exact_entity(self):
        state = ScopedState()
        state.apply([('register', '导管R7', '', None),
                     ('register', '导管R71', '', None)], 'e1')
        state.apply([('set', '导管R7', '旧架', None),
                     ('set', '导管R71', '新架', None)], 'e2')
        result = state.lookup('R71号导管在哪？')
        self.assertEqual((result['entity'], result['readout']), ('导管R71', '新架'))

    def test_scoped_alias_without_scope_is_ambiguous_and_blocks_carryover(self):
        state = ScopedState()
        state.apply([('register', '设备R17', '', None),
                     ('register', '设备R71', '', None)], 'e1')
        state.apply([('bind', '设备R17', '小扳手', '早班'),
                     ('bind', '设备R71', '小扳手', '晚班')], 'e2')
        self.assertEqual(state.lookup('早班的小扳手在哪？')['entity'], '设备R17')
        ambiguous = state.lookup('刚问的那个小扳手在哪？')
        self.assertIsNone(ambiguous['entity'])
        self.assertEqual(ambiguous['entity_status'], 'ambiguous')
        self.assertEqual(ambiguous['location_status'], None)
        self.assertEqual(ambiguous['evidence_event_ids'], ['e2'])

    def test_alias_can_update_location_and_stale_clear_is_rejected(self):
        state = ScopedState()
        state.apply([('register', '设备R17', '', None)], 'e1')
        state.apply([('bind', '设备R17', '小扳手', '早班')], 'e2')
        applied, rejected = state.apply([('set', '小扳手', '北柜', '早班')], 'e3')
        self.assertEqual(applied, [['set', '小扳手', '北柜', '早班']])
        self.assertEqual(rejected, [])
        applied, rejected = state.apply([('clear', '设备R17', '旧柜', None)], 'e4')
        self.assertEqual(applied, [])
        self.assertEqual(rejected[0]['reason'], 'clear guard does not match current location')
        self.assertEqual(state.lookup('设备R17在哪？')['readout'], '北柜')

    def test_conflicting_explicit_identifier_blocks_alias_and_carryover(self):
        state = ScopedState()
        state.apply([('register', '工具R17', '', None),
                     ('register', '工具R71', '', None)], 'e1')
        state.apply([('bind', '工具R17', '小扳手', None)], 'e2')
        self.assertEqual(state.lookup('工具R71在哪里？')['entity'], '工具R71')
        result = state.lookup('刚才问的R71号小扳手在哪？')
        self.assertIsNone(result['entity'])
        self.assertEqual(result['entity_status'], 'ambiguous')

    def test_short_identifier_reuses_canonical_for_all_operations(self):
        state = ScopedState()
        state.apply([('register', '应急照明器E42', '', None)], 'e1')
        state.apply([('set', 'E42', '北门箱', None)], 'e2')
        state.apply([('register', 'E42', '', None)], 'e3')
        state.apply([('bind', 'E42', '应急灯', '夜班')], 'e4')
        result = state.lookup('夜班的应急灯在哪？')
        self.assertEqual(result['entity'], '应急照明器E42')
        self.assertEqual(result['readout'], '北门箱')
        self.assertEqual(len(state.entities), 1)
        self.assertEqual(state.entities['应急照明器e42'], '应急照明器E42')
        applied, rejected = state.apply([('revoke', 'E42', '应急灯', '夜班')], 'e5')
        self.assertEqual(applied, [['revoke', 'E42', '应急灯', '夜班']])
        self.assertEqual(rejected, [])
        self.assertEqual(state.lookup('应急照明器E42在哪？')['readout'], '北门箱')

    def test_unambiguous_carryover_phrases(self):
        for phrase in ('接着上一条问的那片', '我接着问那只套盒'):
            state = ScopedState()
            state.apply([('register', '滤片F42', '', None)], 'e1')
            state.apply([('set', '滤片F42', '蓝屉', None)], 'e2')
            state.lookup('滤片F42在哪？')
            result = state.lookup(f'{phrase}现在放在哪？')
            self.assertEqual(result['entity'], '滤片F42', phrase)
            self.assertEqual(result['link_method'], 'carryover', phrase)


if __name__ == '__main__':
    unittest.main()
