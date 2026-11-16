import unittest
from runtime.source_order import SourceOrderedState


class SourceOrderTests(unittest.TestCase):
    def test_late_observation_cannot_revert_latest_source_state(self):
        state=SourceOrderedState()
        state.apply('new',source='camera',slot='lamp',sequence=20,value='green')
        late=state.apply('old',source='camera',slot='lamp',sequence=10,value='red')
        self.assertFalse(late['applied']);self.assertEqual(late['reason'],'late_ignored')
        self.assertEqual(state.view('lamp')['resolved_value'],'green')

    def test_idempotence_sequence_conflict_and_copy(self):
        state=SourceOrderedState();value={'color':'green'}
        receipt=state.apply('e',source='s',slot='lamp',sequence=2,value=value)
        value['color']='red';receipt['value']['color']='blue'
        duplicate=state.apply('e',source='s',slot='lamp',sequence=2,value={'color':'green'})
        self.assertEqual(duplicate['value'],{'color':'green'})
        with self.assertRaises(ValueError):state.apply('e',source='s',slot='lamp',sequence=2,value='red')
        with self.assertRaises(ValueError):state.apply('other',source='s',slot='lamp',sequence=2,value='red')

    def test_cross_source_sequences_not_comparable(self):
        state=SourceOrderedState()
        state.apply('a',source='camera',slot='lamp',sequence=999,value='red')
        state.apply('b',source='audio',slot='lamp',sequence=1,value='green')
        view=state.view('lamp');self.assertTrue(view['cross_source_conflict'])
        self.assertFalse(view['has_value']);self.assertIsNone(view['resolved_value'])
        state.apply('c',source='camera',slot='lamp',sequence=1000,value='green')
        self.assertEqual(state.view('lamp')['resolved_value'],'green')

    def test_limits_fail_without_dropping_watermark(self):
        state=SourceOrderedState(slot_limit=1,event_limit=2)
        state.apply('a',source='s',slot='lamp',sequence=5,value='green')
        with self.assertRaises(OverflowError):state.apply('b',source='x',slot='lamp',sequence=1,value='red')
        state.apply('late',source='s',slot='lamp',sequence=3,value='red')
        with self.assertRaises(OverflowError):state.apply('new',source='s',slot='lamp',sequence=6,value='blue')
        self.assertEqual(state.view('lamp')['resolved_value'],'green')


if __name__=='__main__':unittest.main()
