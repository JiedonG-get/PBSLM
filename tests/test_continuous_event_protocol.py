import json
from pathlib import Path
import unittest
from evaluation.continuous_event_protocol import load,system_event,FORMAT

ROOT=Path(__file__).resolve().parents[1]


class ContinuousEventProtocolTests(unittest.TestCase):
    def test_system_observation_has_no_oracle_or_future(self):
        fixture=load(ROOT/'evaluation/fixtures/continuous_event_dev.json')
        view=system_event(fixture,2,delivered_at_ms=25)
        self.assertEqual(view['event_id'],'e3')
        self.assertEqual([x['event_id'] for x in view['visible_prior_events']],['e1','e2'])
        serialized=json.dumps(view,ensure_ascii=False)
        for prohibited in ('oracle','current_target','图书馆门口','relevance'):
            self.assertNotIn(prohibited,serialized)

    def test_schema_order_and_id_validation(self):
        fixture=load(ROOT/'evaluation/fixtures/continuous_event_dev.json')
        bad=dict(fixture);bad['events']=list(reversed(fixture['events']))
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.json';path.write_text(json.dumps(bad),encoding='utf-8')
            with self.assertRaises(ValueError):load(path)
        with self.assertRaises(ValueError):system_event(fixture,99,0)


if __name__=='__main__':unittest.main()
