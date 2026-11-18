import json
import tempfile
from pathlib import Path
import unittest
from evaluation.continuous_event_protocol import load

ROOT=Path(__file__).resolve().parents[1]


class ContinuousEventFixturesTests(unittest.TestCase):
    def test_jsonl_events_validate_without_future_in_model_view(self):
        path=ROOT/'evaluation/fixtures/continuous_events_dev.jsonl'
        rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows),2)
        for row in rows:
            with tempfile.TemporaryDirectory() as directory:
                one=Path(directory)/'fixture.json';one.write_text(json.dumps(row),encoding='utf-8')
                fixture=load(one)
                self.assertEqual(len(fixture['events']),8)
                self.assertTrue(all('oracle' in e for e in fixture['events']))

    def test_stream_does_not_leak_oracle_future_or_gold_target(self):
        from evaluation.continuous_event_protocol import system_event
        rows=[json.loads(x) for x in (ROOT/'evaluation/fixtures/continuous_events_dev.jsonl').read_text(encoding='utf-8').splitlines()]
        event_before_revision=system_event(rows[0],2,25)
        self.assertNotIn('图书馆门口',json.dumps(event_before_revision,ensure_ascii=False))
        event_at_revision=system_event(rows[0],3,50)
        current=json.dumps(event_at_revision,ensure_ascii=False)
        self.assertIn('图书馆门口',current) # Present in the delivered observation, not as oracle metadata.
        self.assertNotIn('oracle',current)
        self.assertNotIn('target',current)
        self.assertEqual(len(event_at_revision['visible_prior_events']),3)


if __name__=='__main__':unittest.main()
