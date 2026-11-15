"""Protocol-only tests; no model calls, training, networking, or concurrency claims."""
import copy
import json
import unittest

from runtime.session import RuntimeSession, TransitionError, StaleProposalError


class RuntimeSessionTests(unittest.TestCase):
    def ready(self):
        s = RuntimeSession("test")
        s.observe("hello", event_time=1)
        s.advance()
        return s

    def test_explicit_transition_sequence(self):
        s = RuntimeSession()
        self.assertEqual(s.state, "idle")
        s.observe("hello", event_time=1)
        self.assertEqual(s.state, "observing")
        self.assertEqual(s.advance(), {"revision": 1, "input": "hello"})
        p = s.propose("abc")
        self.assertEqual(s.state, "proposed")
        s.commit(p, expected_offset=0, count=1)
        self.assertEqual(s.state, "proposed")
        s.commit(p, expected_offset=1, count=2)
        self.assertEqual(s.state, "ready")
        self.assertEqual(s.output, "abc")

    def test_interrupt_partial_commit_and_resume(self):
        s = self.ready()
        p = s.propose("abcdef")
        s.commit(p, expected_offset=0, count=2)
        pending = s.propose_write({"claim": "ab"}, provenance="committed_output", available_at=1, proposal_id=p)
        s.interrupt("correction")
        self.assertEqual(s.output, "ab")
        self.assertEqual(s.candidates[pending]["status"], "rejected")
        self.assertEqual(s.confirmed_writes, [])
        with self.assertRaises(StaleProposalError):
            s.commit(p, expected_offset=2, count=1)
        s.observe(" corrected", event_time=2)
        self.assertEqual(s.state, "interrupted")
        s.resume()
        self.assertEqual(s.state, "observing")
        s.advance()
        p2 = s.propose("new")
        s.commit(p2, expected_offset=0, count=3)
        self.assertEqual(s.output, "abnew")

    def test_new_input_invalidates_uncommitted_output(self):
        s = self.ready()
        p = s.propose("answer")
        candidate = s.propose_write("guess", provenance="prediction", available_at=1, proposal_id=p)
        s.observe("new evidence", event_time=2)
        with self.assertRaises(StaleProposalError):
            s.commit(p, expected_offset=0, count=6)
        self.assertEqual(s.output, "")
        self.assertEqual(s.candidates[candidate]["status"], "rejected")

    def test_commit_idempotent_even_after_interruption(self):
        s = self.ready()
        p = s.propose("abcd")
        r1 = s.commit(p, expected_offset=0, count=2)
        before = copy.deepcopy(s.events)
        r2 = s.commit(p, expected_offset=0, count=2)
        self.assertEqual(r1, r2)
        self.assertEqual(s.events, before)
        self.assertEqual(s.output, "ab")
        with self.assertRaises(TransitionError):
            s.commit(p, expected_offset=0, count=1)
        s.interrupt()
        self.assertEqual(s.commit(p, expected_offset=0, count=2), r1)
        self.assertEqual(s.output, "ab")

    def test_complete_commit_retry_does_not_duplicate_text(self):
        s = self.ready()
        p = s.propose("a")
        s.commit(p, expected_offset=0, count=1)
        s.observe("next", event_time=2)
        s.commit(p, expected_offset=0, count=1)
        self.assertEqual(s.output, "a")
        self.assertEqual(len(s.committed), 1)

    def test_illegal_transitions_and_offsets(self):
        s = RuntimeSession()
        for operation in (s.advance, s.resume, s.interrupt, lambda: s.propose("x")):
            with self.assertRaises(TransitionError):
                operation()
        s.observe("x", event_time=1)
        with self.assertRaises(TransitionError):
            s.propose("x")
        s.advance()
        p = s.propose("abc")
        with self.assertRaises(TransitionError):
            s.propose("second")
        with self.assertRaises(TransitionError):
            s.advance()
        with self.assertRaises(TransitionError):
            s.commit(p, expected_offset=1, count=1)
        with self.assertRaises(ValueError):
            s.commit(p, expected_offset=0, count=4)
        with self.assertRaises(ValueError):
            s.commit(p, expected_offset=0, count=True)
        self.assertTrue(s.interrupt())
        snapshot = s.snapshot()
        self.assertFalse(s.interrupt())
        self.assertEqual(snapshot, s.snapshot())

    def test_future_evidence_and_monotonic_clock(self):
        s = self.ready()
        before = s.snapshot()
        with self.assertRaises(ValueError):
            s.observe("past", event_time=.5)
        with self.assertRaises(ValueError):
            s.propose_write("future", provenance="external_verified", available_at=2)
        self.assertEqual(before, s.snapshot())
        w = s.propose_write("present", provenance="observed_input", available_at=1)
        with self.assertRaises(ValueError):
            s.confirm_write(w, confirmation_provenance="feedback", available_at=2)
        with self.assertRaises(ValueError):
            s.confirm_write(w, confirmation_provenance="feedback", available_at=.5)
        self.assertEqual(s.candidates[w]["status"], "pending")

    def test_commit_never_confirms_write_or_mutates_memory(self):
        s = self.ready()
        p = s.propose("unverified")
        candidate = s.propose_write("claim", provenance="prediction", available_at=1, proposal_id=p)
        s.commit(p, expected_offset=0, count=10)
        self.assertEqual(s.candidates[candidate]["status"], "pending")
        self.assertEqual(s.confirmed_writes, [])
        self.assertNotIn("memory", vars(s))
        for source in ("prediction", "committed_output", "confidence", "agreement"):
            with self.assertRaises(ValueError):
                s.confirm_write(candidate, confirmation_provenance=source, available_at=1)
        result = s.confirm_write(candidate, confirmation_provenance="observed_input", available_at=1)
        self.assertEqual(result["fact_status"], "observed_unverified")
        self.assertEqual(result["provenance"], "prediction")

    def test_confirmed_log_survives_interrupt_and_cannot_be_rejected(self):
        s = self.ready()
        p = s.propose("ab")
        s.commit(p, expected_offset=0, count=1)
        w = s.propose_write({"text": "a"}, provenance="committed_output", available_at=1, proposal_id=p)
        s.confirm_write(w, confirmation_provenance="feedback", available_at=1)
        log = s.confirmed_writes
        s.interrupt()
        self.assertEqual(s.confirmed_writes, log)
        with self.assertRaises(TransitionError):
            s.reject_write(w)
        with self.assertRaises(TransitionError):
            s.confirm_write(w, confirmation_provenance="external_verified", available_at=1)
        log[0]["value"]["text"] = "tampered"
        self.assertEqual(s.confirmed_writes[0]["value"], {"text": "a"})

    def test_observed_input_can_be_value_and_reject_is_explicit(self):
        s = self.ready()
        w = s.propose_write({"observed": "hello"}, provenance="observed_input", available_at=1)
        s.reject_write(w, "not needed")
        self.assertEqual(s.candidates[w]["reason"], "not needed")
        self.assertEqual(s.confirmed_writes, [])
        with self.assertRaises(ValueError):
            s.propose_write(float('nan'), provenance="observed_input", available_at=1)

    def test_snapshot_roundtrip_json_and_deterministic_continuation(self):
        s = self.ready()
        p = s.propose("abcdef")
        s.commit(p, expected_offset=0, count=2)
        w = s.propose_write({"item": [1, 2]}, provenance="committed_output", available_at=1, proposal_id=p)
        s.confirm_write(w, confirmation_provenance="external_verified", available_at=1)
        restored = RuntimeSession.restore(json.loads(json.dumps(s.snapshot())))
        self.assertEqual(s.snapshot(), restored.snapshot())
        for obj in (s, restored):
            obj.interrupt("new input")
            obj.observe("next", event_time=2)
            obj.resume()
            obj.advance()
            p2 = obj.propose("next output")
            obj.commit(p2, expected_offset=0, count=4)
        self.assertEqual(s.snapshot(), restored.snapshot())
        self.assertEqual(s.output, restored.output)

    def test_snapshots_all_protocol_states(self):
        s = RuntimeSession()
        self.assertEqual(RuntimeSession.restore(s.snapshot()).state, "idle")
        for operation in (lambda: s.observe("x", event_time=1), s.advance,
                          lambda: s.propose("abc"), s.interrupt, s.resume):
            operation()
            self.assertEqual(RuntimeSession.restore(s.snapshot()).snapshot(), s.snapshot())

    def test_restore_rejects_confirmation_forgery(self):
        s = self.ready()
        w = s.propose_write({"observed": "hello"}, provenance="observed_input", available_at=1)
        s.confirm_write(w, confirmation_provenance="observed_input", available_at=1)
        original = s.snapshot()
        for field, value in (("fact_status", "caller_asserted_verified"), ("value", "forged"),
                             ("confirmation_provenance", "prediction"), ("available_at", 0)):
            d = copy.deepcopy(original)
            d["confirmed"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                RuntimeSession.restore(d)

    def test_causal_write_references_and_confirmation_are_explicit(self):
        s = self.ready()
        p = s.propose("abc")
        with self.assertRaises(ValueError):
            s.propose_write("uncommitted", provenance="committed_output", available_at=1, proposal_id=p)
        with self.assertRaises(ValueError):
            s.propose_write("no source", provenance="prediction", available_at=1)
        w = s.propose_write("external evidence", provenance="external_verified", available_at=1)
        before = s.confirmed_writes
        s.interrupt()
        # Only proposal-derived pending writes are rejected automatically.
        self.assertEqual(s.candidates[w]["status"], "pending")
        self.assertEqual(s.confirmed_writes, before)
        record = s.confirm_write(w, confirmation_provenance="external_verified", available_at=1)
        self.assertEqual(record["fact_status"], "caller_asserted_verified")
        self.assertEqual(record["version"], 1)

    def test_restore_rejects_corrupted_schema_and_state(self):
        s = self.ready()
        p = s.propose("abc")
        s.commit(p, expected_offset=0, count=1)
        baseline = s.snapshot()
        variants = []
        for key, value in (("schema", "wrong"), ("state", "unknown"), ("revision", 99),
                           ("active_proposal", "p99"), ("event_time", 0), ("next_proposal", 1)):
            d = copy.deepcopy(baseline)
            d[key] = value
            variants.append(d)
        d = copy.deepcopy(baseline)
        d["committed"].append(copy.deepcopy(d["committed"][0]))
        variants.append(d)
        d = copy.deepcopy(baseline)
        d["proposals"][p]["offset"] = 2
        variants.append(d)
        for bad in variants:
            with self.subTest(snapshot=bad), self.assertRaises(ValueError):
                RuntimeSession.restore(bad)


if __name__ == '__main__':
    unittest.main()
