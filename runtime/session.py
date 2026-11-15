"""Deterministic synchronous event/state prototype.

This module has no model, API, network, threads, optimizer, or numerical memory.
Committing text is an output action, not a claim that it is true and not a memory
write. Confirmations append provenance records only; no numerical M rollback is
implemented. Snapshots reproduce this local protocol, not external side effects.
"""
import copy
import json
import math


class TransitionError(RuntimeError):
    """An operation is illegal in the current explicit state."""


class StaleProposalError(TransitionError):
    """A proposal belongs to an older input revision or is no longer active."""


STATES = {"idle", "observing", "ready", "proposed", "interrupted"}
SOURCES = {"observed_input", "prediction", "committed_output", "feedback", "external_verified"}
CONFIRMATIONS = {"observed_input", "feedback", "external_verified"}


def _json_copy(value):
    """Reject non-JSON or nonfinite payloads rather than retaining caller aliases."""
    return json.loads(json.dumps(value, allow_nan=False))


def _time(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("event time must be a finite number")
    return value


class RuntimeSession:
    """Local state machine; every caller-supplied event time must be monotonic.

    Transitions: idle -> observe -> observing -> advance -> ready -> propose ->
    proposed -> commit(partial: proposed, complete: ready). Interrupt is allowed
    in observing/ready/proposed; resume returns observing and requires advance.
    New observe invalidates the current proposal and derived pending writes.
    """
    SCHEMA = "pbslm.runtime.session.v1"

    def __init__(self, session_id="local"):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be nonempty text")
        self.session_id = session_id
        self.state = "idle"
        self.revision = 0
        self.event_time = 0
        self.inputs = []
        self.committed = []
        self.events = []
        self.proposals = {}
        self.active_proposal = None
        self.candidates = {}
        self._confirmed = []
        self._next_proposal = 1
        self._next_candidate = 1

    @property
    def output(self):
        return "".join(item["text"] for item in self.committed)

    @property
    def confirmed_writes(self):
        """Return a copy: callers cannot mutate the append-only confirmation log."""
        return copy.deepcopy(self._confirmed)

    def _clock(self, event_time):
        value = _time(event_time)
        if value < self.event_time:
            raise ValueError("event time cannot go backwards")
        return value

    def _record(self, kind, **details):
        self.events.append({"kind": kind, "time": self.event_time,
                            "revision": self.revision, **details})

    def _invalidate(self, reason):
        if self.active_proposal is not None:
            self.proposals[self.active_proposal]["status"] = "stale"
        self.active_proposal = None
        for candidate in self.candidates.values():
            if candidate["status"] == "pending" and candidate["proposal_id"] is not None:
                candidate["status"] = "rejected"
                candidate["reason"] = reason

    def observe(self, text, *, event_time):
        if not isinstance(text, str) or not text:
            raise ValueError("observe requires nonempty text")
        value = self._clock(event_time)
        self._invalidate("new_input")
        self.event_time = value
        self.revision += 1
        self.inputs.append({"text": text, "time": value, "revision": self.revision})
        if self.state != "interrupted":
            self.state = "observing"
        self._record("observe", text=text)
        return self.revision

    def advance(self):
        if self.state not in {"observing", "ready"}:
            raise TransitionError("advance requires observing or ready")
        self.state = "ready"
        self._record("advance")
        return {"revision": self.revision, "input": "".join(x["text"] for x in self.inputs)}

    def propose(self, text):
        if self.state != "ready":
            raise TransitionError("propose requires ready")
        if not isinstance(text, str) or not text:
            raise ValueError("proposal requires nonempty text")
        ident = f"p{self._next_proposal}"
        self._next_proposal += 1
        self.proposals[ident] = {"text": text, "revision": self.revision,
                                 "offset": 0, "status": "active", "receipts": {}}
        self.active_proposal = ident
        self.state = "proposed"
        self._record("propose", proposal_id=ident)
        return ident

    def commit(self, proposal_id, *, expected_offset, count):
        """Commit characters with an exact-offset idempotency receipt.

        Exact duplicate retries return the original receipt even after interruption
        or new input. They do not emit again. Any new commit on a stale proposal
        fails. Reusing an offset with a different count is an error.
        """
        if type(expected_offset) is not int or expected_offset < 0 or type(count) is not int or count < 1:
            raise ValueError("offset/count must be nonnegative/positive integers")
        if proposal_id not in self.proposals:
            raise StaleProposalError("unknown proposal")
        p = self.proposals[proposal_id]
        key = str(expected_offset)
        if key in p["receipts"]:
            receipt = p["receipts"][key]
            if receipt["count"] != count:
                raise TransitionError("offset was committed with a different count")
            return copy.deepcopy(receipt)
        if (self.state != "proposed" or self.active_proposal != proposal_id
                or p["revision"] != self.revision or p["status"] != "active"):
            raise StaleProposalError("proposal is stale or inactive")
        if expected_offset != p["offset"]:
            raise TransitionError("commit expected_offset mismatch")
        end = expected_offset + count
        if end > len(p["text"]):
            raise ValueError("commit exceeds proposal length")
        receipt = {"proposal_id": proposal_id, "offset": expected_offset,
                   "count": count, "text": p["text"][expected_offset:end]}
        p["receipts"][key] = copy.deepcopy(receipt)
        p["offset"] = end
        self.committed.append(copy.deepcopy(receipt))
        if end == len(p["text"]):
            p["status"] = "complete"
            self.active_proposal = None
            self.state = "ready"
        self._record("commit", **receipt)
        return copy.deepcopy(receipt)

    def interrupt(self, reason="new_event"):
        if self.state == "interrupted":
            return False
        if self.state not in {"observing", "ready", "proposed"}:
            raise TransitionError("interrupt requires an active session")
        if not isinstance(reason, str):
            raise ValueError("reason must be text")
        self._invalidate("interrupt:" + reason)
        self.state = "interrupted"
        self._record("interrupt", reason=reason)
        return True

    def resume(self):
        if self.state != "interrupted":
            raise TransitionError("resume requires interrupted")
        self.state = "observing"
        self._record("resume")
        return self.revision

    def propose_write(self, value, *, provenance, available_at, proposal_id=None):
        """Queue JSON evidence, never alter numerical M. Predictions are not facts."""
        timestamp = _time(available_at)
        if provenance not in SOURCES:
            raise ValueError("unknown provenance")
        if timestamp > self.event_time:
            raise ValueError("future evidence is not yet available")
        payload = _json_copy(value)
        if provenance in {"prediction", "committed_output"} and proposal_id is None:
            raise ValueError("model-derived evidence requires proposal_id")
        if proposal_id is not None:
            p = self.proposals.get(proposal_id)
            if p is None or p["status"] == "stale":
                raise StaleProposalError("write references stale proposal")
            if provenance == "committed_output" and p["offset"] == 0:
                raise ValueError("no output has been committed")
        ident = f"w{self._next_candidate}"
        self._next_candidate += 1
        self.candidates[ident] = {"value": payload, "provenance": provenance,
                                 "available_at": timestamp, "proposal_id": proposal_id,
                                 "status": "pending"}
        self._record("propose_write", candidate_id=ident)
        return ident

    def confirm_write(self, candidate_id, *, confirmation_provenance, available_at):
        """Explicit external evidence is required; output submission is not truth.

        Even external_verified means the *caller asserts* external verification;
        this protocol cannot authenticate an outside source. No trust score is
        inferred from a model prediction, confidence, agreement, or commit.
        """
        if confirmation_provenance not in CONFIRMATIONS:
            raise ValueError("self-prediction/commit cannot confirm correctness")
        timestamp = _time(available_at)
        if timestamp > self.event_time:
            raise ValueError("future confirmation is not available")
        candidate = self.candidates.get(candidate_id)
        if candidate is None or candidate["status"] != "pending":
            raise TransitionError("confirmation requires a pending candidate")
        if timestamp < candidate["available_at"]:
            raise ValueError("confirmation cannot precede evidence")
        record = {"candidate_id": candidate_id, "value": copy.deepcopy(candidate["value"]),
                  "provenance": candidate["provenance"], "confirmation_provenance": confirmation_provenance,
                  "available_at": timestamp, "confirmed_at": self.event_time,
                  "fact_status": {"external_verified": "caller_asserted_verified",
                                  "feedback": "reported_feedback", "observed_input": "observed_unverified"}[confirmation_provenance],
                  "version": len(self._confirmed) + 1}
        candidate["status"] = "confirmed"
        self._confirmed.append(record)
        self._record("confirm_write", candidate_id=candidate_id)
        return copy.deepcopy(record)

    def reject_write(self, candidate_id, reason="rejected"):
        candidate = self.candidates.get(candidate_id)
        if candidate is None or candidate["status"] != "pending":
            raise TransitionError("rejection requires a pending candidate")
        if not isinstance(reason, str):
            raise ValueError("reason must be text")
        candidate["status"] = "rejected"
        candidate["reason"] = reason
        self._record("reject_write", candidate_id=candidate_id, reason=reason)

    def snapshot(self):
        return _json_copy({"schema": self.SCHEMA, "session_id": self.session_id,
                          "state": self.state, "revision": self.revision,
                          "event_time": self.event_time, "inputs": self.inputs,
                          "committed": self.committed, "events": self.events,
                          "proposals": self.proposals, "active_proposal": self.active_proposal,
                          "candidates": self.candidates, "confirmed": self._confirmed,
                          "next_proposal": self._next_proposal, "next_candidate": self._next_candidate})

    @classmethod
    def restore(cls, snapshot):
        """Validate JSON protocol state; never re-emit output or execute writes."""
        try:
            d = _json_copy(snapshot)
            expected = set(cls().snapshot())
            if not isinstance(d, dict) or set(d) != expected or d["schema"] != cls.SCHEMA:
                raise ValueError("snapshot schema mismatch")
            obj = cls(d["session_id"])
            if d["state"] not in STATES or type(d["revision"]) is not int or d["revision"] < 0:
                raise ValueError("invalid state/revision")
            _time(d["event_time"])
            for key in ("inputs", "committed", "events", "confirmed"):
                if not isinstance(d[key], list):
                    raise ValueError("invalid snapshot collection")
            for key in ("proposals", "candidates"):
                if not isinstance(d[key], dict):
                    raise ValueError("invalid snapshot mapping")
            for key in ("next_proposal", "next_candidate"):
                if type(d[key]) is not int or d[key] < 1:
                    raise ValueError("invalid counter")
            for key in ("state", "revision", "event_time", "inputs", "committed", "events",
                        "proposals", "active_proposal", "candidates"):
                setattr(obj, key, d[key])
            obj._confirmed = d["confirmed"]
            obj._next_proposal, obj._next_candidate = d["next_proposal"], d["next_candidate"]
            obj._validate_restored()
            return obj
        except (KeyError, TypeError, AttributeError, IndexError) as exc:
            raise ValueError("malformed snapshot") from exc

    def _validate_restored(self):
        if self.event_time < 0 or (self.state == "idle") != (self.revision == 0):
            raise ValueError("idle/revision mismatch")
        if len(self.inputs) != self.revision:
            raise ValueError("input revision mismatch")
        previous = 0
        for revision, item in enumerate(self.inputs, 1):
            if (item["revision"] != revision or not isinstance(item["text"], str) or not item["text"]
                    or not previous <= _time(item["time"]) <= self.event_time):
                raise ValueError("invalid input history")
            previous = item["time"]
        expected_commits = {}
        active = []
        for ident, proposal in self.proposals.items():
            if not ident.startswith("p") or not ident[1:].isdigit() or int(ident[1:]) >= self._next_proposal:
                raise ValueError("invalid proposal id")
            if (not isinstance(proposal["text"], str) or not proposal["text"]
                    or type(proposal["offset"]) is not int or not 0 <= proposal["offset"] <= len(proposal["text"])
                    or type(proposal["revision"]) is not int or not 1 <= proposal["revision"] <= self.revision
                    or proposal["status"] not in {"active", "complete", "stale"}):
                raise ValueError("invalid proposal")
            if proposal["status"] == "active":
                active.append(ident)
                if proposal["revision"] != self.revision or proposal["offset"] == len(proposal["text"]):
                    raise ValueError("invalid active proposal")
            if proposal["status"] == "complete" and proposal["offset"] != len(proposal["text"]):
                raise ValueError("incomplete completed proposal")
            offset = 0
            for key, receipt in sorted(proposal["receipts"].items(), key=lambda item: int(item[0])):
                if (key != str(offset) or receipt["proposal_id"] != ident or receipt["offset"] != offset
                        or type(receipt["count"]) is not int or receipt["count"] < 1
                        or offset + receipt["count"] > len(proposal["text"])
                        or receipt["text"] != proposal["text"][offset:offset + receipt["count"]]):
                    raise ValueError("invalid commit receipt")
                expected_commits[(ident, offset)] = receipt
                offset += receipt["count"]
            if offset != proposal["offset"]:
                raise ValueError("receipt offset mismatch")
        if (self.state == "proposed" and active != [self.active_proposal]) or (self.state != "proposed" and (active or self.active_proposal is not None)):
            raise ValueError("active state mismatch")
        seen = set()
        for item in self.committed:
            key = (item["proposal_id"], item["offset"])
            if key in seen or expected_commits.get(key) != item:
                raise ValueError("invalid committed history")
            seen.add(key)
        if seen != set(expected_commits):
            raise ValueError("missing committed history")
        confirmed_ids = set()
        for version, record in enumerate(self._confirmed, 1):
            ident = record["candidate_id"]
            if (record["version"] != version or ident in confirmed_ids
                    or record["confirmation_provenance"] not in CONFIRMATIONS
                    or not _time(record["available_at"]) <= _time(record["confirmed_at"]) <= self.event_time):
                raise ValueError("invalid confirmation log")
            confirmed_ids.add(ident)
        for ident, candidate in self.candidates.items():
            if (not ident.startswith("w") or not ident[1:].isdigit() or int(ident[1:]) >= self._next_candidate
                    or candidate["status"] not in {"pending", "confirmed", "rejected"}
                    or candidate["provenance"] not in SOURCES
                    or _time(candidate["available_at"]) > self.event_time):
                raise ValueError("invalid candidate")
            if candidate["proposal_id"] is not None and candidate["proposal_id"] not in self.proposals:
                raise ValueError("candidate references unknown proposal")
            if candidate["provenance"] in {"prediction", "committed_output"} and candidate["proposal_id"] is None:
                raise ValueError("model-derived candidate lacks proposal")
            if candidate["proposal_id"] is not None:
                proposal = self.proposals[candidate["proposal_id"]]
                if candidate["status"] == "pending" and proposal["status"] == "stale":
                    raise ValueError("stale proposal has pending write")
                if candidate["provenance"] == "committed_output" and proposal["offset"] == 0:
                    raise ValueError("candidate has no committed source")
            if (candidate["status"] == "confirmed") != (ident in confirmed_ids):
                raise ValueError("confirmation/candidate mismatch")
        for record in self._confirmed:
            candidate = self.candidates.get(record["candidate_id"])
            expected_status = {"external_verified": "caller_asserted_verified", "feedback": "reported_feedback",
                               "observed_input": "observed_unverified"}[record["confirmation_provenance"]]
            if (candidate is None or record["value"] != candidate["value"]
                    or record["provenance"] != candidate["provenance"]
                    or record["fact_status"] != expected_status
                    or record["available_at"] < candidate["available_at"]):
                raise ValueError("confirmation evidence mismatch")
        if not confirmed_ids <= set(self.candidates):
            raise ValueError("unknown confirmed candidate")
        known_events = {"observe", "advance", "propose", "commit", "interrupt", "resume",
                        "propose_write", "confirm_write", "reject_write"}
        last_time = 0
        for event in self.events:
            if (event["kind"] not in known_events or not last_time <= _time(event["time"]) <= self.event_time
                    or type(event["revision"]) is not int or not 0 <= event["revision"] <= self.revision):
                raise ValueError("invalid event log")
            last_time = event["time"]
