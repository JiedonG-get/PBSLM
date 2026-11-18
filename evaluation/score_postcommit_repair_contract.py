"""Score persisted post-commit deliveries and check reversal replay equivalence."""

import argparse
from collections import Counter
import json
from pathlib import Path

from evaluation.fixtures.postcommit_repair_contract_v1 import DEVELOPMENT, FORMAL
from evaluation.postcommit_repair_contract import ARMS, score


TIMINGS = {'at', 'decision_seconds', 'duration', 'finish_perf', 'received_perf',
           'routing_seconds', 'settled_seconds', 'start_perf'}
REPLAY_FIELDS = ('trace', 'committed', 'journal', 'worker_jobs', 'delivery_decisions')


def without_timings(value):
    if isinstance(value, list):
        return [without_timings(item) for item in value]
    if isinstance(value, dict):
        return {key: without_timings(item) for key, item in value.items() if key not in TIMINGS}
    return value


def verify_journal(run):
    journal_units = [item for item in run['journal'] if item['kind'] == 'output_committed']
    journal_pairs = [(item['generation_revision'], item['unit']) for item in journal_units]
    units = [(item['revision'], item['unit']) for item in run['committed']]
    revisions = [revision for revision, _ in units]
    prefix_valid = True
    prior = []
    commits_by_revision = dict(units)
    for event in run['trace']:
        if event['prior_commit_count'] != len(prior) or event['prior_committed_unit'] != (prior[-1] if prior else None):
            prefix_valid = False
        if event['revision'] in commits_by_revision:
            prior.append(commits_by_revision[event['revision']])
    decisions = {item['revision']: item['offered_unit'] for item in run['delivery_decisions'] if item['offered_unit']}
    guarded_offer_verified = all(
        item['reason'] not in {'verified_model', 'template_fallback', 'template'} or item['eligible']
        for item in run['delivery_decisions']
    )
    return (journal_pairs == units and revisions == sorted(set(revisions)) and prefix_valid
            and len(prior) == len(units) and decisions == dict(units) and guarded_offer_verified)


def analyze_report(report):
    cases = {'development': DEVELOPMENT, 'formal': FORMAL}[report['split']]
    by_name = {case.identity: case for case in cases}
    pairs = {(case.identity, arm) for case in cases for arm in ARMS}
    indexed = {(run['case'], run['arm']): run for run in report['runs']}
    if set(indexed) != pairs or len(indexed) != len(report['runs']):
        raise ValueError('missing, duplicate or unexpected case/arm pair')
    declared_order = tuple(report['arms'])
    if declared_order not in (ARMS, ARMS[::-1]):
        raise ValueError('invalid declared arm order')
    if [(run['case'], run['arm']) for run in report['runs']] != [
        (case.identity, arm) for case in cases for arm in declared_order
    ]:
        raise ValueError('report order differs from declared fixture and arm order')

    scored = {}
    for key, run in indexed.items():
        case = by_name[key[0]]
        events = [(event['event_id'], event['text']) for event in run['trace']]
        if events != [(event.identity, event.text) for event in case.events]:
            raise ValueError(f'event delivery differs from fixture for {key!r}')
        scored[key] = score(case, run)

    by_arm = {}
    for arm in ARMS:
        rows = [row for (_, name), rows in scored.items() if name == arm for row in rows]
        runs = [run for (_, name), run in indexed.items() if name == arm]
        update_revisions = [({event['revision'] for event in run['trace'] if event['route'] == 'update'}, run)
                            for run in runs]
        repair_jobs = [job for revisions, run in update_revisions for job in run['worker_jobs']
                       if job['revision'] in revisions]
        repair_rows = [row for row in rows if row['expected_kind'] in {'withdraw_and_replace', 'resolve_unknown'}]
        outcomes = {}
        for kind in ('initial', 'silent', 'withdraw_and_replace', 'resolve_unknown'):
            selected = [row for row in rows if row['expected_kind'] == kind]
            outcomes[kind] = {'total': len(selected), 'pass': sum(row['strict_pass'] for row in selected)}
        by_arm[arm] = {
            'outcomes': outcomes,
            'withdrawals_mentioning_expected_new_value': sum(bool(row['new_value_mentioned']) for row in rows
                                                           if row['expected_kind'] == 'withdraw_and_replace'),
            'withdrawal_prior_claims_matching_gold': sum(bool(row['prior_claim_matches_gold']) for row in rows
                                                        if row['expected_kind'] == 'withdraw_and_replace'),
            'repair_prior_claims_recognized': sum(row['prior_claim_recognized'] for row in repair_rows),
            'repair_prior_claims_matching_gold': sum(bool(row['prior_claim_matches_gold']) for row in repair_rows),
            'repair_delivery_reasons': dict(Counter(row['delivery_reason'] or 'no_delivery_decision' for row in rows
                                                    if row['expected_kind'] in {'withdraw_and_replace', 'resolve_unknown'})),
            'valid_journal_runs': sum(verify_journal(run) for run in runs),
            'generation_errors': sum(item['kind'] == 'generation_failed' for run in runs for item in run['journal']),
            'repair_model_calls': len(repair_jobs),
            'repair_input_tokens': sum(job['input_tokens'] for job in repair_jobs),
            'repair_output_tokens': sum(job['output_tokens'] for job in repair_jobs),
            'repair_worker_seconds': sum(job['duration'] for job in repair_jobs),
            'update_settlement_seconds': sum(step['settled_seconds'] for run in runs for step in run['trace']
                                             if step['route'] == 'update'),
        }
    report_scores = {'split': report['split'], 'arms': by_arm,
                     'per_case': [{'case': case, 'arm': arm, 'scores': rows}
                                  for (case, arm), rows in sorted(scored.items())]}
    if report['split'] == 'formal':
        report_scores['formal_absolute_criteria'] = {
            arm: (
                all(outcome['pass'] == outcome['total'] for name, outcome in stats['outcomes'].items()
                    if name != 'initial')
                and stats['valid_journal_runs'] == len(cases)
                and stats['generation_errors'] == 0
            ) for arm, stats in by_arm.items()
        }
    return report_scores, indexed


def replay_differences(primary, repeat, first_runs, repeat_runs):
    metadata_fields = ('split', 'checkpoint_files_sha256', 'max_new_tokens')
    metadata = [field for field in metadata_fields if primary[field] != repeat[field]]
    if primary['arms'] != repeat['arms'][::-1]:
        metadata.append('arm_order_not_reversed')
    if primary['provenance']['source_sha256_at_capture'] != repeat['provenance']['source_sha256_at_capture']:
        metadata.append('source_sha256_at_capture')
    runs = []
    for (case, arm), first in sorted(first_runs.items()):
        repeated = repeat_runs.get((case, arm))
        changed = ([field for field in REPLAY_FIELDS if without_timings(first[field]) != without_timings(repeated[field])]
                   if repeated is not None else ['missing_run'])
        if changed:
            runs.append({'case': case, 'arm': arm, 'changed_fields': changed})
    runs.extend({'case': case, 'arm': arm, 'changed_fields': ['extra_run']}
                for case, arm in sorted(repeat_runs.keys() - first_runs.keys()))
    return {'equivalent': not metadata and not runs, 'metadata_differences': metadata, 'run_differences': runs}


def read_report(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    primary = read_report(args.input)
    scores, primary_runs = analyze_report(primary)
    if args.replay is not None:
        repeat = read_report(args.replay)
        scores['replay'], repeat_runs = analyze_report(repeat)
        scores['replay_comparison'] = replay_differences(primary, repeat, primary_runs, repeat_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(scores, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    if args.replay is not None and not scores['replay_comparison']['equivalent']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
