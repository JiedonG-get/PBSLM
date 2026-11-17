"""Predeclared top-slot collision guard on saved fixed-state predictions.

No model runs, no label-dependent policy choice, no actual history retrieval.
Answer targets used ONLY for scoring; guard reads saved routing metadata only.
Costs are deferred query fractions, not measured retrieval/compute savings.
"""
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def guarded_counts(predictions,targets,top_slots):
    if len(top_slots)!=8 or any(type(s) is not int or not 0<=s<8 for s in top_slots):
        raise ValueError('eight known-ID top slots required')
    if len(predictions)!=len(targets):raise ValueError('prediction/target batch sizes differ')
    flagged={e for e in range(8) if top_slots.count(top_slots[e])>1}
    total=accepted=correct=deferred=deferred_errors=accepted_errors=0
    for p,t in zip(predictions,targets):
        if len(p)!=8 or len(t)!=8:raise ValueError('eight predictions/targets per stream required')
        for e,(a,b) in enumerate(zip(p,t)):
            total+=1
            if e in flagged:
                deferred+=1;deferred_errors+=int(a!=b)
            else:
                accepted+=1;correct+=int(a==b);accepted_errors+=int(a!=b)
    return {'n':total,'accepted':accepted,'correct_accepted':correct,'accepted_errors':accepted_errors,
            'deferred':deferred,'deferred_original_errors':deferred_errors,
            'flagged_entities':sorted(flagged)}


def main():
    source=ROOT/'experiments/recurrent_state/sparse_train600/results.json'
    output=ROOT/'experiments/recurrent_state/collision_guard.json'
    if output.exists():raise FileExistsError(output)
    data=json.loads(source.read_text(encoding='utf-8'));rows=[]
    for run in data['runs']:
        counts=[]
        for case in run['all_hot']:
            probe=case['final']
            counts.append(guarded_counts(probe['predicted'],probe['targets'],run['top_slots']))
        total={k:sum(c[k] for c in counts) for k in counts[0] if k!='flagged_entities'}
        total['flagged_entities']=counts[0]['flagged_entities']
        total['coverage']=total['accepted']/total['n']
        total['accepted_accuracy']=total['correct_accepted']/total['accepted'] if total['accepted'] else None
        total['deferred_fraction']=total['deferred']/total['n']
        total['original_errors']=total['accepted_errors']+total['deferred_original_errors']
        total['error_capture_fraction']=total['deferred_original_errors']/total['original_errors'] if total['original_errors'] else None
        rows.append({'seed':run['seed'],'arm':run['arm'],**total})
    summary={}
    for arm in ('soft','sparse','sparse_collision'):
        selected=[r for r in rows if r['arm']==arm]
        n=sum(r['n'] for r in selected);accepted=sum(r['accepted'] for r in selected)
        errors=sum(r['original_errors'] for r in selected)
        summary[arm]={'n':n,'coverage':accepted/n,
              'accepted_accuracy':sum(r['correct_accepted'] for r in selected)/accepted if accepted else None,
              'accepted_errors':sum(r['accepted_errors'] for r in selected),
              'deferred_fraction':sum(r['deferred'] for r in selected)/n,
              'error_capture_fraction':sum(r['deferred_original_errors'] for r in selected)/errors if errors else None}
    report={'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'policy':'defer ALL queries of identities whose learned argmax slot is nonunique; no test-label fitting',
            'scope':'offline saved-probe safety/coverage reference; NOT actual fallback or calibrated confidence',
            'rows':rows,'pooled':summary,'model_executed':False,
            'limits':['finite known8IDs and static route','flag includes all colliding identities even if other identity not currently hot',
                      'no uniform perquery predictions saved, so only all-hot final probes evaluated',
                      'no stored-history access, accuracy after fallback unmeasured','not measured latency or compute savings']}
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
