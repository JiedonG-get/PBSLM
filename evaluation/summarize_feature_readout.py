"""Recount isolated readout control from saved predictions; no model invocation."""
import json
from pathlib import Path
import statistics


def main():
    path=Path('experiments/consolidation/feature_readout200.json')
    out=path.with_name('feature_readout200_summary.json')
    if out.exists():raise FileExistsError(out)
    data=json.loads(path.read_text(encoding='utf-8'))
    for run in data['runs']:
        for arm in run['arms'].values():
            for stage in ('before','after'):
                for item in arm[stage].values():
                    actual=sum(a==b for a,b in zip(item['predicted_ids'],item['target_ids']))/len(item['target_ids'])
                    if abs(actual-item['accuracy'])>1e-6:raise AssertionError('prediction/aggregate mismatch')
    report={'source':str(path),'seeds':[r['seed'] for r in data['runs']],
            'arms':{feature:{stage:{key:{'mean':statistics.mean(r['arms'][feature][stage][key]['accuracy'] for r in data['runs']),
                                          'per_seed':[r['arms'][feature][stage][key]['accuracy'] for r in data['runs']]}
                                     for key in data['runs'][0]['arms'][feature][stage]}
                            for stage in ('before','after')}
                    for feature in ('e_key','token_mean')},
            'note':'three-seed descriptive means; no significance claim, explicitly aligned E pretraining prior'}
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
