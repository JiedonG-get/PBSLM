"""Seed-macro summary of continuous-key fallback experiment."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='experiments/recurrent_state/continuous_keys_final800.json')
    parser.add_argument('--output',default='experiments/recurrent_state/continuous_keys_final800_summary.json')
    args=parser.parse_args()
    source=Path(args.input)
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    data=json.loads(source.read_text(encoding='utf-8'));metrics={}
    for key in data['runs'][0]['evaluation']:
        metrics[key]={}
        for arm in ('state','state_abstain_fallback','dictionary'):
            metrics[key][arm]={m:{'mean':statistics.mean(r['evaluation'][key]['metrics'][arm][m] for r in data['runs']),
                                   'per_seed':[r['evaluation'][key]['metrics'][arm][m] for r in data['runs']]}
                               for m in ('accuracy','known_accuracy','unknown_accuracy','unknown_false_accept_rate')}
            if arm=='state_abstain_fallback':
                metrics[key][arm]['fallback_call_rate']={'mean':statistics.mean(r['evaluation'][key]['metrics'][arm]['fallback_call_rate'] for r in data['runs']),
                                    'per_seed':[r['evaluation'][key]['metrics'][arm]['fallback_call_rate'] for r in data['runs']]}
    out={'source':str(source),'seeds':[r['seed'] for r in data['runs']],'metrics':metrics,
         'thresholds':[r['dictionary_threshold'] for r in data['runs']],
         'note':'seed macro averages, not pooled confidence intervals'}
    output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
