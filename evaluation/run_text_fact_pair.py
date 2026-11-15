"""Paired baseline/key-aligned Chinese fact runs, sequential GPU budget."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'experiments/text_facts/paired800'
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    evidence={}
    for tag,weight in [('baseline',0.),('key_aligned',.2)]:
        command=[sys.executable,'-m','evaluation.text_facts','--seeds','0','1','2',
                 '--episodes','800','--eval-episodes','32','--key-align-weight',str(weight),
                 '--output',f'experiments/text_facts/paired800/{tag}.json']
        with (out/f'{tag}.log').open('w',encoding='utf-8') as log:
            run=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        evidence[tag]={'command':command,'returncode':run.returncode}
        (out/'runner.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
        print(tag,run.returncode,flush=True)
        if run.returncode: raise SystemExit(run.returncode)


if __name__=='__main__': main()
