"""Recount text-key transfer predictions and pool count metrics by condition."""
import json
from pathlib import Path
import torch
from evaluation.text_key_transfer import counts


def main():
    source=Path('experiments/recurrent_state/content_fast600/text_key_transfer.json');out=source.with_name('text_key_transfer_record_summary.json')
    if out.exists():raise FileExistsError(out)
    data=json.loads(source.read_text(encoding='utf-8'));pooled={}
    for run in data['runs']:
        recounted={}
        for row in run['traces']:
            for method,predictions in row['predictions'].items():
                key=f"{row['support']}/{row['split']}/{method}"
                c=counts(torch.tensor(predictions),torch.tensor(row['targets']))
                aggregate=recounted.setdefault(key,{name:0 for name in c})
                for name,value in c.items():aggregate[name]+=value
        if recounted!=run['metrics']:raise AssertionError('trace vs saved count mismatch')
        for key,c in recounted.items():
            aggregate=pooled.setdefault(key,{name:0 for name in c})
            for name,value in c.items():aggregate[name]+=value
    for c in pooled.values():
        c['known_accuracy']=c['known_correct']/c['known_n'] if c['known_n'] else None
        c['unknown_false_accept_rate']=c['unknown_false_accept']/c['unknown_n'] if c['unknown_n'] else None
    record_selection={}
    for run in data['runs']:
        for row in run['traces']:
            observed=set(row['record_entities'])
            for feature,geometry in row['geometry'].items():
                key=f"{row['support']}/{row['split']}/{feature}"
                item=record_selection.setdefault(key,{'known_n':0,'correct_record':0})
                for entity,selected in enumerate(geometry['top_record_entity']):
                    if entity in observed:
                        item['known_n']+=1;item['correct_record']+=int(entity==selected)
    for item in record_selection.values():item['record_accuracy']=item['correct_record']/item['known_n']
    out.write_text(json.dumps({'source':str(source),'pooled':pooled,'record_selection':record_selection},indent=2)+'\n',encoding='utf-8')
    for key,value in pooled.items():
        if key.startswith(('8/held/','16/held/')):print(key,value['known_accuracy'],value['unknown_false_accept_rate'])


if __name__=='__main__':main()
