"""Structured external oracle fixtures with isolated observation/query channels.

This serializer never emits gold state, target answers, relevance labels, future events
or expected output text to the system under test. It is scaffolding, NOT a model/test set.
"""
from pathlib import Path
import json
import math

FORMAT='pbslm.continuous_event_stream.v1'


def load(path):
    return validate(json.loads(Path(path).read_text(encoding='utf-8')))


def load_streams(path):
    streams = [validate(json.loads(line)) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not streams or len({stream['stream_id'] for stream in streams}) != len(streams):
        raise ValueError('nonempty fixture with distinct stream IDs required')
    return streams


def validate(data):
    if (not isinstance(data, dict) or data.get('format') != FORMAT
            or not isinstance(data.get('stream_id'), str) or not data['stream_id']
            or not isinstance(data.get('events'), list) or not data['events']):
        raise ValueError('unsupported continuous stream fixture')
    last = -1
    last_delivery = -1
    event_ids = set()
    for event in data['events']:
        if not isinstance(event, dict):
            raise ValueError('event must be an object')
        tick = event.get('source_tick')
        delivery = event.get('deliver_after_ms')
        if type(tick) is not int or tick <= last:
            raise ValueError('source ticks must strictly increase')
        if type(delivery) not in (int, float) or delivery < last_delivery or delivery < 0 or not math.isfinite(delivery):
            raise ValueError('scheduled deliveries must be finite, nonnegative and nondecreasing')
        if not isinstance(event.get('event_id'), str) or not event['event_id'] or event['event_id'] in event_ids:
            raise ValueError('event IDs must be nonempty and unique')
        if event.get('kind') not in ('observation', 'query') or not isinstance(event.get('text'), str) or not event['text'].strip():
            raise ValueError('invalid event kind or text')
        event_ids.add(event['event_id'])
        last = tick
        last_delivery = delivery
    return data


def public_event(event):
    return {key: event[key] for key in ('event_id', 'source_tick', 'deliver_after_ms', 'kind', 'text')}


def public_stream(fixture):
    """Project the fixture onto the fields available at ingress, without annotations."""
    return {'stream_id': fixture['stream_id'], 'events': [public_event(event) for event in fixture['events']]}


def system_event(fixture,index,delivered_at_ms):
    if type(index) is not int or not 0<=index<len(fixture['events']):raise ValueError('event index invalid')
    event=fixture['events'][index]
    prior=[{'event_id':x['event_id'],'source_tick':x['source_tick'],'kind':x['kind'],'text':x['text']}
           for x in fixture['events'][:index] if x['deliver_after_ms']<=event['deliver_after_ms']]
    # No target state, relevance annotation, source reliability/expected answer or future content.
    return {'stream_id':fixture['stream_id'],'event_id':event['event_id'],'source_tick':event['source_tick'],
            'delivered_at_ms':delivered_at_ms,'kind':event['kind'],'text':event['text'],
            'visible_prior_events':prior}
