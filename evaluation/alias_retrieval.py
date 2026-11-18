"""Deterministic retrieval over the append-only prefix of public observations."""

from collections import Counter
from math import log1p
from unicodedata import normalize

from evaluation.continuous_event_protocol import public_event


K1 = 1.2
B = 0.75


def _bigrams(text):
    letters = ''.join(char for char in normalize('NFKC', text).casefold() if char.isalnum())
    return Counter(zip(letters, letters[1:]))


def rank_context(observations, query, *, limit=8) -> list[dict]:
    """Select BM25 top-k from a chronological observation prefix and return it in arrival order.

    Event IDs are unique in the output. If an ID repeats, its first row is retained.
    Equal scores favor later rows. Pass None to return the whole prefix.
    """
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError('context limit must be a nonnegative integer or None')

    unique = {}
    for row in observations:
        if row['kind'] != 'observation':
            raise ValueError('context rows must be observations')
        if row['event_id'] not in unique:
            unique[row['event_id']] = public_event(row)
    rows = list(unique.values())
    if limit is None or len(rows) <= limit:
        return rows
    if limit == 0:
        return []

    documents = [_bigrams(row['text']) for row in rows]
    lengths = [sum(document.values()) for document in documents]
    average = sum(lengths) / len(rows)
    terms = sorted(_bigrams(query))
    if average == 0 or not terms:
        return rows[-limit:]

    # Ascending term order keeps score accumulation independent of hash randomization.
    df = {term: sum(term in document for document in documents) for term in terms}
    idf = {term: log1p((len(rows) - count + .5) / (count + .5)) for term, count in df.items()}
    scores = []
    for document, length in zip(documents, lengths):
        normalizer = K1 * (1 - B + B * length / average)
        score = sum(idf[term] * frequency * (K1 + 1) / (frequency + normalizer)
                    for term in terms if (frequency := document[term]))
        scores.append(score)

    ranked = sorted(range(len(rows)), key=lambda index: (scores[index], index), reverse=True)
    return [rows[index] for index in sorted(ranked[:limit])]
