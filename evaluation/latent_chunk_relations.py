"""Counterfactual relational data for multi-vector PBSLM input experiments.

Each adjacent pair shares the same two-edge phrase y. The x phrase queries one
of y's keys, and the target is the value attached to that key. Held-out splits
reserve ordered value pairs while retaining every individual class in training.
"""
import torch


CLASSES = 8
PHRASE_LENGTH = 8
X_MARKER = CLASSES
Y_MARKER = CLASSES + 1
TARGET_START = CLASSES + 2
VOCAB = TARGET_START + CLASSES
TRAIN_ROWS = 4096
TEST_ROWS = 3072

_HELD_OUT_VALUE_PAIRS = {(value, (value + 2) % CLASSES) for value in range(CLASSES)}


def _sample_pairs(generator, count, device, held_out):
    x_phrases = []
    y_phrases = []
    branch_values = []
    remaining = count

    while remaining:
        candidates = max(64, remaining * 4)
        key_order = torch.rand(candidates, CLASSES, generator=generator, device=device).argsort(-1)
        value_order = torch.rand(candidates, CLASSES, generator=generator, device=device).argsort(-1)
        key0, key1 = key_order[:, 0], key_order[:, 1]
        value0, value1 = value_order[:, 0], value_order[:, 1]

        pair_is_held_out = value1.eq((value0 + 2) % CLASSES)
        accepted = pair_is_held_out if held_out else ~pair_is_held_out
        selected = accepted.nonzero(as_tuple=False).flatten()[:remaining]
        if selected.numel() == 0:
            continue

        key0, key1 = key0[selected], key1[selected]
        value0, value1 = value0[selected], value1[selected]
        selected_count = selected.numel()
        x_noise = torch.randint(
            CLASSES,
            (selected_count, 2, PHRASE_LENGTH - 2),
            generator=generator,
            device=device,
        )
        x_prefix = torch.full((selected_count, 2, 1), X_MARKER, device=device, dtype=torch.long)
        x_keys = torch.stack((key0, key1), dim=1).unsqueeze(-1)
        x_phrases.append(torch.cat((x_prefix, x_keys, x_noise), dim=-1))

        y_noise = torch.randint(
            CLASSES,
            (selected_count, PHRASE_LENGTH - 5),
            generator=generator,
            device=device,
        )
        y_phrase = torch.cat((
            torch.full((selected_count, 1), Y_MARKER, device=device, dtype=torch.long),
            torch.stack((key0, value0, key1, value1), dim=1),
            y_noise,
        ), dim=-1)
        y_phrases.append(y_phrase)
        branch_values.append(torch.stack((value0, value1), dim=1))
        remaining -= selected_count

    x = torch.cat(x_phrases, dim=0)
    y = torch.cat(y_phrases, dim=0)
    targets = torch.cat(branch_values, dim=0)
    ids = torch.cat((x, y.unsqueeze(1).expand(-1, 2, -1)), dim=-1).reshape(-1, 2 * PHRASE_LENGTH)
    source = torch.arange(2, device=device, dtype=torch.long).expand(count, -1).reshape(-1)
    return {
        "ids": ids,
        "target": targets.reshape(-1),
        "source": source,
    }


def make_data(seed, device="cpu"):
    """Return train/test dictionaries with ids [N, 16], target [N], source [N].

    Consecutive rows form counterfactuals: they contain identical y phrases and
    query its first and second edge respectively. The test set holds out eight
    ordered pairs of edge values, with all individual classes present in train.
    """
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    train = _sample_pairs(generator, TRAIN_ROWS // 2, device, held_out=False)
    test = _sample_pairs(generator, TEST_ROWS // 2, device, held_out=True)

    for split in (train, test):
        ids = split["ids"]
        targets = split["target"]
        source = split["source"]
        assert ids.shape == (targets.numel(), 2 * PHRASE_LENGTH)
        assert targets.min().item() >= 0 and targets.max().item() < CLASSES
        assert source.shape == targets.shape
        assert ids.max().item() < TARGET_START
        assert torch.equal(ids[0::2, PHRASE_LENGTH:], ids[1::2, PHRASE_LENGTH:])
        assert bool((targets[0::2] != targets[1::2]).all())
        query_keys = ids[:, 1]
        edge_keys = ids[:, PHRASE_LENGTH + 1:PHRASE_LENGTH + 4:2]
        selected_keys = edge_keys.gather(1, source[:, None]).squeeze(1)
        assert torch.equal(query_keys, selected_keys)
        edge_values = ids[:, PHRASE_LENGTH + 2:PHRASE_LENGTH + 5:2]
        selected_values = edge_values.gather(1, source[:, None]).squeeze(1)
        assert torch.equal(targets, selected_values)

    train_value_pairs = train["ids"][:, PHRASE_LENGTH + 2:PHRASE_LENGTH + 5:2][0::2]
    test_value_pairs = test["ids"][:, PHRASE_LENGTH + 2:PHRASE_LENGTH + 5:2][0::2]
    assert not any(tuple(pair) in _HELD_OUT_VALUE_PAIRS for pair in train_value_pairs.cpu().tolist())
    assert all(tuple(pair) in _HELD_OUT_VALUE_PAIRS for pair in test_value_pairs.cpu().tolist())
    combined_ids = torch.cat((train["ids"], test["ids"]), dim=0)
    _, counts = torch.unique(combined_ids, dim=0, return_counts=True)
    assert bool(torch.all(counts == 1))
    assert set(train["target"].cpu().tolist()) == set(range(CLASSES))
    return train, test
