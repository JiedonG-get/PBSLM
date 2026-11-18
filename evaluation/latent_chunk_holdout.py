"""Balanced relation splits using the existing counterfactual phrase layout."""

import torch

from evaluation.latent_chunk_relations import (
    CLASSES,
    PHRASE_LENGTH,
    TEST_ROWS,
    TRAIN_ROWS,
    _sample_pairs,
)


def _balanced_value_pairs(offsets, group_count, generator):
    groups_per_value = group_count // CLASSES
    groups_per_offset, extra_offsets = divmod(groups_per_value, len(offsets))
    offset_ids = torch.tensor(offsets, dtype=torch.long)
    values = torch.arange(CLASSES).expand(len(offsets), -1)
    partners = (values + offset_ids[:, None]) % CLASSES
    pairs = torch.stack((values, partners), dim=-1).reshape(-1, 2)

    repeats = torch.full_like(offset_ids, groups_per_offset)
    repeats[:extra_offsets] += 1
    pairs = pairs.repeat_interleave(repeats.repeat_interleave(CLASSES), dim=0)
    return pairs[torch.randperm(group_count, generator=generator)]


def _replace_values(split, pairs):
    edge_values = slice(PHRASE_LENGTH + 2, PHRASE_LENGTH + 5, 2)
    split["ids"][:, edge_values] = pairs.repeat_interleave(2, dim=0)
    split["target"] = pairs.flatten()


def make_held_out_data(seed, device="cpu", *, held_out_offsets=(3, 5)):
    """Return train, test, and sorted test relation pairs for modular offsets.

    Each split retains its original row budget and exactly balances the target
    classes. Each held-out ordered pair has the same number of counterfactual
    groups. With offsets 3 and 5 the test set holds out both edge orders.
    CPU generation gives a seed the same examples across training devices.
    """
    all_offsets = set(range(1, CLASSES))
    test_offsets = set(held_out_offsets)
    if not test_offsets or not test_offsets < all_offsets:
        raise ValueError(f"held_out_offsets must be a nonempty proper subset of {sorted(all_offsets)}")

    train_group_count = TRAIN_ROWS // 2
    test_group_count = TEST_ROWS // 2
    if test_group_count % (CLASSES * len(test_offsets)) != 0:
        raise ValueError("test group count must be divisible by the number of held-out value pairs")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    train = _sample_pairs(generator, train_group_count, "cpu", held_out=False)
    test = _sample_pairs(generator, test_group_count, "cpu", held_out=True)
    # The legacy sampler draws keys and noise independently of its values.
    train_pairs = _balanced_value_pairs(sorted(all_offsets - test_offsets), train_group_count, generator)
    test_pairs = _balanced_value_pairs(sorted(test_offsets), test_group_count, generator)
    _replace_values(train, train_pairs)
    _replace_values(test, test_pairs)

    pair_slice = slice(PHRASE_LENGTH + 2, PHRASE_LENGTH + 5, 2)
    train_value_pairs = {
        tuple(pair) for pair in train["ids"][0::2, pair_slice].cpu().tolist()
    }
    test_value_pairs = {
        tuple(pair) for pair in test["ids"][0::2, pair_slice].cpu().tolist()
    }
    expected_test_pairs = {
        (value, (value + offset) % CLASSES)
        for offset in test_offsets
        for value in range(CLASSES)
    }
    assert test_value_pairs == expected_test_pairs
    assert train_value_pairs.isdisjoint(test_value_pairs)

    train_y_phrases = {tuple(row) for row in train["ids"][:, PHRASE_LENGTH:].cpu().tolist()}
    test_y_phrases = {tuple(row) for row in test["ids"][:, PHRASE_LENGTH:].cpu().tolist()}
    assert train_y_phrases.isdisjoint(test_y_phrases)
    assert torch.equal(
        torch.bincount(train["target"], minlength=CLASSES),
        torch.full(
            (CLASSES,), TRAIN_ROWS // CLASSES, device=train["target"].device
        ),
    )
    assert torch.equal(
        torch.bincount(test["target"], minlength=CLASSES),
        torch.full(
            (CLASSES,), TEST_ROWS // CLASSES, device=test["target"].device
        ),
    )

    held_out_pairs = torch.unique(test_pairs, sorted=True, dim=0)
    return (
        {key: value.to(device) for key, value in train.items()},
        {key: value.to(device) for key, value in test.items()},
        [tuple(pair) for pair in held_out_pairs.tolist()],
    )
