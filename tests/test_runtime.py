"""Tests for shared runtime services."""

import random

import numpy as np
import torch

from runtime import Runtime


def test_runtime_round_trips_rank_rng_state(tmp_path):
    runtime = Runtime("cpu", seed=123)
    runtime.save_rng_state(tmp_path)
    expected = torch.rand(4)

    runtime.rng.seed = 999
    torch.rand(8)
    assert runtime.load_rng_state(tmp_path) is True

    torch.testing.assert_close(torch.rand(4), expected)


def test_runtime_load_rng_state_is_optional(tmp_path):
    runtime = Runtime("cpu")

    assert runtime.load_rng_state(tmp_path) is False


def test_runtime_round_trips_all_rng_sources(tmp_path):
    runtime = Runtime("cpu", seed=123)
    runtime.save_rng_state(tmp_path)

    expected = (random.random(), np.random.rand(4), torch.rand(4))
    assert runtime.load_rng_state(tmp_path) is True
    actual = (random.random(), np.random.rand(4), torch.rand(4))

    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2])


def test_runtime_seed_is_reproducible():
    first_runtime = Runtime("cpu", seed=321)
    first = (random.random(), np.random.rand(4), torch.rand(4))

    Runtime("cpu", seed=321)
    second = (random.random(), np.random.rand(4), torch.rand(4))

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2])
    assert first_runtime.rng.seed == 321
