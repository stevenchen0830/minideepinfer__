"""Tests for SamplingParams (S1.1)."""

import pytest

from mini_deepinfer.sampling_params import INT32_MAX, SAMPLING_EPS, SamplingParams


def test_defaults_match_deepinfer():
    sp = SamplingParams(seq_seed=1)
    assert sp.temperature == 1.0
    assert sp.top_p == 1.0
    assert sp.top_k == -1
    assert sp.ignore_eos is False
    assert sp.max_tokens == 16
    assert sp.first_n_token_no_eos == 0
    assert sp.eos_id == 0
    assert sp.golden_token_ids is None


def test_arguments_are_keyword_only():
    with pytest.raises(TypeError):
        SamplingParams(0.7)  # type: ignore[misc]


@pytest.mark.parametrize("temperature", [0.0, 1e-9, SAMPLING_EPS / 2])
def test_low_temperature_means_greedy(temperature):
    sp = SamplingParams(temperature=temperature, top_p=0.5, top_k=8)
    assert sp.temperature == SAMPLING_EPS
    assert sp.top_p == 1.0
    assert sp.top_k == -1


def test_temperature_at_eps_is_left_alone():
    # The comparison is strict: exactly SAMPLING_EPS is not forced.
    sp = SamplingParams(temperature=SAMPLING_EPS, top_p=0.5, top_k=8)
    assert sp.temperature == SAMPLING_EPS
    assert sp.top_p == 0.5
    assert sp.top_k == 8


def test_validation_runs_before_greedy_forcing():
    # temperature=0 would reset top_p to 1.0, but validation sees top_p=0 first.
    with pytest.raises(ValueError, match="top_p"):
        SamplingParams(temperature=0.0, top_p=0.0)


def test_max_tokens_zero_becomes_one():
    assert SamplingParams(max_tokens=0).max_tokens == 1


def test_nan_temperature_is_rejected():
    # A deliberate difference: DeepInfer accepts NaN here.
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=float("nan"))


def test_nan_top_p_is_rejected():
    with pytest.raises(ValueError, match="top_p"):
        SamplingParams(top_p=float("nan"))


def test_unvalidated_fields_are_stored_as_given():
    sp = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        first_n_token_no_eos=5,
        eos_id=151645,
        golden_token_ids=[9, 10],
        seq_seed=1,
    )
    assert sp.ignore_eos is True
    assert sp.first_n_token_no_eos == 5
    assert sp.eos_id == 151645
    assert sp.golden_token_ids == [9, 10]


@pytest.mark.parametrize("kwargs", [{"max_tokens": 1.5}, {"max_tokens": 2.0}, {"top_k": 8.5}])
def test_float_counts_pass_here_like_deepinfer(kwargs):
    # Rejected later, when a Sequence is built (see test_sequence.py).
    SamplingParams(**kwargs)


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"temperature": -0.1}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.01}, "top_p"),
        ({"top_k": 0}, "top_k"),
        ({"top_k": -2}, "top_k"),
        ({"top_k": INT32_MAX + 1}, "top_k"),
        ({"max_tokens": -1}, "max_tokens"),
        ({"max_tokens": INT32_MAX + 1}, "max_tokens"),
    ],
)
def test_invalid_arguments_raise(kwargs, field):
    with pytest.raises(ValueError, match=field):
        SamplingParams(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_p": 1.0},
        {"top_p": 1e-6},
        {"top_k": 1},
        {"top_k": INT32_MAX},
        {"max_tokens": 1},
        {"max_tokens": INT32_MAX},
    ],
)
def test_boundary_values_are_accepted(kwargs):
    SamplingParams(**kwargs)


def test_random_seed_when_none():
    seeds = {SamplingParams().seq_seed for _ in range(8)}
    assert all(0 <= s < 2**64 for s in seeds)
    # Eight 64-bit draws colliding is practically impossible.
    assert len(seeds) == 8


@pytest.mark.parametrize(
    "seed, expected",
    [
        (0, 0),
        (12345, 12345),
        (2**64 - 1, 2**64 - 1),
        (-1, 2**64 - 1),  # masked to 64 bits, not rejected
        (2**64 + 5, 5),
        (True, 1),  # bool is an int subclass
    ],
)
def test_seed_is_masked_to_uint64(seed, expected):
    assert SamplingParams(seq_seed=seed).seq_seed == expected


@pytest.mark.parametrize("seed", [1.5, "42", [1]])
def test_non_int_seed_raises_type_error(seed):
    with pytest.raises(TypeError):
        SamplingParams(seq_seed=seed)


def test_repr_lists_main_fields():
    text = repr(
        SamplingParams(temperature=0.7, top_k=8, ignore_eos=True, max_tokens=4, seq_seed=3)
    )
    for part in (
        "temperature=0.7",
        "top_p=1.0",
        "top_k=8",
        "ignore_eos=True",
        "max_tokens=4",
        "seq_seed=3",
    ):
        assert part in text
