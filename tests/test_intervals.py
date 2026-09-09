"""Across-seed t-intervals against hand-computed values (NFR-2, PR-E3)."""

from __future__ import annotations

import math

import pytest

from mulegraph.eval.intervals import seed_interval


def test_five_seeds_hand_computed() -> None:
    # values = [1, 2, 3, 4, 5]
    #   mean = 15/5 = 3
    #   deviations -2 -1 0 1 2 -> sum of squares 10; var(ddof=1) = 10/4 = 2.5
    #   sd = sqrt(2.5) = 1.5811388
    #   t.ppf(0.975, df=4) = 2.7764451
    #   half-width = 2.7764451 * 1.5811388 / sqrt(5)
    #              = 2.7764451 * 0.70710678 = 1.9632432
    mean, low, high = seed_interval([1, 2, 3, 4, 5])
    assert mean == pytest.approx(3.0)
    half = (high - low) / 2
    assert half == pytest.approx(1.9632432, abs=1e-6)
    assert low == pytest.approx(3.0 - 1.9632432, abs=1e-6)
    assert high == pytest.approx(3.0 + 1.9632432, abs=1e-6)


def test_single_seed_has_no_interval() -> None:
    """One observation carries no information about spread; a zero-width CI would lie."""
    mean, low, high = seed_interval([0.42])
    assert mean == pytest.approx(0.42)
    assert math.isnan(low)
    assert math.isnan(high)


def test_constant_vector_is_zero_width() -> None:
    mean, low, high = seed_interval([0.7] * 5)
    assert mean == pytest.approx(0.7)
    assert low == pytest.approx(0.7)
    assert high == pytest.approx(0.7)


def test_interval_is_symmetric_and_widens_with_spread() -> None:
    _, tight_low, tight_high = seed_interval([0.50, 0.51, 0.49, 0.50, 0.50])
    _, wide_low, wide_high = seed_interval([0.20, 0.80, 0.35, 0.65, 0.50])
    assert (wide_high - wide_low) > (tight_high - tight_low)


def test_wider_confidence_gives_a_wider_interval() -> None:
    # t.ppf(0.995, 4) = 4.6040949 > t.ppf(0.975, 4) = 2.7764451
    _, low95, high95 = seed_interval([1, 2, 3, 4, 5], conf=0.95)
    _, low99, high99 = seed_interval([1, 2, 3, 4, 5], conf=0.99)
    assert (high99 - low99) / 2 == pytest.approx(4.6040949 * 1.5811388 / math.sqrt(5), abs=1e-6)
    assert (high99 - low99) > (high95 - low95)


def test_three_seeds_hand_computed() -> None:
    # The reference GNN runs 3 seeds (v1b), so the n = 3 case is real.
    # values = [0.4, 0.5, 0.6]: mean 0.5, sd = sqrt(0.02/2 * ... ) by hand:
    #   deviations -0.1, 0, 0.1 -> sum sq 0.02; var(ddof=1) = 0.01; sd = 0.1
    #   t.ppf(0.975, df=2) = 4.3026527
    #   half-width = 4.3026527 * 0.1 / sqrt(3) = 4.3026527 * 0.057735027 = 0.2484130
    mean, low, high = seed_interval([0.4, 0.5, 0.6])
    assert mean == pytest.approx(0.5)
    assert (high - low) / 2 == pytest.approx(0.2484130, abs=1e-6)


def test_rejects_empty_and_bad_confidence() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        seed_interval([])
    with pytest.raises(ValueError, match="conf"):
        seed_interval([1, 2, 3], conf=1.0)
    with pytest.raises(ValueError, match="conf"):
        seed_interval([1, 2, 3], conf=0.0)
