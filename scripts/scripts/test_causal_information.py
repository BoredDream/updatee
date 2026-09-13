"""Q3/Q4-3 information-cutoff regression tests.

The test perturbs values that are still unknown at each publication time.  A
causal forecast/scenario generator must remain bit-for-bit unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import q3_multistage as Q3  # noqa: E402
import q4_solver as Q4  # noqa: E402


def max_diff(left, right) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def assert_same(name: str, before, after, tol: float = 1e-12) -> None:
    error = max_diff(before, after)
    if error > tol:
        raise AssertionError(f"{name}: future-data perturbation changed output by {error:.3e}")
    print(f"[OK] {name}: max change {error:.3e}")


def q3_stage0_test(day: int = 100) -> None:
    Q3.clear_forecast_caches()
    before_pv = Q3.pv_hat(day, 0).copy()
    before_load = Q3.load_hat(day, 0).copy()
    before_sl, before_sg = Q3.scenarios0_145(day, 8)
    old_g = float(Q3.GKW[day - 1, -1])
    old_l = float(Q3.L[day - 1, -1])
    try:
        # Previous template row's final column is today's [00:00,00:10), unknown at 00:00.
        Q3.GKW[day - 1, -1] = old_g + 100000.0
        Q3.L[day - 1, -1] = old_l + 100000.0
        Q3.clear_forecast_caches()
        after_pv = Q3.pv_hat(day, 0)
        after_load = Q3.load_hat(day, 0)
        after_sl, after_sg = Q3.scenarios0_145(day, 8)
    finally:
        Q3.GKW[day - 1, -1] = old_g
        Q3.L[day - 1, -1] = old_l
        Q3.clear_forecast_caches()
    assert_same("Q3 00:00 PV forecast", before_pv, after_pv)
    assert_same("Q3 00:00 load forecast", before_load, after_load)
    assert_same("Q3 00:00 load scenarios", np.stack(before_sl), np.stack(after_sl))
    assert_same("Q3 00:00 PV scenarios", np.stack(before_sg), np.stack(after_sg))


def q3_intraday_test(day: int = 100) -> None:
    for stage in (1, 2, 3):
        issue = Q3.TSTAGE[stage]
        expected_anchor = float(Q3.GKW[day, issue - 1])
        actual_anchor = float(Q3.pv_from_forecast(day, stage)[issue])
        assert_same(f"Q3 stage {stage} last-completed PV anchor", [expected_anchor], [actual_anchor])
        Q3.clear_forecast_caches()
        before_pv = Q3.pv_hat(day, stage).copy()
        before_load = Q3.load_hat(day, stage).copy()
        before_sl, before_sg = Q3.scenarios(day, stage, 8)
        old_g = Q3.GKW[day, issue:].copy()
        old_l = Q3.L[day, issue:].copy()
        try:
            Q3.GKW[day, issue:] += 100000.0
            Q3.L[day, issue:] += 100000.0
            Q3.clear_forecast_caches()
            after_pv = Q3.pv_hat(day, stage)
            after_load = Q3.load_hat(day, stage)
            after_sl, after_sg = Q3.scenarios(day, stage, 8)
        finally:
            Q3.GKW[day, issue:] = old_g
            Q3.L[day, issue:] = old_l
            Q3.clear_forecast_caches()
        assert_same(f"Q3 stage {stage} PV forecast", before_pv, after_pv)
        assert_same(f"Q3 stage {stage} load forecast", before_load, after_load)
        assert_same(f"Q3 stage {stage} load scenarios", np.stack(before_sl), np.stack(after_sl))
        assert_same(f"Q3 stage {stage} PV scenarios", np.stack(before_sg), np.stack(after_sg))


def q4_price_test(day: int = 100) -> None:
    Q4.clear_price_caches()
    before = Q4.price_hat(day, 0).copy()
    before_scenarios = Q4.scenarios4_145(day, 8)[2]
    old_midnight = float(Q4.PMAT[day - 1, -1])
    try:
        Q4.PMAT[day - 1, -1] = old_midnight + 1000.0
        Q4.clear_price_caches()
        after = Q4.price_hat(day, 0)
        after_scenarios = Q4.scenarios4_145(day, 8)[2]
    finally:
        Q4.PMAT[day - 1, -1] = old_midnight
        Q4.clear_price_caches()
    assert_same("Q4-3 00:00 price forecast", before, after)
    assert_same("Q4-3 00:00 price scenarios", np.stack(before_scenarios), np.stack(after_scenarios))

    for stage in (1, 2, 3):
        issue = Q4.TSTAGE[stage]
        Q4.clear_price_caches()
        before = Q4.price_hat(day, stage).copy()
        before_scenarios = Q4.scenarios4(day, stage, 8)[2]
        old = Q4.PMAT[day, issue:].copy()
        try:
            Q4.PMAT[day, issue:] += 1000.0
            Q4.clear_price_caches()
            after = Q4.price_hat(day, stage)
            after_scenarios = Q4.scenarios4(day, stage, 8)[2]
        finally:
            Q4.PMAT[day, issue:] = old
            Q4.clear_price_caches()
        assert_same(f"Q4-3 stage {stage} price forecast", before, after)
        assert_same(f"Q4-3 stage {stage} price scenarios", np.stack(before_scenarios), np.stack(after_scenarios))


if __name__ == "__main__":
    q3_stage0_test()
    q3_intraday_test()
    q4_price_test()
    print("Causal information-cutoff regression tests passed.")
