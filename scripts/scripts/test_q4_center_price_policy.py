"""Q4-3 方案 A 回归：共享项用中心预测价，追索项用场景价。"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
os.environ["Q3_DATA"] = str(ROOT / "data" / "data")
sys.path.insert(0, str(ROOT / "src" / "src"))

import q4_solver as M  # noqa: E402


EXPECTED_POLICY = {
    "shared_price_basis": "center_forecast",
    "recourse_price_basis": "scenario_price",
    "objective_type": "hybrid_center_and_scenario_recourse",
    "scenario_price_centering": "not_recentered",
}


def assert_close(actual, expected, label: str) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-12):
        delta = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        raise AssertionError(f"{label} 不一致，最大绝对误差 {delta:.3e}")


def capture_objectives(run_case):
    captured = []
    original = M.linprog

    def fake_linprog(c, **_kwargs):
        captured.append(np.asarray(c, dtype=float).copy())
        return SimpleNamespace(
            x=np.zeros(len(c), dtype=float),
            success=True,
            status=0,
            message="objective-capture stub",
        )

    M.linprog = fake_linprog
    try:
        run_case()
    finally:
        M.linprog = original
    if len(captured) != 1:
        raise AssertionError(f"预期截获 1 个目标向量，实际 {len(captured)} 个")
    return captured[0]


def test_policy_metadata() -> None:
    policy = M.price_objective_policy()
    if policy != EXPECTED_POLICY:
        raise AssertionError(f"价格口径元数据不一致：{policy!r}")
    policy["shared_price_basis"] = "mutated"
    if M.price_objective_policy() != EXPECTED_POLICY:
        raise AssertionError("price_objective_policy() 未返回独立副本")


def test_stage0_shared_and_recourse_coefficients() -> None:
    K, H, tn = 2, M.T + 1, M.TSTAGE[1]
    p_center = np.linspace(0.20, 0.90, M.T)
    scenario_prices = [
        np.r_[0.31, p_center + 0.20],
        np.r_[0.47, p_center + 0.40],
    ]
    scenario_mean = np.mean(np.stack(scenario_prices), axis=0)[1:]
    if np.allclose(p_center, scenario_mean):
        raise AssertionError("测试数据未区分中心价与场景均价")
    zeros = [np.zeros(H), np.zeros(H)]

    obj = capture_objectives(
        lambda: M.stage0_lp4_145(
            committed_q=0.0,
            S_cur=6000.0,
            scenL=zeros,
            scenG=zeros,
            scenP=scenario_prices,
            p_center=p_center,
            lam=M.LAM,
            commit_end=tn,
        )
    )

    expected_shared = np.r_[p_center[:tn], 0.5 * p_center[tn:]]
    assert_close(obj[:M.T], expected_shared, "0:00 共享合同系数")
    if np.allclose(obj[:tn], scenario_mean[:tn], rtol=0.0, atol=1e-12):
        raise AssertionError("0:00 共享合同系数错误地退化为场景均价")

    n_recourse = M.T - tn
    per_scenario = 2 * n_recourse + 5 * H
    for k, prices in enumerate(scenario_prices):
        offset = M.T + k * per_scenario
        q_offset = offset
        up_offset = offset + n_recourse
        emergency_offset = offset + 2 * n_recourse + 2 * H
        assert_close(
            obj[q_offset:q_offset + n_recourse],
            0.5 * prices[tn + 1:] / K,
            f"0:00 场景{k}未来合同系数",
        )
        assert_close(
            obj[up_offset:up_offset + n_recourse],
            prices[tn + 1:] / K,
            f"0:00 场景{k}超计划系数",
        )
        assert_close(
            obj[emergency_offset:emergency_offset + H],
            M.ALPHA * prices / K,
            f"0:00 场景{k}紧急购电系数",
        )


def test_intraday_shared_and_recourse_coefficients() -> None:
    K, m = 2, 1
    tm, tn = M.TSTAGE[m], M.TSTAGE[2]
    n_stage = tn - tm
    n_future = M.T - tn
    n_horizon = M.T - tm
    p_center = np.linspace(0.25, 0.95, M.T)
    scenario_prices = [p_center + 0.15, p_center + 0.45]
    scenario_mean = np.mean(np.stack(scenario_prices), axis=0)
    if np.allclose(p_center, scenario_mean):
        raise AssertionError("测试数据未区分中心价与场景均价")
    zeros = [np.zeros(M.T), np.zeros(M.T)]

    obj = capture_objectives(
        lambda: M.stage_lp4(
            m=m,
            x_ref=np.zeros(M.T),
            S_cur=6000.0,
            scenL=zeros,
            scenG=zeros,
            scenP=scenario_prices,
            p_center=p_center,
            adjust=True,
            lam=M.LAM,
            commit_end=tn,
        )
    )

    assert_close(obj[:n_stage], 0.5 * p_center[tm:tn], "日内共享合同系数")
    assert_close(obj[n_stage:2 * n_stage], p_center[tm:tn], "日内超计划系数")
    if np.allclose(obj[n_stage:2 * n_stage], scenario_mean[tm:tn], rtol=0.0, atol=1e-12):
        raise AssertionError("日内共享合同系数错误地退化为场景均价")

    base = 2 * n_stage
    per_scenario = 2 * n_future + 5 * n_horizon
    for k, prices in enumerate(scenario_prices):
        scenario_offset = base + k * per_scenario
        future_q_offset = scenario_offset
        future_up_offset = scenario_offset + n_future
        emergency_offset = scenario_offset + 2 * n_future + 2 * n_horizon
        assert_close(
            obj[future_q_offset:future_q_offset + n_future],
            0.5 * prices[tn:] / K,
            f"日内场景{k}未来合同系数",
        )
        assert_close(
            obj[future_up_offset:future_up_offset + n_future],
            prices[tn:] / K,
            f"日内场景{k}超计划系数",
        )
        assert_close(
            obj[emergency_offset:emergency_offset + n_horizon],
            M.ALPHA * prices[tm:] / K,
            f"日内场景{k}紧急购电系数",
        )


def test_solve_day_call_chain_and_reference_result() -> None:
    """用已落盘的首个交付日证明调用链传中心价，且数值结果没有变化。"""
    reference_path = ROOT / "q4" / "q4" / "detail_q4-3_K30.npz"
    if not reference_path.exists():
        raise AssertionError(f"缺少单日数值回归基准：{reference_path}")

    d = 31  # 2025-02-01
    calls = {}
    original_stage0 = M.stage0_lp4_145
    original_stage = M.stage_lp4

    def traced_stage0(committed_q, S_cur, scenL, scenG, scenP, p_center, lam,
                      commit_end=M.TSTAGE[1]):
        calls[0] = {
            "p_center": np.asarray(p_center).copy(),
            "scenario_mean": np.mean(np.stack(scenP), axis=0)[1:],
        }
        return original_stage0(
            committed_q, S_cur, scenL, scenG, scenP, p_center, lam,
            commit_end=commit_end,
        )

    def traced_stage(m, x_ref, S_cur, scenL, scenG, scenP, p_center,
                     adjust=True, alpha=M.ALPHA, lam=0.478, commit_end=None):
        calls[m] = {
            "p_center": np.asarray(p_center).copy(),
            "scenario_mean": np.mean(np.stack(scenP), axis=0),
        }
        return original_stage(
            m, x_ref, S_cur, scenL, scenG, scenP, p_center,
            adjust=adjust, alpha=alpha, lam=lam, commit_end=commit_end,
        )

    with np.load(reference_path) as reference:
        if str(reference["dates"][d]) != "2025-02-01":
            raise AssertionError("Q4-3 单日回归基准的日期索引已变化")
        S0 = float(reference["S0"][d])
        committed_q = float(reference["natural_q"][d, 0])
        expected = {
            "x": reference["x"][d].copy(),
            "q": reference["q"][d].copy(),
            **{name: reference[name][d].copy() for name in ("c", "g", "z", "w", "S")},
        }

    M.clear_price_caches()
    M.stage0_lp4_145 = traced_stage0
    M.stage_lp4 = traced_stage
    try:
        x, q, _out, natural, _S_end = M.solve_day4(
            d, S0, committed_q, K=30, stages=(0, 1, 2, 3)
        )
    finally:
        M.stage0_lp4_145 = original_stage0
        M.stage_lp4 = original_stage

    if set(calls) != {0, 1, 2, 3}:
        raise AssertionError(f"未覆盖全部 Q4-3 决策阶段：{sorted(calls)}")
    for m in range(4):
        assert_close(calls[m]["p_center"], M.price_hat(d, m), f"阶段{m}调用链中心价")
        gap = float(np.max(np.abs(calls[m]["p_center"] - calls[m]["scenario_mean"])))
        if gap <= 1e-6:
            raise AssertionError(f"阶段{m}的测试日中心价与场景均价未形成有效区分")

    assert_close(x, expected["x"], "首交付日计划购电量")
    assert_close(q, expected["q"], "首交付日最终合同量")
    for name in ("c", "g", "z", "w", "S"):
        assert_close(natural[name], expected[name], f"首交付日自然日{name}轨迹")


if __name__ == "__main__":
    test_policy_metadata()
    test_stage0_shared_and_recourse_coefficients()
    test_intraday_shared_and_recourse_coefficients()
    test_solve_day_call_chain_and_reference_result()
    print("PASS: Q4-3 方案A中心价/场景追索价格口径回归测试")
