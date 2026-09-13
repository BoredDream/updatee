"""重跑问题4-2，保存SAA参考轨迹，并独立重放实际执行过程。"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "src"
DATA = ROOT / "data" / "data"
OUT = ROOT / "q4" / "q4"
sys.path.insert(0, str(SRC))

import q4_q2_solver as Q42  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(actual, grid, ref_charge, ref_discharge, initial_soc, cfg):
    """独立实现4-2实时补救逻辑；不调用q2_solver.causal_dispatch。"""
    count = len(actual)
    charge = np.zeros(count)
    discharge = np.zeros(count)
    emergency = np.zeros(count)
    surplus = np.zeros(count)
    soc = np.zeros(count + 1)
    soc[0] = initial_soc
    for t in range(count):
        c = max(float(ref_charge[t]), 0.0)
        g = max(float(ref_discharge[t]), 0.0)
        simultaneous = min(c, g)
        c -= simultaneous
        g -= simultaneous
        c = min(c, cfg.interval_limit_kwh, (cfg.soc_max_kwh - soc[t]) / cfg.eta_charge)
        g = min(g, cfg.interval_limit_kwh, (soc[t] - cfg.soc_min_kwh) * cfg.eta_discharge)
        gap = float(actual[t] - (grid[t] + g - c))
        if gap > 0:
            remove_charge = min(c, gap)
            c -= remove_charge
            gap -= remove_charge
            add_discharge = max(0.0, min(
                gap,
                cfg.interval_limit_kwh - g,
                (soc[t] - cfg.soc_min_kwh) * cfg.eta_discharge - g,
            ))
            g += add_discharge
            gap -= add_discharge
            emergency[t] = max(gap, 0.0)
        elif gap < 0:
            excess = -gap
            remove_discharge = min(g, excess)
            g -= remove_discharge
            excess -= remove_discharge
            add_charge = max(0.0, min(
                excess,
                cfg.interval_limit_kwh - c,
                (cfg.soc_max_kwh - soc[t]) / cfg.eta_charge - c,
            ))
            c += add_charge
            excess -= add_charge
            surplus[t] = max(excess, 0.0)
        charge[t] = c
        discharge[t] = g
        soc[t + 1] = soc[t] + cfg.eta_charge * c - g / cfg.eta_discharge
    return {"c": charge, "g": discharge, "z": emergency, "w": surplus, "S": soc}


def max_gap(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main() -> int:
    started = time.perf_counter()
    cfg = Q42.B.Config()
    records, data = Q42.backtest(DATA, cfg=cfg)
    days = sorted(records)

    rerun = {
        "x": np.stack([records[d]["x"] for d in days]),
        "natural_x": np.stack([records[d]["natural_x"] for d in days]),
        "c": np.stack([records[d]["natural"]["c"] for d in days]),
        "g": np.stack([records[d]["natural"]["g"] for d in days]),
        "z": np.stack([records[d]["natural"]["z"] for d in days]),
        "w": np.stack([records[d]["natural"]["w"] for d in days]),
        "S": np.stack([records[d]["natural"]["S"] for d in days]),
        "S0": np.array([records[d]["S0"] for d in days]),
        "reference_charge_145": np.stack([records[d]["reference_charge"] for d in days]),
        "reference_discharge_145": np.stack([records[d]["reference_discharge"] for d in days]),
        "price_forecast_145": np.stack([records[d]["price_forecast_145"] for d in days]),
        "scenario_count": np.array([records[d]["scenario_count"] for d in days]),
        "residual_window_days": np.array([records[d]["window"] for d in days]),
        "weekday_weight": np.array([records[d]["weight"] for d in days]),
        "dates": np.array([str(data["dates"][d]) for d in days]),
    }

    delivered_path = OUT / "detail_q4-2_K30.npz"
    delivered = np.load(delivered_path, allow_pickle=False)
    compare_keys = ["x", "natural_x", "c", "g", "z", "w", "S", "S0"]
    delivered_gaps = {k: max_gap(rerun[k], delivered[k]) for k in compare_keys}

    actual = data["net_actual"].copy()
    actual[0, 0] = data["cold_start_net"][0]
    replayed = {k: [] for k in ["c", "g", "z", "w", "S"]}
    for d in days:
        out = replay(
            actual[d], records[d]["natural_x"],
            records[d]["reference_charge"][:144], records[d]["reference_discharge"][:144],
            records[d]["S0"], cfg,
        )
        for k in replayed:
            replayed[k].append(out[k])
    replayed = {k: np.stack(v) for k, v in replayed.items()}
    replay_gaps = {k: max_gap(replayed[k], rerun[k]) for k in replayed}

    OUT.mkdir(parents=True, exist_ok=True)
    trajectory_path = OUT / "q4-2_reference_trajectory_K30.npz"
    np.savez_compressed(trajectory_path, **rerun)
    elapsed = time.perf_counter() - started
    tolerance = 1e-7
    passed = max([*delivered_gaps.values(), *replay_gaps.values()]) <= tolerance
    result = {
        "status": "PASS" if passed else "FAIL",
        "created_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": elapsed,
        "scope": "2025-01-01 through 2025-12-31 continuous run; delivery period begins 2025-02-01",
        "days": len(days),
        "tolerance_kwh": tolerance,
        "trajectory_file": str(trajectory_path.relative_to(ROOT)),
        "trajectory_sha256": sha256(trajectory_path),
        "reference_shapes": {
            "charge": list(rerun["reference_charge_145"].shape),
            "discharge": list(rerun["reference_discharge_145"].shape),
        },
        "rerun_vs_delivered_max_abs_gap": delivered_gaps,
        "independent_replay_vs_rerun_max_abs_gap_kwh": replay_gaps,
        "scenario_count_range": [int(rerun["scenario_count"].min()), int(rerun["scenario_count"].max())],
        "source_hashes": {
            str((DATA / n).relative_to(ROOT)): sha256(DATA / n)
            for n in ["附件1.xlsx", "附件2.xlsx", "附件4.xlsx"]
        } | {
            str((SRC / n).relative_to(ROOT)): sha256(SRC / n)
            for n in ["q2_solver.py", "q4_q2_solver.py"]
        } | {str(delivered_path.relative_to(ROOT)): sha256(delivered_path)},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "solver": "scipy.optimize.linprog method=highs",
        },
        "claim_boundary": (
            "This verifies exact reproduction of the delivered 4-2 execution arrays and an independent "
            "replay of the causal dispatch using newly saved SAA mean reference trajectories. It does not "
            "prove uniqueness or global optimality beyond the implemented LP."
        ),
    }
    verification_path = OUT / "q4-2_reference_replay_verification.json"
    verification_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
