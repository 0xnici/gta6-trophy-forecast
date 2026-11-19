#!/usr/bin/env python3
"""Walk-forward validation for the GTA VI Trophy Rarity Forecaster.

The kit ships two practice cases (tests/p00.in, tests/p01.in) whose graded
answers live in reads the kit does not include, so this script validates the
honest way that remains: inside each case's own recorded history. For every
held-out read j (from the fourth read on), it rebuilds a case from the reads
before j, forecasts read j at the cadence horizon, and scores the forecast
against what read j actually printed. Each case is validated on its own
folds only - leave-one-case-out in the sense that no fold ever trains on,
or is scored against, the other case; the forecaster is stateless per case,
so there is nothing cross-case to leak.

Three forecasters run on identical folds:

  model        this repository's main.forecast_case
  persistence  baseline 1: median is the last training rate, quantiles are
               logit-normal with sd 0.5 * sqrt(horizon in days) - the
               house baseline's shape
  curve        baseline 2: an unshrunk least-squares line of logit rate on
               time over the trailing 24 hours, read at the target, with
               the persistence spread and the fit's residual variance
               combined; fewer than 3 training reads falls back to
               persistence

Scores follow the contest: pinball loss on the logit scale, rates in
percent clamped to [0.01, 99.99] before the logit, loss per item
2 * max(tau * d, (tau - 1) * d) with d = logit(y) - logit(q_tau).
Also reported: 90% interval coverage (share of held-out rates inside
[q05, q95]; nominal 0.90), sharpness (mean q95 - q05 in percentage
points) and typical miss (mean |y - q50| in percentage points).

Usage: python3 tools/validate.py [--tests-dir tests]
"""

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main as forecaster  # noqa: E402  (the entrypoint under test)

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = tuple(NormalDist().inv_cdf(t) for t in LEVELS)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def pinball_item(y_pct, q_pct, tau):
    d = logit_pct(y_pct) - logit_pct(q_pct)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def persistence_rows(case):
    last = case["history"][-1]
    horizon_days = case["horizon_minutes"] / 1440.0
    sd = 0.5 * math.sqrt(horizon_days)
    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        rate = last["rates"][idx]
        centre = logit_pct(rate if rate is not None else 0.01)
        rows.append({
            "id": trophy["id"],
            "q": [round(inv_logit_pct(centre + z * sd), 4) for z in Z],
        })
    return rows


def curve_rows(case):
    """Baseline 2: extend each trophy's recent logit line to the target."""
    history = case["history"]
    if len(history) < 3:
        return persistence_rows(case)
    horizon_days = case["horizon_minutes"] / 1440.0
    times = []
    t0 = None
    for read in history:
        ts = forecaster.parse_instant(read.get("slot") or read.get("at"))
        if t0 is None:
            t0 = ts
        times.append((ts - t0) / 86400.0)
    rows = []
    base = persistence_rows(case)
    for idx, trophy in enumerate(case["trophies"]):
        pts = [
            (t, logit_pct(read["rates"][idx]))
            for t, read in zip(times, history)
            if read["rates"][idx] is not None
        ]
        if len(pts) < 3:
            rows.append(base[idx])
            continue
        t_last = pts[-1][0]
        window = [(t, y) for t, y in pts if t_last - t <= 1.0]
        if len(window) < 3:
            window = pts
        n = len(window)
        x_bar = sum(t for t, _ in window) / n
        y_bar = sum(y for _, y in window) / n
        sxx = sum((t - x_bar) ** 2 for t, _ in window)
        if sxx <= 1e-12:
            rows.append(base[idx])
            continue
        slope = sum((t - x_bar) * (y - y_bar) for t, y in window) / sxx
        centre = y_bar + slope * ((t_last + horizon_days) - x_bar)
        resid_var = sum(
            (y - (y_bar + slope * (t - x_bar))) ** 2 for t, y in window
        ) / max(1, n - 2)
        sd = math.sqrt(0.25 * horizon_days + resid_var)
        rows.append({
            "id": trophy["id"],
            "q": [round(inv_logit_pct(centre + z * sd), 4) for z in Z],
        })
    return rows


def make_fold_case(case, j):
    """A synthetic case: history is reads[:j], target is read j's slot."""
    history = case["history"]
    prev_ts = forecaster.parse_instant(
        history[j - 1].get("slot") or history[j - 1].get("at"))
    next_ts = forecaster.parse_instant(
        history[j].get("slot") or history[j].get("at"))
    horizon = max(1, round((next_ts - prev_ts) / 60.0))
    return {
        "id": case["id"] + f"-fold{j}",
        "at": history[j].get("slot"),
        "horizon_minutes": horizon,
        "levels": list(LEVELS),
        "trophies": case["trophies"],
        "history": history[:j],
    }


def evaluate(rows, trophies, actual_rates):
    by_id = {row["id"]: row["q"] for row in rows}
    total = 0.0
    items = 0
    covered = 0
    counted = 0
    width = 0.0
    miss = 0.0
    for idx, trophy in enumerate(trophies):
        y = actual_rates[idx]
        if y is None:
            continue
        q = by_id[trophy["id"]]
        for tau, qv in zip(LEVELS, q):
            total += pinball_item(y, qv, tau)
            items += 1
        counted += 1
        if q[0] <= y <= q[-1]:
            covered += 1
        width += q[-1] - q[0]
        miss += abs(y - q[2])
    return {
        "pinball": total / items if items else float("nan"),
        "coverage": covered / counted if counted else float("nan"),
        "sharpness_pp": width / counted if counted else float("nan"),
        "typical_miss_pp": miss / counted if counted else float("nan"),
        "items": items,
        "trophies_scored": counted,
    }


def run(tests_dir):
    methods = {
        "model": lambda c: forecaster.forecast_case(c)["trophies"],
        "persistence": persistence_rows,
        "curve": curve_rows,
    }
    pooled = {name: [] for name in methods}
    per_case = {}
    for path in sorted(Path(tests_dir).glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        history = case["history"]
        case_stats = {name: [] for name in methods}
        for j in range(3, len(history)):
            fold = make_fold_case(case, j)
            actual = history[j]["rates"]
            for name, fn in methods.items():
                stats = evaluate(fn(fold), case["trophies"], actual)
                case_stats[name].append(stats)
                pooled[name].append(stats)
        per_case[case["id"]] = case_stats
    return per_case, pooled


def summarise(fold_stats):
    items = sum(s["items"] for s in fold_stats)
    pinball = sum(s["pinball"] * s["items"] for s in fold_stats) / items
    scored = sum(s["trophies_scored"] for s in fold_stats)
    coverage = sum(
        s["coverage"] * s["trophies_scored"] for s in fold_stats) / scored
    sharp = sum(
        s["sharpness_pp"] * s["trophies_scored"] for s in fold_stats) / scored
    miss = sum(
        s["typical_miss_pp"] * s["trophies_scored"] for s in fold_stats
    ) / scored
    return {
        "folds": len(fold_stats),
        "trophies_scored": scored,
        "pinball": pinball,
        "coverage": coverage,
        "sharpness_pp": sharp,
        "typical_miss_pp": miss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-dir", default=str(ROOT / "tests"))
    args = parser.parse_args()
    per_case, pooled = run(args.tests_dir)
    print("Walk-forward validation (folds inside each case's own history)")
    print("=" * 66)
    for case_id, case_stats in per_case.items():
        print(f"\nCase {case_id}:")
        for name, folds in case_stats.items():
            s = summarise(folds)
            print(
                f"  {name:<12} pinball {s['pinball']:.6f}  "
                f"coverage90 {s['coverage']:.3f}  "
                f"sharpness {s['sharpness_pp']:.3f} pp  "
                f"typical miss {s['typical_miss_pp']:.3f} pp  "
                f"({s['folds']} folds, {s['trophies_scored']} trophies)"
            )
    print("\nPooled:")
    for name, folds in pooled.items():
        s = summarise(folds)
        print(
            f"  {name:<12} pinball {s['pinball']:.6f}  "
            f"coverage90 {s['coverage']:.3f}  "
            f"sharpness {s['sharpness_pp']:.3f} pp  "
            f"typical miss {s['typical_miss_pp']:.3f} pp  "
            f"({s['folds']} folds, {s['trophies_scored']} trophies)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
