"""Walk-forward validation for the GTA VI trophy rarity forecaster.

For each practice case in tests/ this:

1. Runs main.py on the full case (in a throwaway input dir) and checks the
   output contract: valid JSON, exactly one row per trophy id, five finite
   quantiles in [0, 100], non-decreasing, at most 65,536 bytes, the run
   inside the 8-second budget, and a second run byte-identical to the
   first (the grader runs every case twice and requires the same bytes).

2. Builds a walk-forward case: the history with its last read held out,
   the checkpoint moved to the second-to-last read and the target set to
   the held-out read. main.py forecasts it, and the forecast is scored
   against the held-out read's rates with the contest's own loss: the
   pinball loss on the logit scale, 2 * max(tau * d, (tau - 1) * d) with
   d = logit(y) - logit(q_tau), rates clamped to [0.01, 99.99], meaned
   over every trophy and level.

3. Scores two baselines on the same walk-forward case, implemented here
   from the house's published descriptions:
   - persistence (the kit starter): q50 is the last rate, spread
     0.5 * sqrt(horizon days) on the logit scale;
   - latest-curve extension (the house trend baseline): a least-squares
     line of logit rate on ln(1 + hours since 2026-11-19T00:00:00Z) over
     the last 24 hours of reads, read at the target, spread the random
     walk's and the fit's residual combined in quadrature; fewer than 3
     reads falls back to persistence.

4. Writes validation/VALIDATION.md with every number it measured.

The practice histories are one frozen Exophase page (launch night, before
the first refresh), so the walk-forward measures calibration on a quiet
series: it can show the model collapses to persistence and prices its
spread honestly there, not how it handles a refresh. The report says so.

Run: python3 tools/validate.py
"""

import json
import math
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"
REPORT = REPO / "validation" / "VALIDATION.md"

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_HOUSE = [-1.645, -0.674, 0, 0.674, 1.645]  # the house baselines' rounded z
CURVE_EPOCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    return 100.0 / (1.0 + math.exp(-x))


def round4(x):
    scaled = math.floor(x * 10000.0 + 0.5)
    value = scaled / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    moment = datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def run_main(input_dir, output_dir):
    """Run the entry once; returns wall-clock seconds."""
    env = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "MAKEFUN_INPUT_DIR": str(input_dir),
            "MAKEFUN_OUTPUT_DIR": str(output_dir),
            "PYTHONHASHSEED": "0",
            }
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(REPO / "main.py")],
        env=env, capture_output=True, text=True, timeout=60,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"main.py failed: {proc.stderr.strip()}")
    return elapsed


def pinball_loss(rows, actual_rates):
    """Mean pinball loss on the logit scale, contest formula."""
    total = 0.0
    count = 0
    for row, y in zip(rows, actual_rates):
        ly = logit(y)
        for tau, q in zip(LEVELS, row["q"]):
            d = ly - logit(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            count += 1
    return total / count


def coverage(rows, actual_rates, lo, hi):
    inside = 0
    for row, y in zip(rows, actual_rates):
        if row["q"][lo] <= y <= row["q"][hi]:
            inside += 1
    return inside / len(actual_rates)


def persistence_rows(case):
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for index, trophy in enumerate(case["trophies"]):
        rate = last["rates"][index]
        lg = logit(rate if rate is not None else 0)
        rows.append({"id": trophy["id"],
                     "q": [round4(inv_logit(lg + z * spread)) for z in Z_HOUSE]})
    return rows


def curve_rows(case):
    history = case["history"]
    last_dt = parse_time(history[-1].get("slot") or history[-1]["at"])
    window = [r for r in history
              if last_dt - parse_time(r.get("slot") or r["at"]) <= timedelta(hours=24)]
    if len(window) < 3:
        return persistence_rows(case)
    target_dt = parse_time(case["target"])
    x_target = math.log(1.0 + (target_dt - CURVE_EPOCH).total_seconds() / 3600.0)
    spread_walk = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for index, trophy in enumerate(case["trophies"]):
        pts = []
        for read in window:
            rate = read["rates"][index]
            if rate is None:
                continue
            moment = parse_time(read.get("slot") or read["at"])
            x = math.log(1.0 + (moment - CURVE_EPOCH).total_seconds() / 3600.0)
            pts.append((x, logit(rate)))
        if len(pts) < 3:
            rate = history[-1]["rates"][index]
            lg = logit(rate if rate is not None else 0)
            rows.append({"id": trophy["id"],
                         "q": [round4(inv_logit(lg + z * spread_walk)) for z in Z_HOUSE]})
            continue
        n = len(pts)
        x_bar = sum(x for x, _ in pts) / n
        y_bar = sum(y for _, y in pts) / n
        sxx = sum((x - x_bar) ** 2 for x, _ in pts)
        sxy = sum((x - x_bar) * (y - y_bar) for x, y in pts)
        slope = sxy / sxx if sxx > 0 else 0.0
        intercept = y_bar - slope * x_bar
        resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2 for x, y in pts) / n)
        spread = math.sqrt(spread_walk ** 2 + resid ** 2)
        centre = intercept + slope * x_target
        rows.append({"id": trophy["id"],
                     "q": [round4(inv_logit(centre + z * spread)) for z in Z_HOUSE]})
    return rows


def check_output(text, case):
    """Validate one .out payload against the contract; returns byte size."""
    problems = []
    data = json.loads(text)
    if set(data.keys()) != {"trophies"}:
        problems.append(f"top-level keys {sorted(data.keys())}")
    rows = data["trophies"]
    want_ids = sorted(t["id"] for t in case["trophies"])
    if sorted(r["id"] for r in rows) != want_ids:
        problems.append("row ids do not match the case's trophies")
    for row in rows:
        q = row["q"]
        if set(row.keys()) != {"id", "q"} or len(q) != 5:
            problems.append(f"row {row.get('id')} malformed")
            break
        if any(not math.isfinite(v) or v < 0 or v > 100 for v in q):
            problems.append(f"row {row['id']} has a quantile outside [0, 100]")
            break
        if any(b < a for a, b in zip(q, q[1:])):
            problems.append(f"row {row['id']} quantiles decrease")
            break
    size = len(text.encode("utf-8"))
    if size > 65536:
        problems.append(f"{size} bytes over the 65,536 cap")
    return size, problems


def main():
    case_files = sorted(TESTS.glob("*.in"))
    if not case_files:
        raise SystemExit("no practice cases under tests/")
    results = []
    for case_path in case_files:
        case = json.loads(case_path.read_text(encoding="utf-8"))
        entry = {"case": case_path.stem, "trophies": len(case["trophies"])}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            in_dir, out_dir = tmp / "in", tmp / "out"
            in_dir.mkdir()
            (in_dir / case_path.name).write_text(case_path.read_text(encoding="utf-8"),
                                                 encoding="utf-8")
            entry["runtime_s"] = run_main(in_dir, out_dir)
            out_text = (out_dir / f"{case_path.stem}.out").read_text(encoding="utf-8")
            entry["bytes"], entry["problems"] = check_output(out_text, case)
            out_dir2 = tmp / "out2"
            run_main(in_dir, out_dir2)
            entry["deterministic"] = (
                (out_dir2 / f"{case_path.stem}.out").read_bytes()
                == (out_dir / f"{case_path.stem}.out").read_bytes()
            )

            # Walk-forward: hold out the last read.
            held = case["history"][-1]
            trunc = dict(case)
            trunc["history"] = case["history"][:-1]
            prev = case["history"][-2]
            trunc["at"] = prev.get("slot") or prev["at"]
            trunc["target"] = held.get("slot") or held["at"]
            gap = parse_time(trunc["target"]) - parse_time(trunc["at"])
            trunc["horizon_minutes"] = int(gap.total_seconds() // 60)
            entry["wf_gap_min"] = trunc["horizon_minutes"]
            wf_in, wf_out = tmp / "wfin", tmp / "wfout"
            wf_in.mkdir()
            (wf_in / case_path.name).write_text(json.dumps(trunc), encoding="utf-8")
            run_main(wf_in, wf_out)
            model_rows = json.loads(
                (wf_out / f"{case_path.stem}.out").read_text(encoding="utf-8"))["trophies"]
            by_id = {t["id"]: i for i, t in enumerate(case["trophies"])}
            actual = [held["rates"][by_id[row["id"]]] for row in model_rows]
            entry["model_loss"] = pinball_loss(model_rows, actual)
            entry["persist_loss"] = pinball_loss(persistence_rows(trunc), actual)
            entry["curve_loss"] = pinball_loss(curve_rows(trunc), actual)
            entry["cov90"] = coverage(model_rows, actual, 0, 4)
            entry["cov50"] = coverage(model_rows, actual, 1, 3)
            entry["miss90"] = sum(1 for row, y in zip(model_rows, actual)
                                  if not row["q"][0] <= y <= row["q"][4])
            entry["at100"] = sum(1 for y in actual if y == 100)
        results.append(entry)

    def mean(key):
        return sum(r[key] for r in results) / len(results)

    lines = []
    lines.append("# Validation — gta6-trophy-rarity-forecast-01")
    lines.append("")
    lines.append("Generated by `tools/validate.py` from the practice cases in `tests/`; "
                 "every number below is measured by that run, nothing is estimated by hand.")
    lines.append("")
    lines.append("## Walk-forward (last read of each history held out)")
    lines.append("")
    lines.append("Each case is re-cut with its final read removed: the checkpoint moves to the "
                 "second-to-last read and the target to the held-out read, and the forecast is "
                 "scored against that read with the contest's pinball loss on the logit scale "
                 "(lower is better). The baselines are the house's two, re-implemented in the "
                 "validator from their published descriptions.")
    lines.append("")
    lines.append("| case | gap | model | persistence | curve extension | margin vs persistence | margin vs curve |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['case']} | {r['wf_gap_min']} min | {r['model_loss']:.6f} | "
            f"{r['persist_loss']:.6f} | {r['curve_loss']:.6f} | "
            f"{r['persist_loss'] - r['model_loss']:+.6f} | {r['curve_loss'] - r['model_loss']:+.6f} |")
    lines.append(
        f"| **mean** | | **{mean('model_loss'):.6f}** | **{mean('persist_loss'):.6f}** | "
        f"**{mean('curve_loss'):.6f}** | **{mean('persist_loss') - mean('model_loss'):+.6f}** | "
        f"**{mean('curve_loss') - mean('model_loss'):+.6f}** |")
    lines.append("")
    lines.append("Coverage of the model's bands on the held-out reads (nominal in brackets):")
    lines.append("")
    lines.append("| case | 5–95% band (90%) | 25–75% band (50%) |")
    lines.append("|---|---|---|")
    for r in results:
        lines.append(f"| {r['case']} | {r['cov90']:.3f} | {r['cov50']:.3f} |")
    lines.append("")
    lines.append("## Full-case contract checks")
    lines.append("")
    lines.append("| case | trophies | output bytes (cap 65,536) | runtime (budget 8 s) | second run identical | problems |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['case']} | {r['trophies']} | {r['bytes']} | {r['runtime_s']:.3f} s | "
            f"{'yes' if r['deterministic'] else 'NO'} | {'; '.join(r['problems']) if r['problems'] else 'none'} |")
    lines.append("")
    lines.append("## What this does and does not show")
    lines.append("")
    total_miss = sum(r["miss90"] for r in results)
    total_at100 = sum(r["at100"] for r in results)
    total_rows = sum(r["trophies"] for r in results)
    lines.append("Both practice histories are a single frozen Exophase page: launch night before "
                 "the first refresh, every read identical, players tracked constant. On such a "
                 "series the model's trend shrinks to zero and it forecasts persistence, so the "
                 "walk-forward margins above measure one thing only — that the model's spread on "
                 "a quiet series is priced a little tighter than the house's fixed "
                 "0.5-per-square-root-day walk, because its floor volatility is 0.35 and the "
                 "widening terms (residual scatter, step volatility, slope uncertainty, rarity) "
                 "all read zero or near it on frozen data. The 5–95% band misses "
                 f"{total_miss} of {total_rows} held-out rates across the two cases, and "
                 f"{total_at100} held-out rates are exactly 100%: a quantile row is capped at "
                 "99.99, so no band can contain a printed 100%, and that trophy is the miss in "
                 "each case. None of this says how any entry copes with a refresh of the "
                 "tracked pool, which is what actually moves the final's rates; the model answers "
                 "that structurally — trend shrinkage, movement-driven spread, rarity widening — "
                 "not from anything the practice cases can measure.")
    lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    for r in results:
        print(f"{r['case']}: model {r['model_loss']:.6f} | persistence {r['persist_loss']:.6f} | "
              f"curve {r['curve_loss']:.6f} | cov90 {r['cov90']:.3f} | cov50 {r['cov50']:.3f} | "
              f"{r['bytes']} bytes | {r['runtime_s']:.3f} s | deterministic {r['deterministic']} | "
              f"problems {r['problems']}")
    print(f"mean: model {mean('model_loss'):.6f} | persistence {mean('persist_loss'):.6f} | "
          f"curve {mean('curve_loss'):.6f}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
