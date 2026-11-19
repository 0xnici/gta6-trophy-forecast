"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Reads every case file in the input directory and writes, for each trophy in
the case, five quantiles (5/25/50/75/95%) of its Exophase earn rate at the
case's target time. Pure Python 3.13 standard library, deterministic, no
network, no clock: every time used comes from the case file itself.

Model, per trophy
-----------------
Work on the logit scale, where a rate's moves are closer to symmetric and a
miss from 1% to 2% counts like one from 33% to 50% (the scoring scale).

1. Location. Fit a weighted least-squares line of logit(rate) on time over
   the trophy's reads (weights: recency, with a 3-day half-life, times the
   square root of the tracked-player share, so later, larger reads count
   more). Then shrink the fitted line's forecast back toward persistence --
   the last observed rate -- by the factor b^2 / (b^2 + se_b^2): a slope
   that is statistically indistinguishable from zero earns no trust, so a
   quiet series forecasts exactly its last rate, while a series with a
   clear, well-measured trend keeps most of it. The trend's contribution
   is also capped at +/-2 logit units so a short noisy series can never
   extrapolate to an absurd rate.

2. Spread. A normal distribution on the logit scale whose standard
   deviation combines, in quadrature:
     - a floor volatility of 0.35 logit units per square-root day of
       forecast gap (the gap runs from the last read to the target, not
       from the checkpoint: that is the stretch with no observations);
     - the series' own movement: its weighted residual scatter around the
       fitted line, projected over the gap, combined with a random-walk
       volatility estimated from the read-to-read logit differences (a
       line fit absorbs a steady climb into its slope, but a refresh of
       the tracked pool moves rates in steps, and the differences are
       what see those steps) -- a series that has actually been jumping
       between reads gets a wider band than a frozen one;
     - the standard error of the shrunk slope times the gap, so an
       uncertain trend widens the band instead of pretending precision.
   The total is then widened by up to 15% for the rarest trophies (factor
   1 + 0.15 * (1 - p) at last rate p): their earn counts are small, so a
   refresh of the tracked pool moves their rates most in relative terms.

3. Quantiles. mu + z_tau * sd pushed back through the logistic, clamped
   to [0.01, 99.99], rounded to 4 decimals (half up) and forced
   non-decreasing, so the five numbers are a valid quantile row by
   construction.

Input/output contract (from the kit): the grader runs
`python3 -P -s /bot/main.py` with one case in /input; outside the grader,
MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output.
Each <id>.in is JSON {"schema": "makefun/gta6-trophies-case/v1", "id",
"at", "horizon_minutes", "target", "levels", "game", "trophies",
"history", "notes"}; history reads are oldest first, each {"slot", "at",
"players_tracked", "trophies_earned", "rates"} with rates in the order of
trophies (null where a read lacked the trophy). Each <id>.out is
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [NormalDist().inv_cdf(level) for level in LEVELS]

RATE_LO = 0.01   # the scorer clamps rates into [0.01, 99.99] before the logit
RATE_HI = 99.99

SIGMA_DAY = 0.35          # floor logit volatility per square-root day of gap
RECENCY_HALF_LIFE = 3.0   # days; older reads weigh half as much per this span
MAX_DRIFT = 2.0           # cap on the trend's move away from the last rate
MIN_SD = 0.02             # never claim a thinner band than this
RARITY_WIDEN = 0.15       # extra spread factor for a trophy nobody has earned


def clamp_rate(pct):
    return min(RATE_HI, max(RATE_LO, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return 100.0 * ex / (1.0 + ex)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int so the
    JSON writes 100 rather than 100.0. Keeps output bytes stable."""
    scaled = math.floor(x * 10000.0 + 0.5)
    value = scaled / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp like 2026-11-19T04:30:00Z (a
    fractional-second part is allowed) into an aware datetime."""
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    moment = datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def weighted_line(points):
    """Weighted least squares for y ~ a + b*t over (t, y, w) points.

    Returns (a, b, resid_sd, se_b): the intercept and slope, the weighted
    root-mean-square residual, and the slope's standard error. Caller
    guarantees at least 3 points spanning a nonzero time range.
    """
    sw = sum(w for _, _, w in points)
    t_bar = sum(w * t for t, _, w in points) / sw
    y_bar = sum(w * y for _, y, w in points) / sw
    sxx = sum(w * (t - t_bar) ** 2 for t, _, w in points)
    sxy = sum(w * (t - t_bar) * (y - y_bar) for t, y, w in points)
    slope = sxy / sxx
    intercept = y_bar - slope * t_bar
    resid_var = sum(w * (y - (intercept + slope * t)) ** 2 for t, y, w in points) / sw
    resid_sd = math.sqrt(resid_var)
    se_slope = math.sqrt(resid_var / sxx) if sxx > 0 else float("inf")
    return intercept, slope, resid_sd, se_slope


def forecast_trophy(points, t_target, t_last, gap_days, last_rate):
    """(mu, sd) of the trophy's logit rate at the target.

    points: (t_days, logit_rate, weight) for every read that listed the
    trophy, oldest first. last_rate: its rate in the latest such read, or
    None when no read listed it.
    """
    base_sd = SIGMA_DAY * math.sqrt(max(gap_days, 0.0))
    if not points:
        # Never observed: sit on the clamp floor with the plain gap spread.
        return logit(RATE_LO), max(base_sd, MIN_SD)

    mu_persist = points[-1][1]
    mu = mu_persist
    resid_sd = 0.0
    trend_sd = 0.0
    # Random-walk volatility from the read-to-read logit differences:
    # sum of squared increments over sum of their day spans is the
    # per-day variance of a walk that moves the way this series has.
    step_var_day = 0.0
    if len(points) >= 2:
        sq = 0.0
        span_days = 0.0
        for (t0, y0, _), (t1, y1, _) in zip(points, points[1:]):
            dt = t1 - t0
            if dt > 0:
                sq += (y1 - y0) ** 2
                span_days += dt
        if span_days > 0:
            step_var_day = sq / span_days
    span = points[-1][0] - points[0][0]
    if len(points) >= 3 and span > 0:
        intercept, slope, resid_sd, se_slope = weighted_line(points)
        # Shrinkage: trust the slope in proportion to how clearly it stands
        # out from its own standard error. A flat series has slope ~ 0 and
        # collapses to persistence exactly (and a perfectly flat series has
        # slope and standard error both 0, which is also a shrink of 0).
        denom = slope * slope + se_slope * se_slope
        shrink = (slope * slope) / denom if denom > 0 else 0.0
        drift = shrink * (intercept + slope * t_target - mu_persist)
        drift = min(MAX_DRIFT, max(-MAX_DRIFT, drift))
        mu = mu_persist + drift
        trend_sd = shrink * se_slope * max(gap_days, 0.0)

    gap = max(gap_days, 0.0)
    noise_sd = math.sqrt(resid_sd ** 2 * (1.0 + gap) + step_var_day * gap)
    sd = math.sqrt(base_sd ** 2 + noise_sd ** 2 + trend_sd ** 2)
    if last_rate is not None:
        share = clamp_rate(last_rate) / 100.0
        sd *= 1.0 + RARITY_WIDEN * (1.0 - share)
    return mu, max(sd, MIN_SD)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")

    trophies = case["trophies"]
    target_dt = parse_time(case["target"])
    read_times = [parse_time(read.get("slot") or read["at"]) for read in history]
    origin = read_times[0]

    def days(moment):
        return (moment - origin).total_seconds() / 86400.0

    t_target = days(target_dt)
    t_last = days(read_times[-1])
    gap_days = max(t_target - t_last, 0.0)
    players_max = max((read.get("players_tracked") or 0) for read in history) or 1

    rows = []
    for index, trophy in enumerate(trophies):
        points = []
        last_rate = None
        for read, moment in zip(history, read_times):
            rate = read["rates"][index]
            if rate is None:
                continue
            t = days(moment)
            players = read.get("players_tracked") or 0
            weight = math.sqrt(players / players_max) if players > 0 else 1.0
            weight *= 0.5 ** ((t_last - t) / RECENCY_HALF_LIFE)
            points.append((t, logit(rate), weight))
            last_rate = rate
        mu, sd = forecast_trophy(points, t_target, t_last, gap_days, last_rate)
        quantiles = []
        previous = RATE_LO
        for z in Z:
            q = round4(clamp_rate(inv_logit(mu + z * sd)))
            q = max(q, previous)
            quantiles.append(q)
            previous = q
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Name order, so two runs over the same inputs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
