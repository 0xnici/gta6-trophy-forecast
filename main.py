#!/usr/bin/env python3
"""GTA VI Trophy Rarity Forecaster.

Reads one make.fun case file (makefun/gta6-trophies-case/v1) per run from the
input directory and writes, for every trophy in the case, five quantiles
(5/25/50/75/95 percent) of its Exophase earn rate at the case target instant.

Method, in one paragraph: work on the logit scale, where a move from 1% to 2%
counts about as much as a move from 33% to 50%. For each trophy, fit a
recency-weighted straight line to the recent logit rates, then shrink that
slope toward zero (persistence) by an amount that depends on how much
evidence backs it: many reads and a slope that is large compared with its
own standard error survive, a wobble on three reads does not. The shrunken
slope, mildly saturated so a hot first day is not extrapolated linearly for
a week, gives the forecast location. Around that location sit a Beta law
fitted by moments: its spread grows with the square root of the horizon,
with the slope's own uncertainty, and with rarity, because a rate built on
a handful of earners can jump when the tracked crowd refreshes. Quantiles
come from inverting the regularised incomplete beta function, implemented
below with a continued fraction and bisection, using nothing outside the
Python standard library.

The program is deterministic: no randomness, no clock reads, no network.
Input directory: $MAKEFUN_INPUT_DIR, else --input-dir, else /input.
Output directory: $MAKEFUN_OUTPUT_DIR, else --output-dir, else /output.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Rates are printed to two decimals, so 0 and 100 carry no information about
# how far past the edge the truth sits. Clip proportions to the same window
# the scorer uses, expressed as a proportion: [0.0001, 0.9999].
P_LO = 0.0001
P_HI = 0.9999

# Trend fitting.
WINDOW_DAYS = 3.0          # only reads inside this trailing window fit the line
HALF_LIFE_DAYS = 1.5       # recency weight halves this fast going backwards
MIN_POINTS_FOR_TREND = 3   # fewer usable reads than this means: no trend
SHRINK_READS = 6.0         # evidence scale in the read-count shrinkage factor
SLOPE_CAP = 2.5            # logit units per day; a harder move is not believed
SATURATION_DAYS = 10.0     # horizon damping: growth bends after this scale

# Spread (logit scale, before the moment match to a Beta law).
SIGMA_FLOOR = 0.07         # irreducible doubt, even a minute ahead
SIGMA_WALK = 0.42          # random-walk growth per square-root day
RARITY_KAPPA = 0.5         # extra variance for thinly-earned trophies


def clip_prop(p):
    if p < P_LO:
        return P_LO
    if p > P_HI:
        return P_HI
    return p


def to_logit(pct):
    """Percent (0..100) to logit, clipped consistently."""
    p = clip_prop(pct / 100.0)
    return math.log(p / (1.0 - p))


def from_logit(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def parse_instant(text):
    """Parse an ISO-8601 UTC instant like 2026-11-19T03:00:00.309Z."""
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# --------------------------------------------------------------------------
# Regularised incomplete beta and its inverse (stdlib only).
# The continued fraction is the classic Lentz iteration; the inverse is
# bisection, which is slow but bulletproof, and 55 trophies x 5 levels x
# ~60 evaluations is still far inside the time budget.
# --------------------------------------------------------------------------

def _beta_fraction(a, b, x):
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < 3e-14:
            break
    return h


def beta_cdf(x, a, b):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_fraction(a, b, x) / a
    return 1.0 - front * _beta_fraction(b, a, 1.0 - x) / b


def beta_ppf(prob, a, b):
    """The x with beta_cdf(x, a, b) == prob, by bisection on [0, 1]."""
    if prob <= 0.0:
        return 0.0
    if prob >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if beta_cdf(mid, a, b) < prob:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def beta_from_mean_logit_sd(mean_p, logit_sd):
    """Moment-match a Beta law.

    mean_p is the forecast mean as a proportion. logit_sd is the spread on
    the logit scale; the delta method turns it into a variance on the
    proportion scale, var = (m (1-m))^2 * sd^2, and the Beta parameters
    follow from the usual mean/variance identities. The variance is capped
    just under the Bernoulli bound m(1-m) so the parameters stay positive.
    """
    m = clip_prop(mean_p)
    var = (m * (1.0 - m)) ** 2 * logit_sd * logit_sd
    bound = 0.98 * m * (1.0 - m)
    if var > bound:
        var = bound
    if var <= 1e-15:
        var = 1e-15
    common = m * (1.0 - m) / var - 1.0
    if common < 0.05:
        common = 0.05
    alpha = m * common
    beta = (1.0 - m) * common
    if alpha < 0.02:
        alpha = 0.02
    if beta < 0.02:
        beta = 0.02
    return alpha, beta


# --------------------------------------------------------------------------
# Per-trophy trend: recency-weighted least squares on logit rates, with
# the slope shrunk toward persistence.
# --------------------------------------------------------------------------

def fit_trend(times_days, logits, players_last):
    """Return (slope_per_day, slope_se, n_eff) for one trophy's series.

    times_days are days since the first read of the whole case history;
    only the trailing WINDOW_DAYS are fitted. Weights decay with a
    HALF_LIFE_DAYS half-life so the freshest reads speak loudest.
    """
    if not times_days:
        return 0.0, 0.0, 0.0
    t_last = times_days[-1]
    pts = [
        (t, y) for t, y in zip(times_days, logits)
        if t_last - t <= WINDOW_DAYS
    ]
    if len(pts) < MIN_POINTS_FOR_TREND:
        return 0.0, 0.0, float(len(pts))
    weights = [0.5 ** ((t_last - t) / HALF_LIFE_DAYS) for t, _ in pts]
    sw = sum(weights)
    if sw <= 0:
        return 0.0, 0.0, float(len(pts))
    x_bar = sum(w * t for w, (t, _) in zip(weights, pts)) / sw
    y_bar = sum(w * y for w, (_, y) in zip(weights, pts)) / sw
    sxx = sum(w * (t - x_bar) ** 2 for w, (t, _) in zip(weights, pts))
    if sxx <= 1e-12:
        # Every fitted read shares one instant (or one value of time):
        # no slope is identifiable.
        return 0.0, 0.0, sw
    sxy = sum(
        w * (t - x_bar) * (y - y_bar)
        for w, (t, y) in zip(weights, pts)
    )
    slope = sxy / sxx
    resid = [
        y - (y_bar + slope * (t - x_bar)) for t, y in pts
    ]
    dof = max(1.0, sw - 2.0)
    s2 = sum(w * r * r for w, r in zip(weights, resid)) / dof
    slope_se = math.sqrt(s2 / sxx) if s2 > 0 else 0.0
    return slope, slope_se, sw


def shrink_slope(slope, slope_se, n_eff):
    """Pull a fitted slope toward zero (persistence).

    Two doubts multiply: too few reads (n_eff against SHRINK_READS) and a
    slope that is small next to its own standard error. A slope many
    standard errors from zero, backed by many reads, passes through.
    """
    if slope == 0.0:
        return 0.0
    by_count = n_eff / (n_eff + SHRINK_READS)
    signal = (slope * slope) / (slope * slope + slope_se * slope_se + 0.01)
    kept = slope * by_count * signal
    if kept > SLOPE_CAP:
        kept = SLOPE_CAP
    if kept < -SLOPE_CAP:
        kept = -SLOPE_CAP
    return kept


# --------------------------------------------------------------------------
# Forecast one trophy.
# --------------------------------------------------------------------------

def forecast_trophy(series, horizon_days, players_last):
    """series: list of (time_days, rate_pct) with no missing rates.

    Returns the five quantiles in percent, non-decreasing.
    """
    if not series:
        # No observation at all: a wide, low prior rather than a crash.
        alpha, beta = beta_from_mean_logit_sd(0.01, 1.2)
        return [round4(100.0 * beta_ppf(tau, alpha, beta)) for tau in LEVELS]
    times = [t for t, _ in series]
    logits = [to_logit(r) for _, r in series]
    last_logit = logits[-1]

    slope, slope_se, n_eff = fit_trend(times, logits, players_last)
    kept_slope = shrink_slope(slope, slope_se, n_eff)

    # Saturating extrapolation: growth bends over SATURATION_DAYS instead
    # of running linearly for the whole horizon.
    bend = horizon_days / (1.0 + horizon_days / SATURATION_DAYS)
    mu_logit = last_logit + kept_slope * bend
    mean_p = from_logit(mu_logit)

    # Spread: floor + random walk in sqrt(horizon) + trend doubt + rarity.
    trend_var = (slope_se * bend) ** 2
    if trend_var > 1.0:
        trend_var = 1.0
    # Rarity risk is refresh risk: a thinly-earned rate only moves when the
    # tracked crowd is re-read, and the chance of that grows with the
    # horizon, so the term is gated by a saturating horizon factor rather
    # than charged in full for a half-hour forecast.
    refresh_gate = horizon_days / (horizon_days + 0.25)
    rarity_var = refresh_gate * RARITY_KAPPA * (1.0 - mean_p) / (
        max(1.0, float(players_last)) * mean_p + 5.0
    )
    var_logit = (
        SIGMA_FLOOR ** 2
        + SIGMA_WALK ** 2 * horizon_days
        + trend_var
        + rarity_var
    )
    logit_sd = math.sqrt(var_logit)

    alpha, beta = beta_from_mean_logit_sd(mean_p, logit_sd)
    qs = [100.0 * beta_ppf(tau, alpha, beta) for tau in LEVELS]
    # The Beta law is a distribution, so its quantiles already increase;
    # enforce it anyway against any numerical wobble, then round.
    out = []
    prev = 0.0
    for q in qs:
        q = min(100.0, max(0.0, q))
        if q < prev:
            q = prev
        out.append(round4(q))
        prev = out[-1]
    return out


def round4(x):
    """Round to 4 decimal places, half up; whole numbers stay whole."""
    scaled = math.floor(x * 10000.0 + 0.5) / 10000.0
    if scaled == int(scaled):
        return int(scaled)
    return scaled


def forecast_case(case):
    history = case.get("history") or []
    trophies = case.get("trophies") or []
    horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
    if horizon_days < 0:
        horizon_days = 0.0

    # Read instants, in days since the first read, for trend fitting.
    read_days = []
    t0 = None
    for read in history:
        stamp = read.get("slot") or read.get("at")
        ts = parse_instant(stamp) if stamp else None
        if ts is None:
            read_days.append(float(len(read_days)) * 0.0208)  # ~30 min steps
            continue
        if t0 is None:
            t0 = ts
        read_days.append((ts - t0) / 86400.0)

    players_last = 0
    if history:
        players_last = history[-1].get("players_tracked") or 0

    rows = []
    for idx, trophy in enumerate(trophies):
        series = []
        for j, read in enumerate(history):
            rates = read.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                series.append((read_days[j], float(rates[idx])))
        rows.append({
            "id": trophy["id"],
            "q": forecast_trophy(series, horizon_days, players_last),
        })
    return {"trophies": rows}


def resolve_dirs(argv):
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args, _unknown = parser.parse_known_args(argv)
    in_dir = (
        args.input_dir
        or os.environ.get("MAKEFUN_INPUT_DIR")
        or "/input"
    )
    out_dir = (
        args.output_dir
        or os.environ.get("MAKEFUN_OUTPUT_DIR")
        or "/output"
    )
    return Path(in_dir), Path(out_dir)


def main(argv=None):
    in_dir, out_dir = resolve_dirs(sys.argv[1:] if argv is None else argv)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        payload = json.dumps(result, separators=(",", ":")) + "\n"
        (out_dir / (path.stem + ".out")).write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
