# Validation

## How this was validated

`tools/validate.py` runs walk-forward folds inside each practice case's own
recorded history. For every held-out read from the fourth onward, it rebuilds
a case from the reads before it, forecasts the held-out read at the recording
cadence (30 minutes), and scores the forecast against the rates that read
actually printed. Each case is validated only on its own folds — the
forecaster is stateless per case, so no fold trains on or is scored against
the other case (leave-one-case-out by construction). Two practice cases are
a smoke test, not tuning data: no parameter below was fitted to them.

Scoring follows the contest exactly: pinball loss on the logit scale, rates
clamped to [0.01, 99.99] percent before the logit, per item
`2 * max(tau * d, (tau - 1) * d)`. Coverage is the share of held-out rates
inside [q05, q95] (nominal 0.90); sharpness is the mean q95 − q05 width in
percentage points; typical miss is the mean |y − q50| in percentage points.

## Results (pooled: 9 folds, 495 trophy forecasts)

| forecaster | pinball (logit) | coverage90 | sharpness | typical miss |
|---|---|---|---|---|
| this model | 0.022772 | 0.982 | 0.311 pp | 0.004 pp |
| baseline 1 — persistence | 0.014354 | 0.982 | 0.205 pp | 0.000 pp |
| baseline 2 — curve extension | 0.014354 | 0.982 | 0.205 pp | 0.000 pp |

Per case the picture is identical (p00: 4 folds / 220 trophies, p01: 5 folds
/ 275 trophies; same losses to 6 decimals, because both histories are the
same unrefreshed page).

## How to read these numbers

Every read inside both practice histories is the same page: 638 players
tracked, no rate ever moves. On folds where the truth is exactly "nothing
changed", the narrowest band around the last rate wins by construction, and
the baselines' 0.205 pp band beats this model's 0.311 pp band. The model's
extra width is not noise — it is the price of the risk the folds never
realise: Exophase refreshes a game's counts in jumps (the graded practice
answers are the refreshed page, on which the house baselines score about
1.51 with coverage 0, per the kit), and the final case forecasts 4.5 days
ahead, not 30 minutes. The rarity part of the spread is therefore gated by
a saturating horizon factor, so the insurance is cheap at 30 minutes and
fully charged at a week. Whether that trade pays on the graded practice
answers cannot be measured from the kit — the answering reads are not in
it — and this file does not pretend otherwise.

Coverage is 0.982 for all three forecasters against a 0.90 nominal: on flat
folds almost everything lands inside any honest band. The single uncovered
trophy in every fold is the one sitting at exactly 100.00%, which no
quantile below 100 can cover.

## Model parameters (as shipped)

- Trend: recency-weighted least squares on logit rates over the trailing
  3 days, weight half-life 1.5 days; needs at least 3 usable reads.
- Shrinkage toward persistence: slope multiplied by
  `n_eff / (n_eff + 6)` and by `b² / (b² + se² + 0.01)`, capped at
  ±2.5 logit/day, then extrapolated over a saturating horizon
  `h / (1 + h / 10 days)`.
- Spread on the logit scale: variance = 0.07² (floor) + 0.42² · horizon in
  days (random walk) + (slope SE · bent horizon)² capped at 1, plus a
  rarity term `0.5 · (1 − m) / (players · m + 5)` gated by
  `h / (h + 0.25 days)`.
- Distribution: Beta by method of moments around the forecast mean
  (delta-method variance from the logit spread); quantiles by inverting
  the regularised incomplete beta (continued fraction + bisection),
  standard library only.
- Rates are clipped to [0.01, 99.99] percent before every logit, matching
  the scorer; the five quantiles are forced non-decreasing after rounding
  to 4 decimals.

## Reproduce

```
python3 tools/validate.py
MAKEFUN_INPUT_DIR=tests MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Two runs of `main.py` on the practice cases produce byte-identical outputs
(verified by diff), at roughly 0.4 s per case on one core — well inside the
8 s cap.
