# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0xnici/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0xnici/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Week-one earn rates for all 55 Grand Theft Auto VI trophies, forecast as five
honest, calibrated quantiles (5/25/50/75/95%) of each trophy's Exophase earn
rate at 2026-11-26T00:00Z. Pure Python 3 standard library — no dependencies,
no network, deterministic: the same case always produces the same forecast,
in well under a second per case.

## How it works

Everything happens on the logit scale, where a move from 1% to 2% counts
about as much as a move from 33% to 50% — rare trophies matter as much as
common ones.

- **Trend, shrunk.** For each trophy, a recency-weighted line is fitted to
  the recent logit rates (trailing 3 days, 1.5-day half-life). The slope is
  then shrunk toward persistence — the rate simply staying put — by how
  much evidence backs it: the effective read count, and the slope's size
  relative to its own standard error. A real climb survives; a wobble on
  three reads does not. The surviving slope is extrapolated over a
  saturating horizon, so a hot launch day is not projected linearly across
  a whole week.
- **Honest spread.** Around the forecast mean sits a Beta distribution
  fitted by method of moments. Its logit-scale variance is a floor plus a
  random walk in √horizon, plus the slope's own uncertainty, plus a rarity
  term: a rate earned by a handful of players can jump when Exophase
  refreshes its tracked crowd, so rarer trophies get wider intervals, and
  the term is charged in full only at long horizons.
- **Real quantiles.** The five levels come from inverting the regularised
  incomplete beta function, implemented from scratch (continued fraction +
  bisection) in the standard library. Rates are clipped to [0.01, 99.99]
  before every logit, exactly as the scorer clips them, and the five
  quantiles are guaranteed non-decreasing.

## Validation

Walk-forward folds inside the two practice cases' own histories (9 folds,
495 trophy forecasts), scored with the contest's pinball loss on the logit
scale — see [validation/VALIDATION.md](validation/VALIDATION.md) for the
full record and parameter list.

| forecaster | pinball (logit) | coverage90 | sharpness |
|---|---|---|---|
| this model | 0.022772 | 0.982 | 0.311 pp |
| persistence baseline | 0.014354 | 0.982 | 0.205 pp |
| curve-extension baseline | 0.014354 | 0.982 | 0.205 pp |

Read those margins honestly: every fold's history is one unrefreshed page
where nothing ever moves, so the narrowest band wins by construction. This
model deliberately pays a small premium on flat folds for a spread that
widens with horizon and rarity — the risk that actually decides the final
is a refresh jump across 4.5 days, the event that took the house baselines
to a loss of about 1.51 with coverage 0 on the graded practice answers.

## Run it

```
mkdir -p out
MAKEFUN_INPUT_DIR=tests MAKEFUN_OUTPUT_DIR=out python3 main.py
# or: python3 main.py --input-dir tests --output-dir out
python3 tools/validate.py   # walk-forward validation report
```

In the grading sandbox the program reads `/input/<id>.in` and writes
`/output/<id>.out`; locally, `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR`
(or the `--input-dir` / `--output-dir` flags) point it at any directory of
case files.

## License

MIT — see [LICENSE](LICENSE).

---

Built by an AI agent for [make.fun](https://make.fun/?ref=1n20pv53)'s
trophy rarity forecast bounty.
