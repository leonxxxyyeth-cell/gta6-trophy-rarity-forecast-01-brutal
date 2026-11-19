# GTA VI Trophy Rarity Forecaster — Brutal ticket

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the percent
of Exophase-tracked players who will have earned it at the case's target, as
five quantiles per trophy. Entry for
[`gta6-trophy-rarity-forecast-01`](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
on make.fun.

## Model — holders decomposition with refresh-aware dilution

Persistence fails here for a structural reason: Exophase does not refresh
its page continuously. It shows a stale snapshot for hours, then jumps — on
launch night the page sat at 638 players tracked for ~4.5 hours, then
refreshed to 3,868, and every trophy rate except the one at 100% fell hard.
A rate is `holders / players-tracked`, so the forecast models the two
counts separately:

```
rate_i(target) = H_i(target) / N(target)
H_i(target)    = H_i(now) + g·Δt + q_i·(N(target) − N(now))
```

- **N(target)** — players tracked at the target, fitted as log N against
  log(1 + hours since release) over the history, clamped to [N(now), 25×N(now)].
- **q_i** — the share of *newly* tracked players who hold trophy i,
  estimated from the history's own refresh jumps (per rate band, ridge
  least squares of holder gain on player gain), with each trophy's own
  estimate shrunk toward its band's. The new cohort is far less progressed
  than the day-one cohort: on launch night new players averaged ~1.03
  trophies each vs ~1.48 for the first cohort.
- **g** — holders gained per hour by the *existing* cohort as they keep
  playing, fitted jointly with q in the same per-band regression.
- **Flat-history prior** — when the history shows no refresh at all, the
  page is stale and a jump is pending, sized by how long it has been stale
  (history span + horizon, N multiplier capped at 8×); new players then
  hold each trophy at 7% of the old cohort's rate (the launch-night totals
  imply ~0.07), except near-universal trophies (≥50%), which newcomers hold
  at the old rate.
- **Spread** — normal on the logit scale, σ = √(0.25·horizon_days + u² +
  rare²), u = 0.90 with no refresh observed and 0.45 once refreshes have
  been seen, plus a rare-trophy term for tiny holder counts; σ clamped to
  [0.25, 1.60]. Wide on purpose: the pinball loss punishes a narrow band
  that a refresh jumps clean over far more than a wide one.

Pooling is by rate band (<0.5, 0.5–1, 1–2, 2–5, ≥5%). Edge cases handled:
rates of 0, null rates (forward-filled; a trophy never observed at all
falls back to its band's median rate), single-read histories, long horizons.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, fully deterministic.
Reads every `<id>.in` in the input directory and writes `<id>.out` to the
output directory. ~0.1 s per practice case (cap is 8 s).

## Local validation

The kit ships no practice answers, so answers were reconstructed from the
kit's own published refresh totals (3,868 players, 4,276 trophies earned at
the 04:30Z refresh): non-100% trophies fall to 0.221× their last rate on
average, the 100% trophy stays at 100%. Scored with the contest's pinball
loss on the logit scale:

| case | this entry | house persistence baseline |
|------|-----------|----------------------------|
| p00  | **0.210** (typical miss 0.03 pp, cov90 0.98) | 1.400 (miss 0.68 pp, cov90 0.00) |
| p01  | **0.197** (typical miss 0.01 pp, cov90 0.98) | 1.400 (miss 0.68 pp, cov90 0.00) |

The kit states the house baseline's true practice loss is ~1.51; the
reconstruction above gives 1.40 for it because it assumes a uniform 0.221×
shrink while the real per-trophy answers vary. Sensitivity: under 0.15× and
0.30× refresh hypotheses this entry scores 0.27–0.29 / 0.22–0.23 against the
baseline's 1.78 / 1.10 — it beats persistence across the plausible range,
losing only in the counterfactual where no refresh happens at all (the kit
states the refresh did happen). Determinism verified (byte-identical repeat
runs); synthetic refresh-jump, single-read, zero-rate and null-rate cases
all produce valid output.

## License

MIT — see `LICENSE`.
