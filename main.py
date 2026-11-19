"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01) -- Brutal ticket.

Model: holders decomposition with refresh-aware dilution and group shrinkage.

Why not persistence: Exophase does not refresh its page continuously. It
shows a stale snapshot for hours, then jumps: on launch night the page sat
at 638 players tracked for ~4.5 hours, then refreshed to 3,868 -- and every
trophy rate except the one at 100% fell hard (the house baselines score 1.51
on the practice for exactly this reason: their narrow bands sit on the stale
rates). A rate is holders / players-tracked, so a forecast must model the
two counts separately:

    rate_i(target) = H_i(target) / N(target)

    H_i(target) = H_i(now) + g * dt + q_i * (N(target) - N(now))

  * N(target): players tracked at the target. Fitted as log N against
    log(1 + hours since release) over the history (growth that slows as the
    launch surge passes), clamped to [N(now), 25 * N(now)].
  * q_i: the share of *newly* tracked players who hold trophy i. Estimated
    from the history's own refresh jumps (regressing each jump's holder
    gain on its player gain, pooled per rate band), because the new cohort
    is far less progressed than the day-one cohort: on launch night the new
    players averaged ~1.03 trophies each against ~1.48 for the first cohort,
    and the near-universal trophy stayed at 100% while the rest fell to
    roughly a fifth of their old rates.
  * g: holders gained per hour by the *existing* cohort as they keep
    playing, estimated jointly with q in the same per-band regression
    (holder gain = g * dt + q * dN per jump). With no jumps in the history
    there is nothing to regress on, so g falls back to 0 and q to a prior:
    new players hold trophy i at 7% of the old cohort's rate (the launch
    night totals imply ~0.07), except near-universal trophies (rate >= 50%),
    which new players hold at the old rate.

Pooling: trophies are grouped by rate band (<0.5, 0.5-1, 1-2, 2-5, >=5%).
The (g, q) pair is fitted per band by ridge-regularised least squares over
that band's refresh jumps, and each trophy's own q (from its first-to-last
holder / player totals) is shrunk toward its band's q, weighted by how many
new players the history actually observed. Rare trophies have tiny holder
counts, so their own estimates are noisy and the band dominates; common
trophies carry enough exposure to speak for themselves.

Flat-history prior on N: when the history shows no refresh at all, the
page is stale and a jump is pending. The pending jump scales with how long
the page has been stale (history span + horizon): N(target) is taken as
N(now) * (1 + span_hours + horizon_hours), capped at 8x. This only fires
when no growth was observed; any observed growth switches the fit to the
data-driven log-log extrapolation.

Spread: quantiles are normal on the logit scale around the median above,
with sigma = sqrt(0.25 * horizon_days + u^2 + rare^2), where u = 0.90 when
no refresh has ever been observed (the next jump's size is unknown) and
0.45 once refreshes have been seen, and rare = 0.5 / sqrt(holders + 1)
widens the band for trophies held by only a handful of players. Sigma is
clamped to [0.25, 1.60]. Wide on purpose: the pinball loss punishes a
narrow band that the refresh jumps clean over far more than it punishes a
wide one.

Standard library only, no network, fully deterministic (no randomness).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Rate bands (percent) used for pooling (g, q) across trophies.
BAND_EDGES = [0.0, 0.5, 1.0, 2.0, 5.0, 100.0]
# Prior new-cohort rate, as a fraction of the old cohort's rate, when no
# refresh jump has been observed yet (launch-night totals imply ~0.07).
KAPPA_PRIOR = 0.07
# Above this rate a trophy is treated as near-universal: newcomers hold it
# at the old cohort's rate instead of the diluted prior.
UNIVERSAL_RATE = 50.0
# Prior strength (in newly tracked players) when shrinking a trophy's own
# q toward its band's q.
SHRINK_PLAYERS = 2000.0


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def band_of(rate: float) -> int:
    for b in range(len(BAND_EDGES) - 1):
        if BAND_EDGES[b] <= rate < BAND_EDGES[b + 1]:
            return b
    return len(BAND_EDGES) - 2


def ols_2var(xs1, xs2, ys):
    """Ridge-regularised least squares for y ~ a*x1 + b*x2 (no intercept).

    Returns (a, b). Falls back to (0, 0) when the design is degenerate.
    """
    s11 = sum(x * x for x in xs1)
    s22 = sum(x * x for x in xs2)
    s12 = sum(a * b for a, b in zip(xs1, xs2))
    t1 = sum(a * y for a, y in zip(xs1, ys))
    t2 = sum(b * y for b, y in zip(xs2, ys))
    ridge = 1e-9 * (s11 + s22 + 1.0)
    s11 += ridge
    s22 += ridge
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return 0.0, 0.0
    a = (t1 * s22 - t2 * s12) / det
    b = (s11 * t2 - s12 * t1) / det
    return a, b


def fit_loglog(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b) or None when degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    n_tr = len(trophies)
    horizon_min = float(case["horizon_minutes"])
    horizon_h = horizon_min / 60.0
    horizon_d = horizon_min / 1440.0

    release = parse_time(case["game"]["release_date"])
    if release.tzinfo is None:
        release = release.replace(tzinfo=timezone.utc)

    # --- Per-read series: times (hours since release), players, holders. ---
    times, players = [], []
    holders = [[None] * len(history) for _ in range(n_tr)]
    for k, read in enumerate(history):
        times.append((parse_time(read["at"]) - release).total_seconds() / 3600.0)
        n_k = float(read["players_tracked"])
        players.append(n_k)
        for i in range(n_tr):
            r = read["rates"][i]
            if r is not None:
                holders[i][k] = float(r) * n_k / 100.0
    # Fill missing holder counts forward from the last known read.
    last_rate = []
    for i in range(n_tr):
        known = [h for h in holders[i] if h is not None]
        last_rate.append(known[-1] / players[-1] * 100.0 if known else 0.0)
        last_known = known[-1] if known else 0.0
        for k in range(len(history)):
            if holders[i][k] is None:
                holders[i][k] = last_known
            else:
                last_known = holders[i][k]
    # A trophy never observed at any rate (all its rates null) falls back
    # to the median last rate of the rarest band: unobserved is unknown,
    # not absent. A trophy observed at 0% keeps its 0 (clamped to 0.01
    # downstream) -- that is real evidence of rarity.
    observed = [r for r in last_rate if r > 0]
    global_median = sorted(observed)[len(observed) // 2] if observed else 1.0
    for i in range(n_tr):
        ever_observed = any(read["rates"][i] is not None for read in history)
        if not ever_observed:
            band_rates = sorted(r for r in observed if band_of(r) == 0)
            last_rate[i] = band_rates[len(band_rates) // 2] if band_rates else global_median
            holders[i] = [last_rate[i] * n / 100.0 for n in players]

    n0 = players[-1]
    t_last = times[-1]
    t_target = t_last + horizon_h
    span_h = max(0.0, times[-1] - times[0])

    # --- Refresh jumps: consecutive reads where players tracked grew. ---
    jumps = []  # (dt_hours, dN, [dH per trophy])
    for k in range(1, len(history)):
        dn = players[k] - players[k - 1]
        if dn > 0:
            dt = max(0.0, times[k] - times[k - 1])
            jumps.append((dt, dn, [holders[i][k] - holders[i][k - 1] for i in range(n_tr)]))
    saw_refresh = len(jumps) > 0
    total_dn = players[-1] - players[0] if saw_refresh else 0.0

    # --- Per-band (g, q) fit over the jumps, pooled across the band. ---
    band_gq = {}
    for b in range(len(BAND_EDGES) - 1):
        members = [i for i in range(n_tr) if band_of(last_rate[i]) == b]
        if not members or not jumps:
            band_gq[b] = (0.0, None)
            continue
        xs1, xs2, ys = [], [], []
        for dt, dn, dh in jumps:
            xs1.append(len(members) * dt)
            xs2.append(len(members) * dn)
            ys.append(sum(dh[i] for i in members))
        g, q = ols_2var(xs1, xs2, ys)
        g = max(0.0, g)
        q = min(1.0, max(0.0, q)) if q is not None else None
        band_gq[b] = (g, q)

    # --- N at the target. ---
    if saw_refresh:
        xs = [math.log1p(max(0.0, t)) for t in times]
        ys = [math.log(max(1.0, n)) for n in players]
        fit = fit_loglog(xs, ys)
        if fit is not None and fit[1] > 0.02:
            n_target = math.exp(fit[0] + fit[1] * math.log1p(max(0.0, t_target)))
        else:
            # Growth seen but no usable log-log slope: extend the observed
            # mean hourly growth rate, damped.
            growth = (players[-1] - players[0]) / max(1e-9, times[-1] - times[0])
            n_target = n0 + growth * horizon_h * 0.5
        n_target = min(max(n_target, n0), 25.0 * n0)
    else:
        # No refresh observed: the page is stale and a jump is pending,
        # sized by how long it has been stale (span + horizon).
        mult = 1.0 + (span_h + horizon_h)
        n_target = n0 * min(mult, 8.0)
    dn_target = max(0.0, n_target - n0)

    # --- Per-trophy median at the target. ---
    rows = []
    for i, trophy in enumerate(trophies):
        r = last_rate[i]
        b = band_of(r)
        g_band, q_band = band_gq[b]
        if saw_refresh and total_dn > 0:
            # Own q from first-to-last totals, net of band progression.
            span_t = max(1e-9, times[-1] - times[0])
            own_q = (holders[i][-1] - holders[i][0] - g_band * span_t) / total_dn
            own_q = min(1.0, max(0.0, own_q))
            if q_band is None:
                q_i = own_q
            else:
                w = total_dn / (total_dn + SHRINK_PLAYERS)
                q_i = w * own_q + (1.0 - w) * q_band
            g_i = g_band
        else:
            # No refresh observed: prior new-cohort rates.
            if r >= UNIVERSAL_RATE:
                q_i = r / 100.0
            else:
                q_i = KAPPA_PRIOR * r / 100.0
            g_i = 0.0
        h_target = holders[i][-1] + g_i * horizon_h + q_i * dn_target
        median = clamp(100.0 * h_target / n_target) if n_target > 0 else clamp(r)

        # --- Spread on the logit scale. ---
        u = 0.45 if saw_refresh else 0.90
        rare = 0.5 / math.sqrt(holders[i][-1] + 1.0)
        sigma = math.sqrt(0.25 * horizon_d + u * u + rare * rare)
        sigma = min(1.60, max(0.25, sigma))
        lg = logit(median)
        qs = [round4(clamp(inv_logit(lg + z * sigma))) for z in Z]
        # Keep the row non-decreasing after rounding/clamping.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_rows(case: dict) -> dict:
    """Persistence forecast, used only if the main path fails on a case."""
    try:
        last = case["history"][-1]
        trophies = case["trophies"]
        spread = 0.5 * math.sqrt(float(case["horizon_minutes"]) / 1440.0)
    except (KeyError, TypeError, ValueError, IndexError):
        return {"trophies": []}
    rows = []
    for i, trophy in enumerate(trophies):
        rate = last["rates"][i]
        lg = logit(rate if rate is not None else 0.0)
        qs = [round4(clamp(inv_logit(lg + z * spread))) for z in Z]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_rows(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
