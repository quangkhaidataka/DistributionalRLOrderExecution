# Order-size robustness — does IQN keep its tail advantage as the order grows?

**Short answer: no.** The tail advantage over TWAP decays monotonically with order
size and is gone by ~0.8% of ADV. But the study produces a different, defensible
robustness claim in its place: **IQN degrades gracefully — it converges onto the
TWAP benchmark instead of diverging from it, while DQN and DDQN become materially
worse than TWAP.**

---

## Setup

| item | value |
|---|---|
| instrument | AAPL 2014, 3-min bars, walk-forward (train Jan–Jul / val Aug–Sep / **test Oct–Dec**) |
| temporary impact | η = 1×10⁻⁵, **fixed at every level** — never rescaled with q₀ |
| order sizes | 5,000 / 50,000 / 250,000 / 1,000,000 shares |
| ADV (2014, 252 days) | 32,178,605 shares/day |
| agents | TWAP, DQN, DDQN, IQN-neutral |
| seeds | **5** — {42, 123, 7, 2024, 31} — at 50k / 250k / 1M; **3** — {42, 123, 7} — at 5k |
| training | **12,000 episodes** per agent (the main study uses 50,000) |
| selection | CRN validation, 12 checkpoints, min val-CVaR₀.₉₅ — the main study's rule |
| test | 10,000 held-out episodes, env + global RNG reseeded per agent |

Because η is held fixed, impact cost per share is η·x and the *total* cost of the
parent order is η·Σx², so cost scales with q₀. TWAP's own mean IS rises from
0.35 bps to 46.22 bps across the sweep — impact goes from negligible to dominant.

**Sanity check against the main study.** At q₀ = 5,000 this reproduces the published
AAPL table closely despite the reduced training budget: CVaR₀.₉₅ here is IQN 28.48 /
DDQN 28.22 / DQN 33.93, against the main study's 26.23 / 26.81 / 34.23 (50,000
episodes, 5 seeds). Same ordering, same "DQN is the unstable outlier" structure.

---

## Headline: CVaR₀.₉₅ cut vs TWAP

| q₀ | % ADV | seeds | TWAP CVaR₀.₉₅ | DQN | DDQN | IQN-neutral |
|---|---|---|---|---|---|---|
| 5,000 | 0.016% | 3 | 57.95 | −41.4% | **−51.3%** | −50.9% |
| 50,000 | 0.155% | 5 | 59.00 | −21.3% | **−38.4%** | −31.4% |
| 250,000 | 0.777% | 5 | 68.31 | **−8.1%** | −7.5% | −7.8% |
| 1,000,000 | 3.108% | 5 | 103.35 | +5.4% | +9.0% | **−0.1%** |

Bold = best at that level. Full per-agent Mean IS / Std IS / CVaR₀.₉₀ / CVaR₀.₉₅
with seed dispersion are in `results/table_order_robust.tex` and
`results/q0_*/level_summary.txt`.

---

## What the sweep shows

**1. The advantage is real but size-limited.** At institutional child-order size
(0.016% ADV) every learned agent roughly halves the tail: DDQN −51.3%, IQN −50.9%,
DQN −41.4%. By 0.78% ADV all three have collapsed to within a point of each other
at −7.5% to −8.1%, and by 3.1% ADV only IQN still matches TWAP.

**2. IQN does not lead on the tail below the top level.** DDQN has the lowest
CVaR₀.₉₅ at 5,000 and 50,000 shares; at 250,000 the three agents are separated by
0.6 bps against across-seed sds of 1.3–2.7 bps, i.e. indistinguishable. This is
consistent with the main study's own finding that "IQN vs DDQN is a wash on AAPL" —
the order-size sweep does not rescue it.

**3. The interesting result is at the top level.** At 1,000,000 shares (5 seeds):

| agent | Mean IS | CVaR₀.₉₅ | across-seed sd (CVaR₀.₉₅) |
|---|---|---|---|
| TWAP | 46.22 | 103.35 | 1.49 |
| **IQN-neutral** | **46.80** | **103.30** | **1.74** |
| DQN | 55.65 | 108.95 | 3.01 |
| DDQN | 61.75 | 112.63 | 8.54 |

IQN recovers the TWAP benchmark to within 0.6 bps of mean IS and 0.05 bps of
CVaR₀.₉₅ — a −0.1% cut, i.e. statistically indistinguishable from TWAP — with
comparable seed stability (sd 1.74 vs TWAP's 1.49). DQN and DDQN do not: DDQN is
9% *worse* than TWAP on the tail with an across-seed sd of 8.54 bps, i.e. it fails
unpredictably, and its mean IS is 34% above TWAP's.

The convergence is tight at the seed level, not just on average. At seed 31 the
selected IQN checkpoint matches TWAP to the fourth decimal (mean IS 46.22135 vs
46.21990 bps; CVaR₀.₉₅ 101.109381 vs 101.109381), while DQN and DDQN at that same
seed come in at 112.68 and 113.48 bps.

**Why TWAP becomes optimal.** Under linear temporary impact the total cost of the
parent order is η·Σₜxₜ², which for a fixed Σxₜ = q₀ is minimised by equal splits.
So as impact comes to dominate price risk, the impact-optimal schedule *is* TWAP.
The front-loading that all three agents learn at small size (see
`strategy_analysis/`) is exactly the wrong reflex here, because concentrating the
order inflates Σx². The robustness question is therefore whether an agent can
*unlearn* front-loading when impact dominates. **IQN can; DQN and DDQN cannot.**

---

## How to state this in the thesis

Do **not** claim the tail advantage persists — the numbers do not support it. The
supportable claims are:

> The tail advantage of distributional RL over TWAP is largest for small child
> orders (−51% at 0.016% ADV) and decays monotonically with order size, vanishing
> once the parent order approaches 1% of ADV. At 3.1% of ADV, where temporary
> impact dominates price risk and the impact-optimal schedule is uniform, IQN
> recovers the TWAP benchmark on CVaR₀.₉₅ (103.30 vs 103.35 bps over five seeds),
> whereas DQN and DDQN degrade past it by 5.4% and 9.0% respectively, the latter
> with an across-seed standard deviation of 8.5 bps. Distributional value learning
> therefore does not extend the tail advantage to large orders, but it does confer
> graceful degradation: the learned policy collapses onto the correct benchmark
> rather than diverging from it.

---

## Caveats

1. **Reduced training budget.** 12,000 episodes vs the main study's 50,000. The
   q₀ = 5,000 block reproduces the published numbers to within ~2 bps, so the
   comparison structure is intact, but absolute levels are slightly worse and the
   larger levels may be further from convergence than the small ones.
2. **Seed count is uneven.** 5 seeds at 50k / 250k / 1M, 3 seeds at 5k. Seed
   dispersion remains large for DQN and DDQN at several levels (DDQN ±8.54 bps
   at the top level, ±7.72 for DQN at 50k), so per-level rankings *between the
   learned agents* are not always significant. The *trend* across levels, and the
   IQN-vs-TWAP gap at the top level, are the robust parts.
3. **Impact model is linear and unchanged.** η·x with η fixed; no square-root or
   depth-aware term. The `impact_estimation/` study found real AAPL impact is
   concave (|Δp| ∝ |flow|^0.75), so at the largest sizes a linear model likely
   *overstates* cost — the 1M-share level should be read as a stress test rather
   than a calibrated scenario.
4. **q₀ = 1,000,000 is 3.1% of ADV** executed inside a 60-minute window; that is an
   aggressive parent order, well outside the regime the impact coefficient was
   calibrated for.

---

## Reproducing / files

```
run_order_robust.py                     driver (self-contained; resumable)
results/table_order_robust.tex          the LaTeX table, grouped by q0 level
results/q0_<Q>/level_summary.{json,txt} per-level aggregates
results/q0_<Q>/seed<S>/metrics.json     per-cell metrics + selected checkpoint
results/adv.json                        ADV and % ADV per level
checkpoints/q0_<Q>/seed<S>/<AGENT>_ep<E>.pt   every checkpoint, 12 per agent
```

```bash
python3 run_order_robust.py --level 250000    # all seeds of one level (resumes)
python3 run_order_robust.py --summarize       # rebuild tables from metrics.json
```

Completed cells are skipped on re-run, so an interrupted sweep costs nothing to
restart. **Caveat on resumability:** the skip guard is per *agent* (final
`<AGENT>_ep12000.pt`) and per *cell* (`metrics.json`). `run_cell` calls
`train_segment(..., 0, EPISODES, EPISODES)`, and full resume state
(weights + optimizer + replay buffer + torch/numpy/env RNG) is only persisted when
`ep_end < total_episodes` — so it is never written here. An agent interrupted
mid-training therefore restarts from episode 0 rather than from its last
checkpoint. That is deterministic and protocol-identical (`ep_start == 0` reseeds
the env and global RNG from the seed), just not free.

**Isolation:** every module under `order_robust_check/` is a copy, the AAPL
bar file is a copy in `order_robust_check/data/processed/`, and `_guard_paths()`
refuses any write outside this folder. Nothing in the main repo was read-then-written
or modified.
