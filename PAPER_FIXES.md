# PAPER_FIXES.md — `main_paper.tex` consistency audit

> **`main_paper.tex` is NOT edited by this document.** This is a change-list with precise locations (section / table / tex line) and verdicts.
> **Ground truth:** the `.tex` for what the paper *claims*; code dataclass defaults + script constants and `results/*/logs/` for what is *true*.
> **Method:** every quantitative claim was extracted from the tex, cross-checked against (a) code and (b) result logs, and its abstract-level arithmetic re-derived. All three results tables (`tab:ac_results`, `tab:jd_results`, `tab:taq_results`) were confirmed to reproduce the current `results/*/logs/` **exactly** — i.e. the paper's numbers are the live results, so retraining will supersede them per the classification in §4.
> Line numbers are from the current `main_paper.tex` (928 lines) and should be re-confirmed at edit time (a diff can shift them).

Verdict legend: **MATCH** · **FIX-IN-PAPER** (paper wrong, results stay) · **WILL-CHANGE** (superseded by a Phase-3 retrain — insert a placeholder, not a new number, now) · **FLAG** (verify before acting).

---

## §1 — Parameter tables

### 1.1 `tab:sim_params` "Simulation parameters" (caption L536, label L537, body L534–575)

| Parameter (symbol) | Paper says | Code / logs say | tex line | Verdict |
|---|---|---|---|---|
| Decision periods $N$ | 5 | 5 | L543 | MATCH |
| Horizon $T$ | 60 min | 60.0 | L544 | MATCH |
| Initial inventory $q_0$ | 100,000 | 100000 | L545 | MATCH |
| Arrival price $p_0$ | \$100 | 100.0 | L546 | MATCH |
| Temporary impact $\eta$ | $2.5\times10^{-6}$ | 2.5e-6 | L547 | MATCH |
| Permanent impact $\gamma$ | $2.5\times10^{-7}$ | 2.5e-7 | L548 | MATCH |
| Volatility $\sigma$ | 0.00095 | 0.00095 | L549 | MATCH |
| Penalty $a$ | 0.001 | 0.001 (`sim_config.json`) | L550 | MATCH |
| Discount $\lambda$ | 0.99 | 0.99 | L551 | MATCH |
| **Jump intensity $\lambda_J$** | **0.3 per period** | code 0.3 but multiplied by `dt=12` ⇒ effective **3.6/period** (F1); recalibrated to **0.03/period** (P1-T3) | L554 | **WILL-CHANGE** (JD retrain) |
| **Jump mean $\mu_J$** | **−0.002** | −0.002 now; recalibrated to **−0.30** | L555 | **WILL-CHANGE** |
| **Jump vol $\sigma_J$** | **0.005** | 0.005 now; recalibrated to **~0.40** | L556 | **WILL-CHANGE** |
| Training episodes | 30,000 | 30,000 (`*_training.json` len=30000) | L559 | MATCH |
| Eval episodes | 10,000 | 10,000 (`all_results.json` n=10000) | L560 | MATCH |
| Minibatch $B$ | 64 | 64 | L561 | MATCH |
| Learning rate | $10^{-4}$ | 1e-4 | L562 | MATCH |
| Replay capacity | 50,000 | 50,000 | L563 | MATCH |
| **Target update $C$** | **100 steps** | **500** (`DeepRLConfig`/`AgentConfig`; checkpoint `cfg.target_update_freq=500`) | **L564** | **FIX-IN-PAPER → 500** |
| Online τ $N_\tau$ | 8 | 8 | L567 | MATCH |
| Target τ $N'_\tau$ | 8 | 8 | L568 | MATCH |
| Policy τ $N''_\tau$ | 32 | 32 | L569 | MATCH |
| **Cosine embedding $n$** | **64** | **32** (`AgentConfig.cos_embedding_dim=32`; AC/TAQ checkpoints `i_vals`=(32,)) | **L570** | **FIX-IN-PAPER → 32** |
| Hidden dim $d$ | 64 | 64 (`AgentConfig.hidden_dim=64`; AC/TAQ ckpt) | L571 | MATCH¹ |
| Huber $\kappa$ | 1.0 | 1.0 | L572 | MATCH |

¹ **Provenance caveat (important):** the *current* JD results (`tab:jd_results`) were produced with **d=128, n=64** (JD IQN checkpoint = 43,462 params), whereas AC/TAQ used **d=64, n=32** (11,462 trainable params). So the paper table's "d=64, n=64" matches *neither* run exactly. After the Phase-3 unified retrain (D1: all experiments at d=64, n=32), "d=64" becomes universally true (no fix) and "n" must read **32**.

### 1.2 `tab:taq_params` "AAPL empirical study" (caption L785, label L786, body L765–787)

| Parameter | Paper says | Code says | tex line | Verdict |
|---|---|---|---|---|
| $N$ / $T$ / stride | 5 / 60 min / 12 min | 5 / 60 / 12 | L772–774 | MATCH |
| $q_0$ | 5,000 | 5000 (`run_tag.py:60`) | L775 | MATCH |
| $\eta$ | $10^{-5}$ | 1e-5 (`run_tag.py:62`) | L776 | MATCH |
| $a$ | 0.0001 | 0.0001 (`run_tag.py:61`) | L777 | MATCH |
| Discount | 0.99 | 0.99 | L778 | MATCH |
| Train / Val / Test eps | 50,000 / 300 / 10,000 | 50000 / 300 / 10000 (`run_tag.py:68,74,75`) | L780–782 | MATCH |

> **FLAG (contradicts the brief's D-list):** the D-list lists "TAQ episodes=50,000, q0=5,000, η=1e-5, a=1e-4" as *FIX-IN-PAPER* items. **They already match** — `tab:taq_params` and the `summary_table.txt` header (`q0=5000, N=5, T=60.0, eta=1e-05, a=0.0001`) are correct. **No paper change is needed here.** DQN/DDQN in this study *were* trained at hidden=128 (18,566 params) though; that is a results issue (§2.3), not a params-table issue.

---

## §2 — Results tables (all currently reproduce `results/*/logs/` exactly)

### 2.1 `tab:ac_results` (caption L612, body L596–614) — source `results/almgren_chriss/logs/`
Every cell **MATCHES** the log verbatim (e.g. IQN-neutral Std "0.15" = logged 0.1500; IQN-CVaR₀.₉₅ CVaR₉₀ "1.865" = 1.8650). Retrain impact under D1 (DQN/DDQN 128→64; IQN already 64/32):

| Rows | Verdict |
|---|---|
| TWAP, AC | MATCH — rule-based, unchanged by architecture; stable across the seed-42 re-run |
| IQN-neutral, IQN-CVaR₀.₉₀/₀.₉₅ | MATCH — already at unified 64/32; seed-42 re-run reproduces (re-verify) |
| **DQN, DDQN** | **WILL-CHANGE** — retrained at hidden 64 (18,566→5,190 params) |

### 2.2 `tab:jd_results` (caption L661, body L646–663) — source `results/jump_diffusion/logs/`
Every cell MATCHES the log. **All rows WILL-CHANGE** because both inputs change: (a) JD env recalibration (F1/D3 — the current numbers reflect the buggy ≈18-jumps/episode drift env; note TWAP mean IS 2.9080 bps is inflated by the ≈−3.6 bps/episode drift), and (b) architecture unification (IQN 128/64→64/32; DQN/DDQN 128→64). Also note the current "best val CVaR₉₅" selection produced oddly early checkpoints (IQN ep9000, DDQN ep3000, DQN ep18000) — expected under the drift env; will shift after the fix.

| Rows | Verdict |
|---|---|
| TWAP, AC, DQN, DDQN, IQN-neutral, IQN-CVaR₀.₉₀/₀.₉₅ | **WILL-CHANGE** (env recalibration + arch unification) |

### 2.3 `tab:taq_results` (caption L814, body L798–816) & `tab:taq_comparison` (caption L838, body L827–840) — source `results/taq/AAPL/summary.json`
Every cell MATCHES the log (IQN-CVaR₀.₉₅ CVaR₉₅ "6.13" = logged 6.130010; Max " 6.2367" = 6.236699). Under **D2 (keep TAQ IQN checkpoints; retrain only DQN/DDQN)**:

| Rows | Verdict |
|---|---|
| TWAP, AC | MATCH — rule-based, kept |
| IQN-neutral, IQN-CVaR₀.₉₀/₀.₉₅ | MATCH — kept checkpoints, unchanged |
| **DQN, DDQN** (both tables) | **WILL-CHANGE** — retrained at hidden 64 |
| `tab:taq_comparison` row 3 (IQN-neutral / IQN-CVaR) | MATCH — both kept |
| `tab:taq_comparison` rows 1–2 (DQN, DDQN / IQN-CVaR) | **WILL-CHANGE** — numerator retrained |

> **Single-seed by design (scope decision 2026-07-17):** the entire TAQ study is reported for **seed 42 only** — DQN/DDQN from the Batch-A retrain, IQN kept. There is **no TAQ mean±std / multi-seed table** and none is promised anywhere in this file; seed robustness is characterized in the **simulation** study (AC + JD multi-seed). Add the explicit limitation sentence — see §4.1 item 7. (See PLAN.md Decision log / Risk R-4.)

---

## §3 — Abstract / prose percentage claims (arithmetic re-derived)

| Claim (tex line) | Paper | Re-derivation from logs | Verdict |
|---|---|---|---|
| Abstract: CVaR₉₅ ↓ **88.8%** vs TWAP (TAQ) — L66 | 88.8% | (54.89329−6.130010)/54.89329 = **88.83%** ✓ | **MATCH & STABLE** (TWAP + IQN-CVaR both kept, D2) |
| Abstract: worst-case ↓ **97.0%** vs TWAP (TAQ) — L66 | 97.0% | (207.75745−6.236699)/207.75745 = **97.00%** ✓ | **MATCH & STABLE** |
| Abstract/contrib: max shortfall ↓ **5.2×** vs DQN, **7.3×** vs DDQN (TAQ) — L67, L103 | 5.2× / 7.3× | 32.2486/6.2367=5.17×; 45.2367/6.2367=7.25× ✓ | **WILL-CHANGE** (DQN/DDQN retrained) |
| AC prose: IQN-CVaR ↓ **13.1%** tail, **21.7%** worst-case vs TWAP — L621–624 | 13.1% / 21.7% | (2.1956−1.9082)/2.1956=13.09%; (2.6910−2.1071)/2.6910=21.70% ✓ | MATCH (depends only on TWAP+IQN-CVaR; re-verify after AC re-run) |
| JD prose: IQN-neutral ↓ **10.7%** CVaR, **15.1%** max vs DQN — L669 | 10.7% / 15.1% | (2.7977−2.4957)/2.7977=**10.79%**; (3.5744−3.0259)/3.5744=**15.35%** | **FIX + WILL-CHANGE** — see §5.1 (paper's own numbers round to 10.8%/15.3%, not 10.7%/15.1%); moot after JD retrain but the arithmetic slip should be logged |
| JD prose: DDQN max **49.5%** higher than IQN-neutral — L669 | 49.5% | (4.5275−3.0259)/3.0259=**49.62%** | FIX (→49.6%) + WILL-CHANGE |
| JD prose: all-RL CVaR₉₅ ↓ ≥ **47.1%** vs TWAP — L667 | 47.1% | (5.6719−2.9953)/5.6719=47.19% ✓ | WILL-CHANGE |
| TAQ prose: all-RL CVaR₉₅ < 7 bps, ↓ ≥ **87.3%** vs TWAP — L823 | 87.3% | (54.8933−6.9884)/54.8933=87.27% ✓ | WILL-CHANGE (worst-RL is DDQN, retrained) |
| TAQ prose: max shortfall 41.39→6.24, **84.9%** ↓ (IQN-neutral→CVaR) — L844 | 84.9% | (41.3920−6.2367)/41.3920=84.93% ✓ | MATCH & STABLE (both IQN, kept) |
| TAQ prose: vol ↓ **3.93×** vs IQN-neutral, **5.36×** vs DDQN — L842 | 3.93× / 5.36× | 1.0666/0.2715=3.93×; 1.4546/0.2715=5.36× ✓ | 3.93× MATCH (kept); 5.36× WILL-CHANGE (DDQN) |
| TAQ prose: pays **4.02** bps more, eliminates **48.76** bps tail vs TWAP — L846 | 4.02 / 48.76 | 5.6086−1.5917=4.02; 54.8933−6.13=48.76 ✓ | MATCH & STABLE |
| Commented abstract (L76–80): 55.8% / 64.5% (JD), 10.8% / 15.3% (JD), 81× vol (TAQ) | — | JD figures match logs; not compiled | **WILL-CHANGE** + see §5.2 (dead duplicate abstract) |

---

## §4 — Change classification & instructions

### 4.1 FIX-IN-PAPER now (results unchanged)
1. **`tab:sim_params` L564:** target update $C$ **100 → 500**.
2. **`tab:sim_params` L570:** cosine embedding $n$ **64 → 32**.
3. **Jump equation (Eq. `jd_price`, L514):** replace $N_t\sim\text{Poisson}(\lambda\,\Delta t)$ with the **per-period** form $N_t\sim\text{Poisson}(\lambda_J)$ so the equation matches the "per period" table label and the fixed code (P1-T3). (Do this even though the jump *values* also change — the functional form is a framing fix independent of the retrain.)
4. **"Fair comparison" paragraph, L577–584:** rewrite to state explicit trainable-parameter counts — **IQN = 11,462, DQN/DDQN = 5,190** (identical 2-layer LayerNorm+ReLU backbone; IQN's only addition is the cosine quantile embedding, the minimal machinery of the method, cite **Dabney et al. 2018**). NB: this claim only becomes *true* after the Phase-3 unified retrain — do not paste the counts until DQN/DDQN (and JD IQN) are re-run at width 64.
5. **Jump-diffusion framing, L506–522 & L183-198-analog:** reframe as an explicit **stress-test scenario** with rare, economically-meaningful jumps (P(≥1 jump/episode) ≈ 10–20%); cite **Moazeni, Coleman & Li (2013)** alongside Merton (1976).
6. **Notation collisions (§5.3).**
7. **Empirical-study limitation sentence (single-seed case study).** Near the start of the AAPL empirical study (around `tab:taq_results` / `tab:taq_comparison`, **L798–840** — e.g. in the study's setup paragraph or the table notes), insert a sentence such as: *"All empirical results are reported for a single training seed (seed 42); seed-level robustness of the learned policies is characterized in the simulation study (Table~\ref{tab:jd_results} and the multi-seed AC/JD tables). The evaluation window (Oct–Dec 2014) is fixed market data and does not vary with the training seed."* Makes the single-seed scope explicit (scope decision 2026-07-17; PLAN.md Decision log / Risk R-4). **Results-independent — safe to add now.**

### 4.2 WILL-CHANGE-AFTER-RETRAIN (insert a placeholder note, NOT a number, until Phase 3 completes)
- **`tab:jd_results` (L646–663):** every row — JD env recalibration + arch unification.
- **`tab:sim_params` jump rows (L554–556):** $\lambda_J, \mu_J, \sigma_J$ → recalibrated values.
- **`tab:ac_results` DQN & DDQN rows (L596–614):** arch 128→64.
- **`tab:taq_results` DQN & DDQN rows (L798–816)** and **`tab:taq_comparison` DQN/DDQN rows (L827–840):** arch 128→64.
- **Abstract/intro/prose numbers that depend on DQN/DDQN:** the 5.2×/7.3× TAQ multipliers (L67, L103), the 5.36× vol ratio (L842), the JD-vs-DQN/DDQN percentages (L667, L669), the ≥87.3% worst-RL claim (L823).
- **Stable through the retrains (safe to keep, re-verify):** TAQ TWAP-relative headlines **88.8% / 97.0%** (L66), the IQN-only TAQ claims (84.9% ↓, 4.02/48.76 bps, 3.93×), and the AC IQN-vs-TWAP 13.1%/21.7%.

> **Suggested placeholder convention** (in the tex, at edit time): `\textcolor{red}{[TBD-retrain: JD table pending Phase-3 rerun]}` so nothing ships with a superseded number.

---

## §5 — Internal inconsistencies & notation (independent of retrains)

### 5.1 Rounding slips in JD prose (L669)
Paper says "**10.7%** and **15.1%**"; its own table values give **10.79% → 10.8%** and **15.35% → 15.3%** (matching the commented abstract). And "**49.5%** higher" → **49.6%**. These become moot once JD is retrained, but flag them so the recomputation is done consistently.

### 5.2 Dead duplicate abstract (L76–80)
A second, commented-out abstract uses **jump-diffusion** headlines (55.8% / 64.5%) while the active abstract (L49–75) uses **TAQ** headlines (88.8% / 97.0%). Delete the stale commented block (or reconcile) so there is one source of headline numbers.

### 5.3 Symbol collisions (reused letters) — recommend disambiguation
- **$\lambda$** = discount factor (L551, L778) **and** jump intensity in Eq. `jd_price` (L514) **vs** $\lambda_J$ in the table (L554). Use $\gamma_d$ or keep $\lambda$ only for discount; always write $\lambda_J$ for jumps.
- **$\eta$** = temporary impact (L547, L776) **and** CVaR truncation level (L348, L362) **and** $\eta_{\text{lr}}$ learning rate (L562). The conclusion (L901–902) calls the risk dial "$\alpha$" but the methodology uses $\eta$ — pick one symbol for the CVaR level.
- **$\gamma$** = permanent impact (L548) **and** the discount factor in the distributional Bellman equations (L240, L244).

### 5.4 Depth wording (L580 vs Eq. L276)
Prose says "two hidden layers"; the encoder equation (L276) is written with generic $W_1\ldots W_L$. Confirm the concrete depth (code: `n_hidden_layers=2`) and make the equation state $L=2$.

---

## §6 — Flagged for verification (do NOT edit paper until confirmed)

- **F-A — Already-correct TAQ params:** §1.2 — the D-list's TAQ "fixes" are already satisfied; do not change them.
- **F-B — AAPL price scale (data integrity, verified → MEDIUM, NOT results-invalidating):** `tab:data_summary` (L723–740) reports mid-price mean \$295.84 / max \$651.16. **Confirmed non-split-adjusted:** `data/processed/AAPL_2014.parquet` `mid_price` = **647.84 (2014-06-06) → 93.19 (2014-06-09)** at AAPL's 7:1 split. **But the reported results are NOT corrupted:** `TAQEnv` sets arrival price per-episode from the first bar (`taq_env.py:163`) and normalizes IS/state relative to it (`:200,215,243`); episodes are `N` consecutive intraday bars so none straddle the split; the test window (Oct–Dec) is entirely post-split. **The headline 88.8%/97.0% (TAQ) stand.** Residual (real but mild): temporary impact is absolute-dollar (`eta*x_t`, `:207`), so its bps cost scales inversely with price — training on Jan–Jul mixed price levels sees a ~7× impact-scale inconsistency that could bias the policy. **Recommendation for the thesis version:** use split-adjusted prices or price-relative impact; escalate to the user, do not auto-fix. Also note `tab:data_summary`'s mixed-split descriptive stats (mean \$295.84) are technically accurate for the raw series but misleading — consider a footnote.
- **F-C — JD checkpoint selection:** early "best" checkpoints (IQN ep9000 / DDQN ep3000) are an artifact of the drift env; just confirm the post-fix reruns select sensible epochs.
