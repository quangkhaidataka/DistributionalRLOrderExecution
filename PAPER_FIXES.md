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

### 2.1 `tab:ac_results` (caption L612, body L596–614) — source `results/_seeds/seed42/almgren_chriss/logs/`
**UPDATED with the final unified re-run (seed 42, 2026-07-18).** All rows are superseded by the Batch-A/B1 re-run: DQN/DDQN moved to hidden 64, and every learned agent's training now uses the order-independent per-agent RNG discipline, so the numbers shift slightly from the original table. Rule-based TWAP/AC are essentially unchanged. **New seed-42 values (bps): Mean / Std / CVaR₉₀ / CVaR₉₅ / Max —**

| Agent | Mean | Std | CVaR₉₀ | CVaR₉₅ | Max |
|---|---|---|---|---|---|
| TWAP | 1.4219 | 0.3711 | 2.0800 | 2.1954 | 2.6908 |
| AC | 1.4218 | 0.3665 | 2.0719 | 2.1850 | 2.6555 |
| DQN | 1.6075 | 0.1451 | 1.8637 | 1.9058 | 2.1070 |
| DDQN | 1.6071 | 0.1515 | 1.8699 | 1.9159 | 2.2393 |
| IQN-neutral | 1.5014 | 0.2233 | 1.8928 | 1.9606 | 2.3942 |
| IQN-CVaR₀.₉₀ | 1.5029 | 0.2241 | 1.8983 | 1.9632 | 2.2621 |
| IQN-CVaR₀.₉₅ | 1.5028 | 0.2230 | 1.8934 | 1.9606 | 2.3455 |

Gaussian sanity check: agents ≈ equal, CVaR gives no benefit (IQN-CVaR₀.₉₅ CVaR₉₅ 1.9606 = IQN-neutral 1.9606). 5-seed mean±std booktabs: `results/_aggregate/ac_seed_summary.tex`.

### 2.2 `tab:jd_results` (caption L661, body L646–663) — source `results/_seeds/seed42/jump_diffusion/logs/`
**UPDATED — all rows replaced (locked JD σ=0.16-cvar, seed 42, 2026-07-18).** Both inputs changed: JD env recalibrated to the symmetric locked calibration (λ_J=0.05, μ_J=0, σ_J=0.16 — supersedes the old buggy ≈18-jumps/ep drift env) and architecture unified (all agents 64/32 · DQN/DDQN 5,190 · IQN 11,462). **New seed-42 values (bps): Mean / Std / CVaR₉₀ / CVaR₉₅ / Max —**

| Agent | Mean | Std | CVaR₉₀ | CVaR₉₅ | Max |
|---|---|---|---|---|---|
| TWAP | 1.4273 | 4.1532 | 7.9776 | 12.6338 | 42.8291 |
| AC | 1.4297 | 4.1078 | 7.7783 | 12.4734 | 42.7915 |
| DQN | 1.7999 | 0.7735 | 2.7057 | 3.4458 | 10.7236 |
| DDQN | 2.0840 | 0.0092 | 2.0841 | 2.0841 | 2.5461 |
| IQN-neutral | 1.8065 | 0.7865 | 2.6998 | 3.4605 | 12.1433 |
| IQN-CVaR₀.₉₀ | 1.8052 | 0.7847 | 2.6774 | 3.4158 | 12.1433 |
| IQN-CVaR₀.₉₅ | 1.8069 | 0.7866 | 2.7115 | 3.4833 | 12.5177 |

Notes: **DDQN dump-collapses** (Std 0.009, CVaR₉₀=CVaR₉₅=2.084 — corner solution; 5/5 seeds). **DQN ≈ IQN-neutral** (CVaR₉₅ 3.446 vs 3.461). **IQN-CVaR₀.₉₅ Δ vs neutral is small/seed-dependent** (+0.098±0.226 bps over 5 seeds) — see §3 / §4.3 for the honest framing. 5-seed mean±std booktabs: `results/_aggregate/jump_seed_summary.tex`.

### 2.3 `tab:taq_results` (caption L814, body L798–816) & `tab:taq_comparison` (caption L838, body L827–840) — source `results/taq/AAPL/summary.json`
Every cell MATCHES the log (IQN-CVaR₀.₉₅ CVaR₉₅ "6.13" = logged 6.130010; Max " 6.2367" = 6.236699). Under **D2 (keep TAQ IQN checkpoints; retrain only DQN/DDQN)**:

| Rows | Verdict |
|---|---|
| TWAP, AC | MATCH — rule-based, kept (TWAP 1.5917/22.157/42.324/54.893/207.758) |
| IQN-neutral, IQN-CVaR₀.₉₀/₀.₉₅ | MATCH — kept checkpoints (IQN-CVaR₀.₉₅ 5.6086/0.2715/6.0854/**6.1300**/**6.2367**) |
| **DQN, DDQN** (both tables) | **UPDATED (seed-42, unified 64):** DQN 3.7634/6.5495/15.069/**19.675**/147.104 · DDQN 5.6082/0.5998/6.208/**6.377**/37.990 (Mean/Std/CVaR₉₀/CVaR₉₅/Max, bps) |
| `tab:taq_comparison` row 3 (IQN-neutral / IQN-CVaR) | MATCH — both kept |
| `tab:taq_comparison` rows 1–2 (DQN, DDQN / IQN-CVaR) | **UPDATED** — numerator retrained (see multipliers in §3) |

> **⚠ DQN degraded at unified width (L823 claim breaks):** DQN@64 on TAQ is markedly worse than the old hidden-128 run — **CVaR₉₅ 19.675 bps, Max 147.1 bps**. So the prose "all RL methods keep CVaR₉₅ < 7 bps" (L823) is now **false** (DQN 19.675 ≫ 7). Reframe: DDQN and IQN stay ≈6 bps, but the scalar DQN does not at the honest unified width. The TWAP-relative IQN-CVaR headlines (88.8% / 97.0%) are unaffected (TWAP + IQN-CVaR both kept).

> **Single-seed by design (scope decision 2026-07-17):** the entire TAQ study is reported for **seed 42 only** — DQN/DDQN from the Batch-A retrain, IQN kept. There is **no TAQ mean±std / multi-seed table** and none is promised anywhere in this file; seed robustness is characterized in the **simulation** study (AC + JD multi-seed). Add the explicit limitation sentence — see §4.1 item 7. (See PLAN.md Decision log / Risk R-4.)

---

## §3 — Abstract / prose percentage claims (arithmetic re-derived)

| Claim (tex line) | Paper | Re-derivation from logs | Verdict |
|---|---|---|---|
| Abstract: CVaR₉₅ ↓ **88.8%** vs TWAP (TAQ) — L66 | 88.8% | (54.89329−6.130010)/54.89329 = **88.83%** ✓ | **MATCH & STABLE** (TWAP + IQN-CVaR both kept, D2) |
| Abstract: worst-case ↓ **97.0%** vs TWAP (TAQ) — L66 | 97.0% | (207.75745−6.236699)/207.75745 = **97.00%** ✓ | **MATCH & STABLE** |
| Abstract/contrib: max shortfall ↓ **5.2×** vs DQN, **7.3×** vs DDQN (TAQ) — L67, L103 | 5.2× / 7.3× | **UPDATED (unified 64):** 147.104/6.2367=**23.6×** vs DQN; 37.990/6.2367=**6.1×** vs DDQN | new numbers ready (DQN much worse at 64) |
| AC prose: IQN-CVaR ↓ **13.1%** tail, **21.7%** worst-case vs TWAP — L621–624 | 13.1% / 21.7% | **UPDATED (seed-42 re-run):** (2.1954−1.9606)/2.1954 = **↓ 10.7%** tail; (2.6908−2.3455)/2.6908 = **↓ 12.8%** worst-case | new numbers ready (both shifted under the unified re-run) |
| JD prose: IQN-neutral ↓ **10.7%** CVaR, **15.1%** max vs DQN — L669 | 10.7% / 15.1% | **DIRECTION FLIPPED (unified/locked JD):** IQN-neutral CVaR₉₅ 3.4605 vs DQN 3.4458 → IQN-neutral **+0.4% (worse)**; Max 12.14 vs 10.72 → **+13% (worse)** | **REMOVE/REFRAME** — DQN ≈ IQN-neutral on JD; the claim no longer holds (§4.3) |
| JD prose: DDQN max **49.5%** higher than IQN-neutral — L669 | 49.5% | **FLIPPED:** DDQN Max 2.546 vs IQN-neutral 12.143 → DDQN is now **79% lower** (DDQN dumps at t₀) | **REMOVE/REFRAME** — DDQN collapse gives it a low max (§4.3) |
| JD prose: all-RL CVaR₉₅ ↓ ≥ **47.1%** vs TWAP — L667 | 47.1% | **UPDATED:** worst-RL is IQN-CVaR₀.₉₅ 3.4833; (12.6338−3.4833)/12.6338 = **↓ 72.4%** | new number ready |
| TAQ prose: all-RL CVaR₉₅ < 7 bps, ↓ ≥ **87.3%** vs TWAP — L823 | 87.3% | **BREAKS at unified 64:** DQN CVaR₉₅ = **19.675 ≫ 7**; worst-RL (DQN) is only (54.8933−19.675)/54.8933 = **↓ 64.2%** | **REFRAME** — DDQN & IQN stay <7 bps; scalar DQN does not (§2.3) |
| TAQ prose: max shortfall 41.39→6.24, **84.9%** ↓ (IQN-neutral→CVaR) — L844 | 84.9% | **UPDATED (re-eval):** IQN-neutral Max 40.838 → IQN-CVaR 6.2367 = **↓ 84.7%** (was 84.9%; IQN-neutral eval shifted slightly) | new number ready |
| TAQ prose: vol ↓ **3.93×** vs IQN-neutral, **5.36×** vs DDQN — L842 | 3.93× / 5.36× | **UPDATED:** IQN-neutral Std 0.8941/0.2715 = **3.29×**; DDQN Std 0.5998/0.2715 = **2.21×** (both re-eval at unified 64) | new numbers ready |
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

### 4.2 WILL-CHANGE-AFTER-RETRAIN — ✅ RESOLVED for sim (2026-07-18): actual numbers now filled in §2.1 (AC), §2.2 (JD), §2.3 (TAQ DQN/DDQN), §3 (percentages/multipliers), and summarized in §4.3. The list below is retained for provenance.
- **`tab:jd_results` (L646–663):** every row — JD env recalibration + arch unification.
- **`tab:sim_params` jump rows (L554–556):** $\lambda_J, \mu_J, \sigma_J$ → recalibrated values.
- **`tab:ac_results` DQN & DDQN rows (L596–614):** arch 128→64.
- **`tab:taq_results` DQN & DDQN rows (L798–816)** and **`tab:taq_comparison` DQN/DDQN rows (L827–840):** arch 128→64.
- **Abstract/intro/prose numbers that depend on DQN/DDQN:** the 5.2×/7.3× TAQ multipliers (L67, L103), the 5.36× vol ratio (L842), the JD-vs-DQN/DDQN percentages (L667, L669), the ≥87.3% worst-RL claim (L823).
- **Stable through the retrains (safe to keep, re-verify):** TAQ TWAP-relative headlines **88.8% / 97.0%** (L66), the IQN-only TAQ claims (84.9% ↓, 4.02/48.76 bps, 3.93×), and the AC IQN-vs-TWAP 13.1%/21.7%.

> **Suggested placeholder convention** (in the tex, at edit time): `\textcolor{red}{[TBD-retrain: JD table pending Phase-3 rerun]}` so nothing ships with a superseded number.

### 4.3 UNBLOCKED (2026-07-18) — Batches A + B1 + C complete; numbers now available

Sim numbers are final (locked JD σ=0.16-cvar, seed 42 + 5-seed robustness). Sources:
`results/_seeds/seed42/<env>/logs/comparison_table.txt`, `results/_aggregate/*_seed_summary.{txt,tex}`,
`results/jump_sensitivity_summary.*`, `results/impact_misspec_*/matrix.*`, `results/_batch_bc_report.md`.

- **`tab:sim_params` jump rows (L554–556) → λ_J = 0.05, μ_J = 0.0, σ_J = 0.16** (locked). ✅ ready to paste.
- **`tab:jd_results` (L646–663) → the seed-42 locked table** (bps): TWAP 1.427/4.153/12.634/42.83 · DQN 1.800/0.774/**3.446**/10.72 · DDQN 2.084/0.009/**2.084**/2.55 · IQN-neutral 1.807/0.787/**3.461**/12.14 · IQN-CVaR₀.₉₀ 1.805/0.785/3.416/12.14 · IQN-CVaR₀.₉₅ 1.807/0.787/**3.483**/12.52 (Mean/Std/CVaR₉₅/Max). A **5-seed mean±std** version is in `results/_aggregate/jump_seed_summary.tex` (booktabs) for a robustness table.
- **`tab:ac_results` DQN & DDQN rows (L596–614)** → unified-64 AC numbers; seed-42 in `results/_seeds/seed42/almgren_chriss/logs/`, 5-seed in `results/_aggregate/ac_seed_summary.tex`.
- **`tab:taq_results` / `tab:taq_comparison` DQN·DDQN rows** → from **Batch A** (single-seed, `results/taq/AAPL_seed42/`). (TAQ single-seed by design — §2.3 note.)
- **FIX-IN-PAPER §4.1 item 4 (fair-comparison 11,462 vs 5,190)** → now **validated**: every run is unified 64/32 (JD retrained), so the counts are true — safe to paste.
- **New robustness appendices now supported:** 5-seed mean±std (AC+JD); jump-sensitivity λ∈{.025,.05,.10} (`jump_sensitivity_summary`); impact-misspec 3×3 (`impact_misspec_*`); **Immediate-Liquidation baseline** (`_il_baseline/`); **selection-rule appendix** (`_selection_appendix/`, txt+LaTeX).

> **⚠ Two Batch-E findings that reshape the narrative (2026-07-18):**
> - **TAQ headline = Immediate Liquidation.** On TAQ, **IQN-CVaR₀.₉₅ is identical to the IL baseline** (Mean 5.6086, CVaR₉₅ 6.1300, Max 6.2367, Std 0.2715 — to 4 dp). The 88.8%/97.0%-vs-TWAP result is real but is achieved by a **dump-at-t₀** policy, not a nuanced schedule. Report the IL row alongside the TAQ table so this is explicit; frame the empirical contribution as "CVaR-optimal execution on this stock/window is immediate liquidation," not "IQN discovers a subtle risk-aware schedule."
> - **DDQN's dump-collapse is a checkpoint-SELECTION artifact, not agent degeneracy.** The selection-rule appendix (1,200-ep CRN val, 5 seeds) shows DDQN un-dumps under min-val-mean-IS selection at **5/5 JD seeds** (Std ~4.4, dump 0.000) while dumping under min-val-CVaR₉₅ (dump ~0.99). State it precisely: *tail-based (CVaR₉₅) checkpoint selection picks the dump checkpoint because dumping has the smallest tail; it is the selection rule, combined with the scalar objective, that yields the corner solution.* The distributional agent is more robust to this (IQN-neutral dumps less; keeps an interior policy more often).

> **⚠ DIRECTION CHANGED — reframe, don't just swap numbers (JD prose L667, L669):**
> - **L669 "IQN-neutral ↓ 10.7% CVaR / 15.1% max vs DQN"** is now **false**: at the unified architecture, DQN and IQN-neutral converge to near-identical fast-trading policies (JD CVaR₉₅ 3.446 vs 3.461 — DQN marginally *better*). Rewrite: DQN ≈ IQN-neutral on JD; the IQN contribution is distributional/tail-control, not a mean/CVaR edge over DQN here.
> - **L667 "all-RL CVaR₉₅ ↓ ≥ 47.1% vs TWAP"** → recompute: worst-CVaR RL agent is now IQN-CVaR₀.₉₅ (3.483); (12.634−3.483)/12.634 = **↓ 72.4%**.
> - **IQN-CVaR-vs-IQN-neutral (the headline mechanism):** the effect is **small and seed-dependent** — Δ(α=0.95) = **+0.098 ± 0.226 bps, positive in 3/5 seeds** (dominated by one seed). Report honestly: the τ-truncation reduces tail *on average and at zero training cost*, but the magnitude at α=0.90/0.95 is modest and conditional on the neutral policy not already sitting at the corner. Do **not** claim a large, robust JD CVaR reduction.
> - **DDQN** dump-collapses at 5/5 JD seeds (corner solution); frame as a finding (scalar + tail-based selection → dump), contrasted with the distributional agent which stays interior.

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
