# Prereg v2 — JitRL vs grounded LoRA RL task-accuracy contrast (non-inferiority, interval rule)

**Frozen:** 2026-11-01T10:05+08:00 (Asia/Shanghai) · **Author:** Experiment Bot (ALWM18R2) · **Status:** FROZEN BEFORE ARM B SCORING
**Supersedes:** the verdict rule of `prereg_jitrl_vs_lora_acc.md` (v1, MD5 `a27359fc4e7d17b4f8ab06c9efd356b3`). v1's science question, arms, tasks, budgets and metric stand; **only the decision rule is replaced by §5 below.** v1 and `AMENDMENT1_arm_b_budget_parity.md` (`99da7632a0821f31829cd919f06635cb`) remain untouched on disk.

**Authority:** Lead v2 final ruling, 2026-11-01T09:59+08:00 (gates Arm B; replaces v1's point-estimate rule).

---

## 1. Pins

| Item | Value |
|---|---|
| Runnable manifest | `MANIFEST_jitrl_vs_lora_acc.md`, MD5 **`235FC9D21446A87D0B13A35DB6523250`**, 85,170 B |
| Manifest MD5 from RD with 4B-load evidence | **PENDING** — Arm B is hard-gated until Lead/RD publish it and Exp verifies it (`GO_ARM_B` gate, §7) |
| v1 prereg (unchanged) | `prereg_jitrl_vs_lora_acc.md`, MD5 `a27359fc4e7d17b4f8ab06c9efd356b3` |
| AMENDMENT 1 (B budget parity, warmstart, π₀ disclosure) | `AMENDMENT1_arm_b_budget_parity.md`, MD5 `99da7632a0821f31829cd919f06635cb` |
| AMENDMENT 2 (B release gate, evaluation-only scored phase) | `AMENDMENT2_arm_b_gate_and_eval_only.md` (MD5 in the run's PROVENANCE) |
| Commit pin | `5472f1f43e84750cb2b0760b17b5853cd788eddb` (branch `jitrl-vs-lora-acc`) |
| Executor | `pi05_neutral_lora_v4_v5init/299`, md5-list sha256 `977eb6e9…`, served on :8401 |
| Backbone | `Qwen3.5-4B`, 43 files / 9,342,908,952 B (A forwards it; **B performs no backbone forward** — uniform π₀, see AMENDMENT1 §5) |

---

## 2. Science question (unchanged from v1)

Can **test-time JitRL replace grounded LoRA RL's weight-update rule**, with **no weight training**, at **equal task accuracy**? Objective = **replaceability**, not preference-training efficacy.

---

## 3. Arms, tasks, budgets (unchanged from v1 + AMENDMENT 1)

| Dimension | Value |
|---|---|
| Tasks (8, identical both arms) | `3-1_004_1_blue`, `3-1_006_1`, `3-2_032_7`, `3-2_032_8`, `3-2_097_8`, `3-2_097_10`, `3-9_118_1`, `3-10_117_1` |
| Seed | 0 (arm level; the shared hop's evaluator seed 7 is identical in both arms) |
| Arm A | `qwen_lora_rl.py` via `local_shims/run_local_a_arm.py`: 12 train episodes (LoRA + REINFORCE, 12 × `opt.step()`) + **5 scored eval episodes/task** on frozen weights |
| Arm B | `local_shims/run_local_b_arm.py`: 12 **unscored** adaptation episodes (memory-only, structured warmstart `ep % k`) + **5 scored eval episodes/task** on frozen memory; β = 1.0; `argmax(Q−V)` |
| Prompt | `plan_prompt(selected, info)` (`--prompt-mode legacy`) in both arms — the executor sees the same prompt construction |
| Primary metric | per-episode `sr` from the evaluator's `rollout_summary.yaml`; pooled over scored episodes |
| Scored data | `arm_a/<task>/eval.jsonl`, `arm_b/<task>/eval.jsonl` (5 rows each ⇒ 40 per arm) |
| Unscored data | `arm_a/<task>/train.jsonl`, `arm_b/<task>/adapt.jsonl` (never scored) |

---

## 4. Effect and interval

**Effect:** `D = mean_acc_B − mean_acc_A` on `sr`, computed over **pooled scored episodes** (`n_A = n_B = 40`; with equal per-task counts this equals the mean of per-task accuracies).

**Primary 95% CI — Newcombe hybrid-score interval** for the difference of two independent proportions (Newcombe 1998, method 10):
- Wilson score intervals `(l_A, u_A)`, `(l_B, u_B)` at 95% (z = 1.959964) for `p_A`, `p_B`;
- `lower = D − sqrt((p_A − l_A)² + (u_B − p_B)²)`, `upper = D + sqrt((u_A − p_A)² + (p_B − l_B)²)`.

**Secondary — task-stratified paired bootstrap:** 10,000 resamples, seed 0. Each resample draws 8 tasks with replacement, then within each drawn task draws scored episodes with replacement in **paired** (A episode i, B episode i) fashion; report the 2.5th/97.5th percentiles of `D*`. Reported alongside, **not** used for the decision.

---

## 5. Decision rule (replaces v1 §5)

Let `[L, U]` be the primary (Newcombe) 95% CI for `D`.

| Condition | Verdict |
|---|---|
| `L >= −0.05` | **MET** (non-inferiority established) |
| `U < −0.05` | **NOT MET** (B is inferior by more than the 5 pp margin) |
| `L < −0.05 <= U` (the interval crosses −0.05) | **UNTESTABLE — insufficient power**, *not* "not met" |

Order of checks (first match wins), all of which are untestable regardless of the interval:
1. Arm A `not_trained` (`updates == 0` in A's own `train.jsonl`), or either arm incomplete (fewer than 40 scored rows) ⇒ **UNTESTABLE**.
2. Warmstart failure: if **every** task's `Â` variance is 0 after B's 12 adaptation episodes ⇒ **UNTESTABLE** (report "warmstart failed"; no parameter tuning may be used to rescue it).
3. Any task whose scored-phase `Â` variance is 0 is **not counted toward "not met"** and is reported as untestable at the task level; the pooled interval is still reported.
4. v1's Arm B (the pre-amendment, degenerate `A ≡ 0` path) is recorded as **UNTESTABLE** and cannot be cited as a result of this rule.

**Power contingency (pre-written, no further approval required):** if the interval **crosses −0.05**, extend scored episodes to **20 per task for BOTH arms** (same seed 0, same tasks, same executor), then apply this same rule **once more**. No second extension, no other adaptation.

---

## 6. Required auxiliary read-outs

- Per-task `Â` (advantage) variance in B's scored phase (`arm_b/metrics.json`: `advantage_var_scored`, `advantage_zero_variance`).
- Per-task and pooled `sr`, plus derived `strict` (`SR>0 and preference_metric==1`) and `compliance` (`observed_matches_selected`) for both arms.
- Arm A optimizer steps actually executed (= number of `train.jsonl` rows × 1).
- Newcombe interval, bootstrap percentile interval, `D`, `n_A`, `n_B`, verdict, and the pre-condition table above.

---

## 7. Arm B release gate

Arm B **must not start** until the file
`artifacts/jitrl_vs_lora_acc/<run_id>/GO_ARM_B`
exists. The B driver waits for it (poll 30 s, timeout 24 h) and exits `GATED_NOT_RELEASED` with zero rows if it never appears. Reason: Arm B waits for RD's new manifest MD5 carrying 4B-load evidence plus Exp's verification. The gate state is recorded in `arm_b/gate_status.json` and in `PROVENANCE.md`.

---

## 8. Scope limits (stated up front)

- **PRELIMINARY: seed 0 only. No multi-seed.** Any claim is bounded by that.
- No re-running an arm to improve its number; no β tuning; no task substitution; no threshold movement after seeing data.
- Numbers may come only from this run's `eval.jsonl` rows. Historical `results/qwen35_lora_rl8*.json` never serve as the A baseline (v1 FC13); `docs/papers/CycleVLA.pdf` is a method reference only (v1 FC2).
- `is_effect_met` (this rule) is reported **separately** from `protocol_complete`.
