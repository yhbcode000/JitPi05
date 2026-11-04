# Addendum to prereg v2 (Lead ruling 2026-11-01 10:04 Asia/Shanghai)

**Pins:** `prereg_jitrl_vs_lora_acc_v2.md`, MD5 **`1daf51411d8cbad1de1da0f1dfcb597c`** (this addendum does not modify it).
**Run:** `20261101T0953_seed0_5472f1f`. **Authority:** Experiment Bot / Lead, 2026-11-01 10:04 Asia/Shanghai.

---

## (1) Â-variance is measured against the memory frozen at the end of the adaptation phase

The zero-Â-variance test uses the memory **as frozen at the end of the 12 adaptation episodes**. The scored phase reads that frozen memory; no label is written during scoring. The per-task variance is computed over that task's scored episodes against that frozen memory.

## (2) Tie-break between the two intervals

The CI tie-break stands exactly as ruled: the **primary method (Newcombe hybrid-score interval) decides the verdict**. The task-stratified paired bootstrap is **secondary / reporting only** and **cannot override** the Newcombe-based decision.

## (3) Theory position

Theory concurs and calls Newcombe the **conservative** choice.

---

## Confirmation against the implementation

Checked `prefmem-libero/local_shims/compare_arms_v2.py` (MD5 `57e296ed5b05096706c0da514c3ac706`) against (1) and (2) — **both followed, no gap**:

- **(1) followed.** The engine reads `advantage` from B's **scored** rows only, and those rows are produced after the adapt phase with `memory_updated=false` (verified: `memory_after` constant `[4,4,4]` across scored episodes). Variance is therefore computed against the frozen memory, per task, over that task's scored episodes. `arm_b/metrics.json` also records `advantage_var_scored` / `advantage_zero_variance` from the same numbers.
- **(2) followed.** The verdict branches read only the Newcombe interval; the bootstrap interval is written to `comparison_v2.json` under `bootstrap_task_stratified_paired_ci95` and never enters any branch.
- **Implementation note (strict reading of "not counted toward not met").** Zero-Â-variance tasks are excluded from the pool used for the **"not met"** branch only. The **"met"** branch is judged on the **full pool**, so an untestable/degenerate task can never help a non-inferiority claim; it can only be kept out of an inferiority claim. Both intervals are reported (`newcombe_ci95` = excluded pool, `newcombe_ci95_full_pool` = full pool, plus `rule_applied`). This was added at 10:0x while Arm B is still gated and produced no episode; engine MD5 was `d8b37f9e…` before the change and is `57e296ed…` after.
- **Validated** (GPU-free): Newcombe reproduced in both argument orders (`56/70 vs 48/80` → `+0.0524, +0.3339`; reversed → `−0.3339, −0.0524`); synthetic branches met / not met / crossing / all-zero-variance; and the exclusion trap (7 tasks identical, 1 zero-Â-variance task where B fails 0/40) → full pool upper `−0.0784` (would read "not met") but the verdict is **untestable**, as this ruling requires.
