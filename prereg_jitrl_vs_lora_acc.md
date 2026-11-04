# Prereg — JitRL vs grounded LoRA RL task-accuracy contrast

**Frozen:** 2026-11-01 (Asia/Shanghai) · **Author:** Experiment Bot (ALWM18R2) · **Status:** FROZEN BEFORE RUN — no number in this file may be edited after the driver starts.

**Pins this prereg to the runnable manifest**

| Item | Value |
|---|---|
| Manifest | `E:\Envs\JitRL_LoRA_Env\MANIFEST_jitrl_vs_lora_acc.md` |
| Manifest MD5 | `235FC9D21446A87D0B13A35DB6523250` |
| Manifest bytes | 85,170 |
| Manifest STATUS at freeze | `READY` (B1–B10 all RESOLVED) |
| Manifest commit pin | `5472f1f43e84750cb2b0760b17b5853cd788eddb` (branch `jitrl-vs-lora-acc`) |

If the manifest MD5 differs from the pin above, this prereg no longer applies and the run MUST NOT be scored against it.

---

## 1. Science question (unchanged from manifest §1 / §7)

Can **test-time JitRL replace grounded LoRA RL's weight-update rule**, with **no weight training**, at **equal task accuracy**?

- Known before this run: preference *training* works.
- Unknown: whether JitRL can **fully replace** the VLA RL update rule.
- Objective: **replaceability**, not merely preference-training efficacy.

Only the **update rule** differs between arms. Task set, backbone, executor, seed, episode budgets and metric are fixed (manifest §7 contract).

---

## 2. Arms (both frozen as written)

### Arm A — grounded LoRA RL (weight-level baseline)

| Item | Frozen value |
|---|---|
| Entry | `prefmem-libero/scripts/qwen_lora_rl.py`, run through `prefmem-libero/local_shims/run_local_a_arm.py` (shim rewrites only the three hard-coded `/hhd2/...` path constants; algorithm/hyper-parameters untouched) |
| Executor server start | `local_shims/serve_py_wrapper.sh scripts/serve_pi05.py --config pi05_libero_lora_neutral_v5init --checkpoint <ext4>/299 --port 8401` |
| Backbone | `Qwen3.5-4B` (`E:\Envs\JitRL_LoRA_Env\models\Qwen3.5-4B`, 43 files / 9,342,908,952 B) |
| Executor | frozen `pi05_neutral_lora_v4_v5init/299` (28 files / 9,363,286,152 B / md5-list sha256 `977eb6e9d52630877c938206ddf883f6b4e3aaa635ea7a7b286155355c219cc2`) |
| Update rule | REINFORCE on a LoRA adapter (`loss.backward()` + `opt.step()`), lr 1e-5, lora r=16, alpha=32, dropout 0.05, AdamW |
| Sampling | temperature 2.0, epsilon 0.3, reward gate `sr`, prompt mode `legacy` |
| Budget | `--train-episodes 12 --eval-episodes 5` per task |
| Seed / port / GPU | `--seed 0 --port 8401 --exec-gpu 0` |
| Eval-time policy | argmax over candidate-plan logits (no sampling) — this is the trained LoRA policy being measured |

### Arm B — test-time JitRL (treatment)

| Item | Frozen value |
|---|---|
| Entry | new local driver `prefmem-libero/local_shims/run_local_b_arm.py`, which composes the **unmodified** reference path named in manifest §6.5: `vlm_text_key.describe` (VLM text key) → JitRL selector arithmetic (`Q`, `V`, `A=Q−V`, `π_β ∝ π_0·exp(β·A)`, argmax) → `scripts/run_plan_with_vla.py` (299 hop) → reward read from the evaluator's `plan_vla_exec` `rollout_summary.yaml` |
| Backbone | same `Qwen3.5-4B` (used only as the frozen planner prior π_0; no gradients, no optimizer) |
| Executor | same frozen `299` served by the same wrapper on port 8401 |
| Update rule | **none to weights.** Non-parametric text memory `TextMemory` gains one entry per episode, gated `sr > 0` and `preference_metric is not None`, credit = selected plan (gain = `profile_pref_reward.plan_choice.mapped`) |
| β | `1.0` (manifest §6.5 template default; reference selector default) |
| Selection | `argmax(A)` with first-index tie-break (reference selector form; β only reweights the reported `probabilities`, matching `docs/preference_plan_story.typ:325` "main result uses argmax-Q") |
| Memory init | cold start per task (empty `TextMemory`; `Q ≡ 0.5` until the first gated write) — manifest B10 cold-start ruling, no synthetic memory |
| Budget | `--eval-episodes 5` per task, **no train episodes**, no checkpoint written |

---

## 3. Fixed shared protocol

| Dimension | Value |
|---|---|
| Tasks (8, same ids both arms) | `3-1_004_1_blue`, `3-1_006_1`, `3-2_032_7`, `3-2_032_8`, `3-2_097_8`, `3-2_097_10`, `3-9_118_1`, `3-10_117_1` |
| Mission dirs | each mission's own `upstream/LIBERO/dataset/series_3/...` package (`mission_info.yaml` present, 8/8) |
| State image (both arms) | each mission's own `screenshot/agentview_image_1.png` |
| Executor config | `pi05_libero_lora_neutral_v5init` on `299` for **all 8 tasks** (single frozen executor line; manifest §7 + FC12) |
| Seed | 0 |
| Eval episodes | 5 per task → 40 per arm |
| Primary metric | `mean_acc` = mean over tasks of per-task mean `sr` (`SR` field of the arm's eval rows) |
| Derived reporting fields | `strict` = `SR>0 AND preference_metric==1`; `compliance` = `observed_matches_selected` (explicit scalar preference compliance). Both are **derived from the same two logged fields in both arms**; neither replaces `sr` in the criterion below |
| Execution order | A fully, then its 299 server stopped, then B (single 16 GB GPU; FC8) |
| Run id | `20261101T<HHMM>_seed0_5472f1f` (HHMM = launch time) |

Per-task accuracy is the mean of that task's 5 eval `SR` values; `mean_acc` is the unweighted mean of the 8 per-task accuracies.

---

## 4. Explicitly excluded from this run

- Historical A numbers `prefmem-libero/results/qwen35_lora_rl8*.json` are a **side note only** — never the A baseline (Lead FC13: those files record `updates=0`, i.e. no evidence of a trained policy).
- `docs/papers/CycleVLA.pdf` (dirty `M`, MD5 `96dc4ba22b674a74d12b2a3fa0672a15`) is a **method reference only**, never a source of numbers (Lead FC2).
- No `full74` / `full74v3` artifact, path or quantity may be read, written or cited.
- No multi-seed. No second β. No arm swap. No re-running an arm to improve its number.

---

## 5. Verdict criterion (fixed before the run)

Evaluated on the replaceability claim, on `mean_acc` with a **5 percentage-point non-inferiority margin**:

- **untestable** if any of:
  - arm A is `not_trained` (zero LoRA update steps / `updates=0` in A's own train logs),
  - either arm fails to complete all 8 tasks × 5 eval episodes,
  - seed / backbone / executor / task set differ between arms,
  - a hard blocker (OOM, env reset failure, missing mission pack) prevents one arm from producing 40 eval rows.
- **met** if both arms complete, A is trained (`updates > 0`), and `mean_acc_B >= mean_acc_A − 0.05`.
- **not met** if both arms complete, A is trained (`updates > 0`), and `mean_acc_B < mean_acc_A − 0.05`.

**Reporting split (mandatory):** `is_effect_met` (the criterion above) is reported **separately** from `protocol_complete`. A run that fails the protocol must report `is_effect_met=NA/untestable` and may not substitute a softer claim.

`updates` for A = number of optimizer steps actually executed, counted from `loss.backward()`/`opt.step()` occurrences in the arm's own train rows (`train.jsonl` rows × steps per row) and cross-checked against the runner's `updates` counter in its cost log. Zero ⇒ `not_trained` ⇒ `untestable`.

---

## 6. What counts as a result file

| Arm | Files |
|---|---|
| A | `artifacts\jitrl_vs_lora_acc\<run_id>\arm_a\cost.jsonl`, `arm_a\<task>\train.jsonl`, `arm_a\<task>\eval.jsonl`, `arm_a\<task>\{train,eval}_ep<N>.json`, `arm_a\lora\` (adapter produced by A's own `save_pretrained`) |
| B | `artifacts\jitrl_vs_lora_acc\<run_id>\arm_b\<task>\decision_ep<N>.json`, `<task>\eval.jsonl`, `<task>\memory.npz`, `memory.json`, `metrics.json`, `run_summary.json`, `progress.json` |
| Both | `artifacts\jitrl_vs_lora_acc\<run_id>\PROVENANCE.md` (commit, seed, full command lines, 299 recomputed fingerprint, GPU snapshot, FC1–FC15 table) |

No file under `runs\policy_output\` may be cited as a result: that tree is the evaluator's scratch, overwritten per episode.

---

## 7. Failure discipline

Any hard blocker is recorded verbatim (log path + line) and reported as a blocker. **No number may be invented, extrapolated, averaged from a partial task set, or copied from a historical file.** If an arm cannot produce 40 eval rows, the verdict is `untestable` and that is the *result*.

---

## 8. Freeze record

| Item | Value |
|---|---|
| Frozen at | 2026-11-01T09:37+08:00 (pre-launch; exact MD5 below) |
| Prereg MD5 | recorded in `artifacts\jitrl_vs_lora_acc\<run_id>\PROVENANCE.md` and in the launch reply; this file is not edited afterwards |
| Copies | this file and a byte-identical copy beside the worktree manifest copy (`JitPi05-jitrl-vs-lora-acc\prereg_jitrl_vs_lora_acc.md`) |
