# Addendum 2 to prereg v2 (Lead ruling 2026-11-01 10:45 Asia/Shanghai)

**Pins:** `prereg_jitrl_vs_lora_acc_v2.md` MD5 `1daf51411d8cbad1de1da0f1dfcb597c`; `ADDENDUM_v2_lead_1004.md` MD5 `819b56cb73eb6f0d5ae9f651b19fe7ad`. This file adds environment and annotation requirements only; it does **not** modify v2 or ADDENDUM 1, and it changes **no verdict logic**.
**Run:** `20261101T0953_seed0_5472f1f`. **Authority:** Experiment Bot / Lead, 2026-11-01 10:45 Asia/Shanghai.

---

## (1) Which backbone Arm B's selector stands on

Arm B's selector operates on the **base `Qwen3.5-4B` plus the same frozen `299` executor** used by Arm A. This is exactly Arm A's **pre-training starting point**, and it satisfies the ruling: Arm A's trained Qwen LoRA adapter is precisely the layer that Arm B proposes to replace. Consequently the contrast stays inside the same grounded-LoRA setup — the only remaining difference is the update rule (weight updates vs test-time `z' = z + β·Â`).

## (2) Arm B's selector must run in its own separate virtualenv

B's selector must use a **dedicated venv of its own** — not Arm A's runner venv, and not the executor's (openpi/jax) venv. Rationale: the three roles have different, partly conflicting dependency pins (Arm A: transformers 5.5.4 + torch 2.11.0+cu130 + peft 0.21.2; executor: torch 2.7.1 + transformers 4.53.2 + jax 0.5.3), and B needs only a minimal numpy/PyYAML stack plus the prefmem scripts.

## (3) Shared-environment protection

`omp` (or any agent) **may not upgrade or uninstall packages in shared environments.** Any dependency work happens in a private env created for that purpose; existing venvs used by a running arm (Arm A's runner venv, the executor venv) are read-only for the duration of the run.

## (4) Theory T7 annotation (logging only, no verdict effect)

**Per task, at the end of the adapt phase**, log:

- `max beta*|Â|` — the largest absolute scaled advantage the selector produced for that task in the adaptation phase;
- the **top1−top2 gap** of the selector's `beta·Â` vector.

These are recorded as an annotation for Theory T7. They are **not** inputs to the verdict, not a gate, and never suppress or promote a result.

---

## Status against this addendum

| Item | Status |
|---|---|
| (1) selector on base 4B + 299 | Holds by construction: B performs no backbone forward; the selector's π₀ is uniform and the executor is the same `299` line (shared contract, FC11/FC12) |
| (2) separate venv for B's selector | **OPEN GAP — reported, not implemented.** The chain invokes the B driver with `$OPENPI_ROOT/.venv/bin/python`, i.e. the **executor's** venv, which this addendum forbids. Fixing it requires a new private venv plus a change to how B is launched; the launching script is executing and must not be edited. Proposal to Lead: create `~/projects/b_selector_venv` (numpy + PyYAML only) and launch B with it once `GO_ARM_B` is being prepared. |
| (3) shared-env protection | Arm A's runner venv and the executor venv are clean (`find site-packages -maxdepth 1 -newermt '2026-11-01 09:53'` → 0 in both). **`libero_rt311` was NOT clean: it was mutated during the run** (torch 2.15.0+cpu 09:07:16, accelerate 1.15.0 10:01:12, transformers 5.5.4 10:09:34, torchvision 0.29.2+cpu 10:26:17) — not by Exp. The Lead froze it at 10:47 at its current versions; affected Arm A eval rows are listed in `PROVENANCE.md` §5.20 |
| (4) T7 annotation | Implemented as logging-only in `prefmem-libero/local_shims/run_local_b_arm.py` (per-task `t7_annotation` block: `max_beta_abs_advantage`, `top1_top2_gap_final`, `top1_top2_gap_max`). No change to `compare_arms_v2.py` or any verdict logic |

---

## Frozen client inventory — `libero_rt311` (frozen at 2026-11-01T10:47 by Lead ruling)

Read-only snapshot of `~/projects/libero_rt311/.venv/lib/python3.11/site-packages`; **no package in this venv was restored, installed, uninstalled or modified by Exp.** Versions below are the CURRENT (= frozen) state.

| Item | Value |
|---|---|
| dist-info count | **95** |
| Freeze listing (`name==version`, sorted, LF) sha256 | **`c0a21a38cf5d3aff7e538881584c84cf9addeb9b671ac5fec40e40d5b428621a`** |
| `torch` | **2.15.0+cpu** — `torch-2.15.0+cpu.dist-info`, RECORD md5 **`f786669fa2b5556d9fc4ec896b3f351d`**, dist-info mtime 09:07:16 |
| `torchvision` | **0.29.2+cpu** — `torchvision-0.29.2+cpu.dist-info`, RECORD md5 **`064e397d37769fdbf0e2b562fdf6b798`**, dist-info mtime 10:26:17 |
| `transformers` | **5.5.4** (dist-info mtime 10:09:34) |
| `accelerate` | **1.15.0** (dist-info mtime 10:01:12) |
| `tensorflow` | **ABSENT** — no `tensorflow*` dist-info and no `tensorflow` directory |
| `peft` | **ABSENT** — 0 `peft*` dist-info, no `peft` directory |
| Other pins of record | numpy 1.26.4, robosuite 1.4.1, mujoco 2.3.7, bddl 3.6.0, gym 0.26.2, msgpack 1.2.4, tokenizers 0.22.2, safetensors 0.8.0, huggingface_hub 1.33.0 |

**Drift note (recorded, not repaired — Lead 10:47 froze the venv):** this venv was mutated **during** the run: torch 2.15.0+cpu at 09:07:16, accelerate 1.15.0 at 10:01:12, transformers 5.5.4 at 10:09:34, torchvision 0.29.2+cpu at 10:26:17. Those are not Exp actions (Exp issued no `pip`/`uv` command against this venv at any point). Consequence for Arm A's already-written eval rows is listed in `PROVENANCE.md` §5.20.

## Factual status of item (1) — recorded, not changed

Exp's shipped B selector (`plan_jitrl_vlm_text_selector.py`) uses a **uniform π₀** and performs **no** Qwen3.5-4B forward pass (`run_local_b_arm.py` → `backbone_forward: false`; disclosed in AMENDMENT 1 §5). Item (1) is therefore a statement about *which model B's selector stands on conceptually* (base 4B + 299 = A's pre-training starting point), not a description of a forward pass. Making π₀ literal 4B logits would be a **protocol change** (new selector semantics, new GPU residency) and is **not** implemented here — proposal to Lead only.
