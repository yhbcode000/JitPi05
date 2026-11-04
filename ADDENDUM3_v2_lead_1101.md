# Addendum 3 to prereg v2 — Lead-approved protocol change (2026-11-01 11:01 / 11:02 CST)

**Pins:** `prereg_jitrl_vs_lora_acc_v2.md` MD5 `1daf51411d8cbad1de1da0f1dfcb597c`; `ADDENDUM_v2_lead_1004.md` MD5 `819b56cb73eb6f0d5ae9f651b19fe7ad`; `ADDENDUM2_v2_lead_1045.md` MD5 `28036cc8b8aa1e01e84717df4282dc12`.
**Supersedes:** run `20261101T0953_seed0_5472f1f` (see `SUPERSEDED.md` in that run dir). Nothing else changes — science question, arms, tasks, executor, metric, 5 pp margin, Newcombe decision rule and the ≥4-valid-task floor all stand as in v2.

---

## (i) Arm A is re-run from scratch on the frozen `libero_rt311`

- **From scratch**, no reuse of any weight, row or adapter from the superseded run.
- **Frozen client env:** `libero_rt311` exactly as frozen at 2026-11-01T10:47 CST. **Freeze identification method (fixed):** run `python -m pip freeze --all` with rt311's own interpreter, sort lexicographically (`LC_ALL=C sort`), LF line endings, UTF-8, then `sha256` the resulting text. (`pip` is not installed in rt311; the executed equivalent is `uv pip freeze --python <rt311 interpreter>`, which emits the same normalised `name==version` lines.) The frozen sorted list is stored at `E:\Envs\JitRL_LoRA_Env\env_freeze\rt311_freeze_sorted.txt` (WSL mirror `~/env_freeze/rt311_freeze_sorted.txt`).
- **Reference value: `pending RD/Exp match`.** Arm A **must not restart** until RD's freeze hash and Exp's freeze hash are equal. (Exp's current measurement with the method above is `d9efcd2b5c63ead92e130c7d3ba62d74a21b729189b57f72e323928fd8c3a7fe`; it is **not** the reference until RD's value matches — no other number is to be treated as the reference.)
- **No process may install, upgrade or uninstall packages into `rt311`** while the re-run is pending or running.
- **Protocol (unchanged from v1/v2 for this arm):** seed 0; hyper-parameters unchanged (`--temperature 2.0 --epsilon 0.3 --reward-gate sr --lr 1e-5 --lora-r 16 --lora-alpha 32 --train-episodes 12 --eval-episodes 5 --prompt-mode legacy --exec-gpu 0 --port 8401`); the **8 tasks in the original order**, sharing **one** LoRA across tasks.
- **New, required:** after each task's training completes, save a LoRA snapshot to `<out-dir>/snapshots/task_<i>_<task_id>/` together with a per-file MD5 manifest. This is what makes any later eval-only question answerable.

## (ii) Cross-task symmetry for Arm B (fixed before B starts)

- **One memory for the whole run**, accumulating across tasks — not a fresh memory per task.
- **Task order identical to Arm A's.**
- **Per task:** 12 adaptation episodes (memory accumulates) → 5 scored episodes (**memory frozen** for scoring) → carry the memory into the next task.
- **At the end of each task**, save a snapshot of B's memory plus its MD5, mirroring Arm A's per-task LoRA snapshot + MD5.

---

## Pins and scope

| Item | Value |
|---|---|
| v2 prereg | `1daf51411d8cbad1de1da0f1dfcb597c` (unchanged) |
| ADDENDUM 1 (Lead 10:04) | `819b56cb73eb6f0d5ae9f651b19fe7ad` (unchanged) |
| ADDENDUM 2 (Lead 10:45) | `28036cc8b8aa1e01e84717df4282dc12` (unchanged) |
| Superseded run | `20261101T0953_seed0_5472f1f` — `SUPERSEDED.md` |
| Verdict rule | unchanged: Newcombe 95% CI on D = p_B − p_A over the valid subset; MET if lower ≥ −0.05, NOT MET if upper < −0.05, else UNTESTABLE + contingency (20 eps/task both arms); ≥4 valid tasks required; bootstrap annotation only |
| Gate | unchanged: Arm B starts only when `GO_ARM_B` exists; this addendum does not release anything |

This addendum adds no other requirement and changes no verdict logic.
