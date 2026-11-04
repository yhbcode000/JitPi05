# Addendum 4 to prereg v2 — same-task retrieval for Arm B (Lead ruling 2026-11-01 11:15 CST)

**Pins:** `prereg_jitrl_vs_lora_acc_v2.md` `1daf51411d8cbad1de1da0f1dfcb597c` · `ADDENDUM_v2_lead_1004.md` `819b56cb73eb6f0d5ae9f651b19fe7ad` · `ADDENDUM2_v2_lead_1045.md` `28036cc8b8aa1e01e84717df4282dc12` · `ADDENDUM3_v2_lead_1101.md` `64c31a8507a87b217217c095e7415e1d`.

1. **Arm B's kNN retrieval returns only entries whose recorded `task_id` equals the current task.** Cross-task neighbours are never retrieved and never contribute to `Q`, `V`, `Â`, the selection, or the reported scores.

2. **Memory still accumulates across tasks in Arm A's order**, and per-task memory snapshots plus MD5 stay exactly as required by ADDENDUM3. Nothing else in ADDENDUM3 changes.

3. **Reason.** Arm A's input contains the instruction, while Arm B's memory key is a scene description only. Borrowing a preference return learned under a different instruction would handicap Arm B and would violate the P4a premise of the comparison.

4. **`cross_task_retrieval_fraction` logging is kept as a check and must be 0.** A non-zero value means the filter is not in force; it is a guard, not a scored quantity.

5. **Cross-machine.** Arm B running on the 4090 is **PROVISIONAL** until the ALWM action-consistency half (deferred until Arm A finishes) passes. If that half fails, Arm B re-runs in full on ALWM.

6. **`GO_ARM_B` requires:** this addendum's MD5, the new MANIFEST MD5, and all static checks passing.
