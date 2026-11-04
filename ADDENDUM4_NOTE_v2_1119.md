# Note on ADDENDUM4 — interpretation only (Lead ruling 2026-11-01 11:22 CST)

**Reference:** `ADDENDUM4_v2_lead_1115.md`. This note records how the ADDENDUM4 retrieval rule must be read when the verdict is reported. It changes no protocol, no threshold, no arm definition and no verdict logic.

1. **Arm B effectively cold-starts each task.** Under ADDENDUM4, B's kNN retrieval returns only entries whose recorded `task_id` equals the current task, so a task's scored episodes are driven by the labels gathered inside that same task. Arm A, by contrast, carries its LoRA gains forward across tasks — its task *k* evaluation runs on weights trained on tasks 1…*k*.

2. **That asymmetry is conservative against Arm B.** B is denied the cross-task carry that A enjoys, so a non-inferiority result cannot be dismissed as an artefact of B borrowing across requirements; the bias runs against the treatment arm.

3. **If the verdict is not met, the annotation must separate two claims.** The report must distinguish
   - *"no cross-task transfer"* — the observed deficit is attributable to withholding cross-task memory under this design, and
   - *"JitRL cannot replace RL"* — the general claim.

   Only the second may be stated if the evidence supports it beyond the same-task-retrieval restriction; a not-met verdict under ADDENDUM4 alone does not license the general claim.
