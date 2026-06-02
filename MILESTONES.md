# Reproduction Milestones

**Current: core_claim**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-06-02 22:10] - none
- Read the WarpRF paper and supplementary tables.
- Implemented the paper's WarpRF warping-consistency computation and a scaled
  tiny NeRF reproduction harness.
- Prepared the first container job; no successful method run has completed yet.

### [2026-06-02 22:25] - none
- Job `475dc534-7af` produced WarpRF artifacts but failed before scoring due to
  a NumPy 2 evaluation API incompatibility.
- Patched the evaluator and prepared a rerun; no successful end-to-end method
  run has completed yet.

### [2026-06-02 22:35] - core_claim
- Job `00b6dbce-c5f` completed successfully and passed `python validate.py --compare`.
- On the scaled tiny-NeRF depth uncertainty benchmark, WarpRF improved AUSE
  from 0.283 to 0.222 and Spearman correlation with depth error from ~0.000 to
  0.627 under the same four-source-view constraint.
- On the active-view proxy benchmark, WarpRF selected the highest-error
  candidate view with zero selection regret under the same five-view budget.

## Stop Justification
- Not stopped.
