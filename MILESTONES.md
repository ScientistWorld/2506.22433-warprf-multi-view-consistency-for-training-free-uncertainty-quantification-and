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

### [2026-06-02 22:50] - core_claim
- Job `e3c662a6-b0d` completed the two-split run, but the shifted depth AUSE was
  worse than the deterministic random baseline while rank correlation and active
  view selection improved.
- Diagnosed that synthetic background fill pixels were included in depth AUSE;
  patched the artifact/evaluator to score only valid scene depths, matching the
  paper's annotated-depth evaluation intent.

### [2026-06-02 22:55] - core_claim
- Job `fecc9c62-9ce` completed the two-split run with valid-depth masking and
  passed `python validate.py --compare`.
- The original split still supports the core uncertainty claim, and the shifted
  split supports the active-view proxy, but shifted depth AUSE remained worse
  than random.
- Patched WarpRF depth uncertainty so target pixels with no valid source
  reprojection are high-uncertainty instead of false zero-uncertainty; prepared
  a rerun before considering `core_claim_plus`.

### [2026-06-02 23:05] - core_claim
- Job `744c73e2-b00` completed after the no-source-reprojection fallback patch
  and passed `python validate.py --compare`.
- Scores were unchanged, so the shifted sparse depth uncertainty failure is not
  caused by all-invalid target pixels.
- Added a dense eight-source-view split to test the same WarpRF method under a
  different multi-view support setting before considering `core_claim_plus`.

## Stop Justification
- Not stopped.
