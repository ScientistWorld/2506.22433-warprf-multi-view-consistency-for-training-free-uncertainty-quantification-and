# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-02 22:46] - core_claim
- Implemented the WarpRF uncertainty algorithm from the paper: render RGB-D source and target views, backward-warp source renderings using target depth and camera poses, and score uncertainty from multi-view consistency.
- Ran a scaled tiny NeRF reproduction where WarpRF improved depth-error ranking and active candidate selection over a deterministic random uncertainty baseline.

### [2026-06-02 22:48] - core_claim_plus
- Expanded the scaled benchmark to shifted sparse and dense source-view splits.
- Dense depth uncertainty and both active-view proxy settings support the central claim under fixed source-view budgets.

### [2026-06-02 23:02] - core_claim_plus
- Continued-agent audit verified that `method/warprf.py` implements the paper's depth and image reprojection uncertainty rather than a learned surrogate.
- Rechecked evaluator separation: `eval/metrics.py` reads saved artifacts and does not import `method/` or regenerate the method.
- Re-ran `python validate.py --compare`; all scoring and structure checks passed.

## Stop Justification
- Completed at milestone `core_claim_plus`.
