# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-02 22:52] - core_claim_plus
- Implemented and ran WarpRF's actual multi-view RGB/depth warping uncertainty on a scaled tiny-NeRF benchmark.
- Reproduced the core claim on the original sparse split and an additional dense source-view split.
- Verified reusable evaluation with `python validate.py --compare`.

### [2026-06-02 22:59] - core_claim
- Continuation audit accepted the core WarpRF implementation but found benchmark packaging issues before restoring `core_claim_plus`.
- Fixed evaluator independence by making `eval/` compute ground-truth depth and candidate errors from tracked `data/` labels and saved predictions.
- Demoted the shifted sparse split to a diagnostic partial negative and submitted a CPU rerun to refresh scores under the stricter artifact contract.

### [2026-06-02 23:07] - core_claim_plus
- Job `8aa25908-8cb` completed successfully and regenerated all scaled benchmark artifacts under the stricter prediction-only artifact contract.
- Verified that artifacts no longer contain method-supplied `depth_error` or `candidate_oracle_errors`; `eval/` computes those labels from tracked `data/`.
- `python validate.py --compare` passes, with the core sparse split and dense source-view split supporting the WarpRF uncertainty and active-view claims.

## Stop Justification
- Completed at milestone `core_claim_plus`.
