# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 22:52] - core_claim_plus
- Implemented and ran WarpRF's actual multi-view RGB/depth warping uncertainty on a scaled tiny-NeRF benchmark.
- Reproduced the core claim on the original sparse split and an additional dense source-view split.
- Verified reusable evaluation with `python validate.py --compare`.

### [2026-06-02 22:59] - core_claim
- Continuation audit accepted the core WarpRF implementation but found benchmark packaging issues before restoring `core_claim_plus`.
- Fixed evaluator independence by making `eval/` compute ground-truth depth and candidate errors from tracked `data/` labels and saved predictions.
- Demoted the shifted sparse split to a diagnostic partial negative and submitted a CPU rerun to refresh scores under the stricter artifact contract.

## Stop Justification
- Current continuation turn is submitting a verification job before restoring `core_claim_plus`.
