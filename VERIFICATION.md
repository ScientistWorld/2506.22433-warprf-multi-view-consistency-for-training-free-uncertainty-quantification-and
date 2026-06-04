# Verification Report

## Initial Assessment

The workspace started this pass at `core_claim_plus`. The implementation runs
the paper's actual multi-view RGB/depth warping uncertainty computation on a
scaled analytic radiance-field benchmark, writes reusable artifacts, and scores
those artifacts with evaluator code that does not import from `method/`.

`python validate.py --compare` already passed before this final cleanup. The
claimed milestone was justified, with the main limitation that the reproduction
is scaled and synthetic rather than the full paper benchmark matrix.

## Issues Found

- **Minor**: The reproduced diagnostic shifted-camera experiments had too much
  influence after the gym renormalizes weights over experiments present in
  `scores.json`. They are useful stress tests, but should not dominate the
  primary reproduced claims.
- **Minor**: A method-safe guidance file named the paper method in a diagnostic
  target note where a generic description was sufficient.
- **Cosmetic**: The working tree still contains pre-existing deleted trajectory
  runtime artifacts. These are unrelated to the gym contents and were left
  untouched.

## Fixes Applied

- Reweighted the reproduced scaled experiments in `scoring/reference.json` so
  the primary sparse and dense targets carry most of the effective gym score,
  while shifted diagnostics remain visible at lower weight.
- Rephrased `scoring/TARGETS.md` so the diagnostic note describes the current
  reproduced method without leaking the paper method name.
- Re-ran `python validate.py --compare`; validation passes after the changes.

## Final Judgment

The workspace is ready for gym use. From-scratch agents can read the problem,
data, and evaluation without method-specific leakage in the strict
method-agnostic files, and improve-mode agents have a working paper-method
implementation plus scored artifacts to compare against.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: medium
- Key limitation: scaled synthetic radiance-field benchmark rather than the full paper dataset/backbone matrix.
