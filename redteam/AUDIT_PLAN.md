The audit runs the real saved-artifact test contract in separate processes:
`bash scripts/evaluate_test.sh` with no arguments. Every case has its own
`run.sh`, saved train/test scores, and execution status. Evaluator sources are
hashed and checked after each run. The harness restores data, outputs and score
files between cases and after the job.

Quality checks: three independent initializations of the original 900-step CPU
reference on each of three camera protocols; a paired ray-distance to camera-z
geometry correction; 2700-step and 90-step training; removal of warping; constant
predictions. All training uses the original backbone, batches and initial views.

Shortcut checks: visible analytic-answer enumeration; fabricated legacy error
labels; false budgets and impossible flags; constant corrupted candidate images;
replacement of the fixed data generator; suppression through empty train result
keys; omission of an extra poor method left on disk; invalid selected-view field;
direct score writing; nonfinite outputs; stale scores after a failed test.

Ranks use absolute coefficients and metric direction from reference.json,
experiment weights, and population standard deviation across the finite audit
attempts. The hidden reference scores are the zero point. This explicit cohort
is necessary because future-attempt normalization is unknowable today. Raw
scores and componentwise comparisons are retained for independent re-ranking.
