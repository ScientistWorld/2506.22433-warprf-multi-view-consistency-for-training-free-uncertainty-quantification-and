# Train/Test Evaluation Design Notes

Current design (iteration 11 hardening; details and audit in `HARDENING.md`):

- `eval/train/evaluate.py` and `eval/test/evaluate.py` are independent copies of
  one protocol. They differ only in their instance folder (`data/train_assets/`,
  16 public scenes, vs `eval/test/assets/`, 32 hidden scenes), slice name and
  output file. Neither imports from the other, from `method/` or from `data/`.
- Methods register `outputs/<name>/predict.json`. Both evaluators run every
  registered command in two rounds, derive the acquisitions from the returned
  scores, and score against their own frozen answers. Baselines (random
  uncertainty, random and farthest-camera acquisition) are recomputed by the
  evaluator itself.
- Invalid or failed runs receive the worst achievable value of every metric.
  Score files are deleted before scoring and written atomically.
- Test instances are built by `eval/test/build_assets.py` from seeds that exist
  only in `eval/test/scene_seeds.json`; train instances by
  `python -m data.build_assets --seeds 0-15`. Both use the public generator
  (`data/scenes.py`) and backbone recipe (`data/backbone.py`).
