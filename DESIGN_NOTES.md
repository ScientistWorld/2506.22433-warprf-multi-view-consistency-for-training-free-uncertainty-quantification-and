# Train/Test Evaluation Design Notes

Current design (final hardening; details and audit in `HARDENING.md`):

- `eval/train/evaluate.py` and `eval/test/evaluate.py` are independent copies of
  one protocol. They differ only in their instance folder (`data/train_assets/`,
  16 public scenes, vs `eval/test/assets/`, 56 hidden scenes), the folder they
  hide from the method, slice name and output file. Neither imports from the other, from `method/` or from `data/`.
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
- Both evaluators run each method command through their own copy of
  `isolate.py`: fresh user, mount and PID namespaces in which `eval/test/` and
  the git object store are covered by empty read-only mounts and `/proc` shows
  only the method's processes. A probe checks that the hidden folder really
  appears empty; if the kernel refuses namespaces the evaluator falls back to
  removing permissions from the hidden instances while the method runs, and
  records the mode used under `_isolation` in the score file.
