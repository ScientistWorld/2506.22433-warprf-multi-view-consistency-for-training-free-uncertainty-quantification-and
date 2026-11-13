"""Build the hidden test instances from the hidden seeds kept in this folder.

Only this directory holds the test seeds, the built test inputs and the test
answers. Run on a GPU node:  python eval/test/build_assets.py [--indices a-b --device cuda:k]
(resumable: finished scenes are kept; the manifest is written once all exist).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from data.build_assets import STEPS, build, parse_range, write_manifest  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(HERE / "assets"))
    p.add_argument("--indices", default="", help="a-b: build only scenes a..b (inclusive)")
    p.add_argument("--device", default=None)
    p.add_argument("--group", type=int, default=80)
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--count", type=int, default=0, help="smoke-test only: use the first N seeds")
    p.add_argument("--steps-scale", type=float, default=1.0, help="smoke-test only")
    args = p.parse_args()
    seeds = [int(s) for s in json.loads((HERE / "scene_seeds.json").read_text())["seeds"]]
    if args.count:
        seeds = seeds[: args.count]
    names = [f"test_{i:03d}" for i in range(len(seeds))]
    steps = {k: max(1, int(round(v * args.steps_scale))) for k, v in STEPS.items()}
    if args.manifest_only:
        write_manifest(Path(args.out), names, steps)
        return
    build(seeds, Path(args.out), names=names, group=args.group, steps=steps, include_seed=False,
          device=args.device, indices=parse_range(args.indices, len(seeds)), block=args.block)


if __name__ == "__main__":
    main()
