"""WarpRF predictor for the benchmark's two-round prediction interface.

Round 1 (one file per scene): pixel uncertainty for the held-out target views
of the sparse and the dense radiance field (Eq. 8), and an image-level score
for every candidate next view of the sparse field (Eq. 9).
Round 2: the evaluator acquires the round-1 choice, continues training the
field, and asks again for candidate scores from the updated renderings.

Usage: python -m method.predict --round {1,2} --input DIR --output DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from method.warprf import image_uncertainty, pixel_uncertainty


def _rgb(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 255.0


def _depth(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32)


def depth_uncertainty(data, setting: str) -> np.ndarray:
    k = data["intrinsics"]
    src_depth = _depth(data[f"{setting}_source_depth"])
    src_poses = data[f"{setting}_source_poses"]
    out = [
        pixel_uncertainty(d, pose, src_depth, src_poses, k)
        for d, pose in zip(_depth(data[f"{setting}_target_depth"]), data["target_poses"])
    ]
    return np.stack(out).astype(np.float32)


def candidate_scores(k, source_rgb, source_poses, candidate_rgb, candidate_depth, candidate_poses) -> np.ndarray:
    src = _rgb(source_rgb)
    return np.array(
        [
            image_uncertainty(img, dep, pose, src, source_poses, k)
            for img, dep, pose in zip(_rgb(candidate_rgb), _depth(candidate_depth), candidate_poses)
        ],
        dtype=np.float64,
    )


def round1(data) -> dict:
    return {
        "sparse_uncertainty": depth_uncertainty(data, "sparse"),
        "dense_uncertainty": depth_uncertainty(data, "dense"),
        "candidate_scores": candidate_scores(
            data["intrinsics"], data["sparse_source_rgb"], data["sparse_source_poses"],
            data["sparse_candidate_rgb"], data["sparse_candidate_depth"], data["candidate_poses"],
        ),
    }


def round2(data) -> dict:
    return {
        "candidate_scores": candidate_scores(
            data["intrinsics"], data["source_rgb"], data["source_poses"],
            data["candidate_rgb"], data["candidate_depth"], data["candidate_poses"],
        )
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, choices=[1, 2], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((inp / "manifest.json").read_text(encoding="utf-8"))
    fn = round1 if args.round == 1 else round2
    for scene in manifest["scenes"]:
        with np.load(inp / f"{scene}.npz") as data:
            np.savez(out / f"{scene}.npz", **fn(data))


if __name__ == "__main__":
    main()
