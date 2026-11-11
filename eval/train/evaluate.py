"""Visible train-split evaluator for radiance-field uncertainty and active view selection.

Scores methods on the public development instances in ``data/train_assets/``
with exactly the same protocol, metrics and baselines as the hidden test.
Nothing is imported from ``method/`` or ``data/``.

Methods are discovered as ``<methods_dir>/<name>/predict.json`` files holding
``{"command": [...argv...]}``. Each command is run from the workspace root as

    <command> --round 1 --input <dir> --output <dir>
    <command> --round 2 --input <dir> --output <dir>

Round 1 gives renderings of the sparse and dense fields; the method returns
pixel uncertainty for the target views of both fields and one score per
candidate view. The evaluator acquires the best-scored candidate, and round 2
gives renderings of the field after continued training with that view; the
method scores the remaining candidates and the evaluator acquires the best
one. Final held-out quality of the acquired pair comes from the frozen
retraining table in ``answers.npz``.

A method that crashes, times out, or returns a missing, misshapen or
non-finite array fails the whole run and receives the worst achievable value
of every metric in every experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
ASSETS = WORKSPACE / "data" / "train_assets"
SLICE = "train"
DEFAULT_SCORES = WORKSPACE / "scoring" / "scores_train.json"
EXPECTED_SCENES = 16
EXPECTED_COUNTS = {"source": 3, "dense_source": 8, "target": 6, "candidate": 8, "eval": 8, "acquisitions": 2}
TIMEOUT = int(os.environ.get("GYM_METHOD_TIMEOUT", "1800"))
DEPTH_EXPERIMENTS = {"depth_uncertainty_sparse": "sparse", "depth_uncertainty_dense": "dense"}
ACTIVE = "active_view_selection"
BASELINES = {"random_uncertainty", "random_view", "farthest_view"}
NO_METHOD = "no_method_registered"
ROUND2_ONLY = ("step2_source_rgb", "step2_source_depth", "step2_candidate_rgb", "step2_candidate_depth")
GOLDEN = 0.6180339887498949


# ----------------------------------------------------------------------------- metrics

def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(x)))


def ause(uncertainty: np.ndarray, error: np.ndarray, tiebreak: np.ndarray) -> float:
    """Area under the sparsification error curve (BayesRays / WarpRF protocol).

    Pixels are removed in decreasing order of predicted uncertainty (ties broken
    by a fixed evaluator-owned key) and the mean absolute depth error of the
    remaining pixels is compared with the oracle curve that removes pixels by
    true error, over removal fractions 0, 0.01, ..., 0.99. Both curves are
    divided by the view's full-set mean error, a constant that does not depend
    on the prediction, so a better ordering always gives a lower AUSE. (The
    released code divides by the larger curve's maximum, which depends on the
    prediction and lets an anti-correlated ordering score below a random one.)
    """
    n = error.size
    ratios = np.linspace(0.0, 1.0, 100, endpoint=False)
    keep = np.maximum(np.array([int((1.0 - r) * n) for r in ratios]), 1)
    oracle = np.cumsum(np.sort(error))[keep - 1] / keep
    order = np.lexsort((tiebreak, uncertainty))
    by_unc = np.cumsum(error[order])[keep - 1] / keep
    scale = float(np.mean(error))
    if scale <= 0.0:
        return 0.0
    return _trapezoid((by_unc - oracle) / scale, ratios)


def _rankdata(a: np.ndarray) -> np.ndarray:
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.int64)
    inv[sorter] = np.arange(sorter.size)
    s = a[sorter]
    obs = np.r_[True, s[1:] != s[:-1]]
    dense = np.cumsum(obs)[inv]
    count = np.r_[np.nonzero(obs)[0], obs.size]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rankdata(a.astype(np.float64))
    rb = _rankdata(b.astype(np.float64))
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    return float(np.sum(ra * rb) / denom) if denom > 0 else 0.0


# ----------------------------------------------------------------------------- suite

class Scene:
    def __init__(self, name: str, inputs: dict, answers: dict):
        self.name = name
        self.inputs = inputs
        self.answers = answers
        gt = answers["target_depth_gt"].astype(np.float64)
        self.valid = np.isfinite(gt)
        self.h, self.w = gt.shape[1:]
        self.errors, self.tiebreak, self.worst_ause = {}, [], {}
        n_pix = self.h * self.w
        for t in range(gt.shape[0]):
            m = int(self.valid[t].sum())
            self.tiebreak.append(np.mod((np.arange(m) + 1.0) * GOLDEN + 0.37 * t, 1.0))
        for setting in ("sparse", "dense"):
            rendered = inputs[f"{setting}_target_depth"].astype(np.float64)
            errs = [np.abs(rendered[t][self.valid[t]] - gt[t][self.valid[t]]) for t in range(gt.shape[0])]
            self.errors[setting] = errs
            self.worst_ause[setting] = [ause(-e, e, tb) for e, tb in zip(errs, self.tiebreak)]
        self.random_uncertainty = [
            np.mod(np.arange(n_pix, dtype=np.float64) * 0.7548776662466927 + 0.5698402909980532 * (t + 1), 1.0).reshape(self.h, self.w)
            for t in range(gt.shape[0])
        ]
        off = ~np.eye(answers["final_psnr"].shape[0], dtype=bool)
        self.table = {m: answers[f"final_{m}"].astype(np.float64) for m in ("psnr", "ssim", "depth_mae")}
        self.off = off
        centers = inputs["candidate_poses"][:, :3, 3].astype(np.float64)
        source_centers = inputs["sparse_source_poses"][:, :3, 3].astype(np.float64)
        self.farthest = _farthest_sequence(centers, source_centers)


def _farthest_sequence(candidates: np.ndarray, sources: np.ndarray) -> tuple[int, int]:
    chosen = []
    current = [s for s in sources]
    for _ in range(EXPECTED_COUNTS["acquisitions"]):
        dist = np.array([min(np.linalg.norm(c - s) for s in current) for c in candidates])
        dist[chosen] = -np.inf
        pick = int(np.argmax(dist))
        chosen.append(pick)
        current.append(candidates[pick])
    return chosen[0], chosen[1]


def load_suite(assets: Path = ASSETS, expected_scenes: int = EXPECTED_SCENES) -> list[Scene]:
    manifest_path = assets / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"evaluator assets missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["scenes"]) != expected_scenes or manifest["counts"] != EXPECTED_COUNTS:
        raise SystemExit("evaluator assets do not match the fixed train protocol")
    scenes = []
    for name in manifest["scenes"]:
        with np.load(assets / name / "inputs.npz", allow_pickle=False) as f:
            inputs = {k: f[k] for k in f.files}
        with np.load(assets / name / "answers.npz", allow_pickle=False) as f:
            answers = {k: f[k] for k in f.files}
        scenes.append(Scene(name, inputs, answers))
    return scenes


# ----------------------------------------------------------------------------- scoring

def depth_view_scores(scene: Scene, setting: str, uncertainty: np.ndarray) -> list[tuple[float, float]]:
    out = []
    for t, err in enumerate(scene.errors[setting]):
        u = uncertainty[t][scene.valid[t]].astype(np.float64)
        out.append((ause(u, err, scene.tiebreak[t]), spearman(u, err)))
    return out


def active_scores(scene: Scene, pair: tuple[int, int]) -> dict:
    a, b = pair
    return {m: float(scene.table[m][a, b]) for m in scene.table}


def _aggregate(per_scene_depth: dict, per_scene_active: list[dict]) -> dict:
    exps = {}
    for exp, setting in DEPTH_EXPERIMENTS.items():
        rows = [r for scene_rows in per_scene_depth[setting] for r in scene_rows]
        exps[exp] = {"ause": float(np.mean([r[0] for r in rows])), "spearman_depth_error": float(np.mean([r[1] for r in rows]))}
    exps[ACTIVE] = {
        "final_psnr": float(np.mean([r["psnr"] for r in per_scene_active])),
        "final_ssim": float(np.mean([r["ssim"] for r in per_scene_active])),
        "final_depth_mae": float(np.mean([r["depth_mae"] for r in per_scene_active])),
    }
    return exps


def worst_values(scenes: list[Scene]) -> dict:
    depth = {s: [[(w, -1.0) for w in sc.worst_ause[s]] for sc in scenes] for s in ("sparse", "dense")}
    active = [
        {
            "psnr": float(np.min(sc.table["psnr"][sc.off])),
            "ssim": float(np.min(sc.table["ssim"][sc.off])),
            "depth_mae": float(np.max(sc.table["depth_mae"][sc.off])),
        }
        for sc in scenes
    ]
    return _aggregate(depth, active), depth, active


def baselines(scenes: list[Scene]) -> tuple[dict, dict]:
    depth_rand = {s: [depth_view_scores(sc, s, sc.random_uncertainty) for sc in scenes] for s in ("sparse", "dense")}
    rand_active = [{m: float(np.mean(sc.table[m][sc.off])) for m in sc.table} for sc in scenes]
    far_active = [active_scores(sc, sc.farthest) for sc in scenes]
    agg_rand = _aggregate(depth_rand, rand_active)
    agg_far = _aggregate(depth_rand, far_active)
    results = {exp: {"random_uncertainty": agg_rand[exp]} for exp in DEPTH_EXPERIMENTS}
    results[ACTIVE] = {"random_view": agg_rand[ACTIVE], "farthest_view": agg_far[ACTIVE]}
    details = {
        "random_uncertainty": {s: depth_rand[s] for s in depth_rand},
        "random_view": rand_active,
        "farthest_view": {"active": far_active, "choices": [list(sc.farthest) for sc in scenes]},
    }
    return results, details


# ----------------------------------------------------------------------------- method protocol

class MethodFailure(Exception):
    pass


def discover_methods(methods_dir: Path) -> list[tuple[str, list[str] | None, str]]:
    found = []
    taken = set(BASELINES) | {NO_METHOD}
    for spec_path in sorted(methods_dir.glob("*/predict.json")) if methods_dir.is_dir() else []:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec_path.parent.name).strip("_") or "method"
        while name in taken:
            name = "method_" + name
        taken.add(name)
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            cmd = spec["command"]
            if not (isinstance(cmd, list) and cmd and all(isinstance(x, str) for x in cmd)):
                raise ValueError("command must be a non-empty list of strings")
            found.append((name, cmd, ""))
        except Exception as exc:  # noqa: BLE001 - any malformed registration is a failed run
            found.append((name, None, f"invalid predict.json: {exc}"))
    return found


def _invoke(cmd: list[str], rnd: int, inp: Path, out: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WORKSPACE) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    out.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            cmd + ["--round", str(rnd), "--input", str(inp), "--output", str(out)],
            cwd=str(WORKSPACE), env=env, capture_output=True, text=True, timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise MethodFailure(f"round {rnd} timed out after {TIMEOUT}s") from exc
    except OSError as exc:
        raise MethodFailure(f"round {rnd} could not start: {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise MethodFailure(f"round {rnd} exited with code {proc.returncode}: {' | '.join(tail)}")


def _read_output(path: Path, spec: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise MethodFailure(f"missing output {path.name}")
    try:
        with np.load(path, allow_pickle=False) as f:
            arrays = {k: np.asarray(f[k]) for k in spec if k in f.files}
    except Exception as exc:  # noqa: BLE001
        raise MethodFailure(f"unreadable output {path.name}: {exc}") from exc
    for key, shape in spec.items():
        if key not in arrays:
            raise MethodFailure(f"{path.name}: missing array '{key}'")
        arr = arrays[key]
        if arr.dtype.kind not in "fiu" or arr.shape != shape:
            raise MethodFailure(f"{path.name}: '{key}' must be a real array of shape {shape}, got {arr.dtype} {arr.shape}")
        arr = arr.astype(np.float64)
        if not np.all(np.isfinite(arr)):
            raise MethodFailure(f"{path.name}: '{key}' contains non-finite values")
        arrays[key] = arr
    return arrays


def _write_inputs(root: Path, rnd: int, payloads: dict[str, dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    first = next(iter(payloads.values()))
    manifest = {
        "round": rnd,
        "scenes": list(payloads),
        "counts": EXPECTED_COUNTS,
        "image_height": int(first["intrinsics"][1, 2] * 2),
        "image_width": int(first["intrinsics"][0, 2] * 2),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, arrays in payloads.items():
        np.savez(root / f"{name}.npz", **arrays)


def _round2_payload(scene: Scene, first: int) -> dict:
    x, a = scene.inputs, scene.answers
    available = np.ones(EXPECTED_COUNTS["candidate"], dtype=bool)
    available[first] = False
    return {
        "intrinsics": x["intrinsics"],
        "selected_candidates": np.array([first], dtype=np.int64),
        "candidate_available": available,
        "source_poses": np.concatenate([x["sparse_source_poses"], x["candidate_poses"][first : first + 1]]),
        "source_images": np.concatenate([x["sparse_source_images"], a["candidate_images"][first : first + 1]]),
        "source_rgb": x["step2_source_rgb"][first],
        "source_depth": x["step2_source_depth"][first],
        "candidate_poses": x["candidate_poses"],
        "candidate_rgb": x["step2_candidate_rgb"][first],
        "candidate_depth": x["step2_candidate_depth"][first],
    }


def run_method(cmd: list[str], scenes: list[Scene]) -> tuple[dict, list[tuple[int, int]]]:
    c, t = EXPECTED_COUNTS["candidate"], EXPECTED_COUNTS["target"]
    with tempfile.TemporaryDirectory(prefix="warprf_eval_") as tmp:
        tmp = Path(tmp)
        r1 = {sc.name: {k: v for k, v in sc.inputs.items() if k not in ROUND2_ONLY} for sc in scenes}
        _write_inputs(tmp / "r1_in", 1, r1)
        _invoke(cmd, 1, tmp / "r1_in", tmp / "r1_out")
        outs1 = {}
        for sc in scenes:
            hw = (t, sc.h, sc.w)
            outs1[sc.name] = _read_output(
                tmp / "r1_out" / f"{sc.name}.npz",
                {"sparse_uncertainty": hw, "dense_uncertainty": hw, "candidate_scores": (c,)},
            )
        firsts = {sc.name: int(np.argmax(outs1[sc.name]["candidate_scores"])) for sc in scenes}
        _write_inputs(tmp / "r2_in", 2, {sc.name: _round2_payload(sc, firsts[sc.name]) for sc in scenes})
        _invoke(cmd, 2, tmp / "r2_in", tmp / "r2_out")
        pairs = []
        for sc in scenes:
            scores = _read_output(tmp / "r2_out" / f"{sc.name}.npz", {"candidate_scores": (c,)})["candidate_scores"]
            scores = scores.copy()
            scores[firsts[sc.name]] = -np.inf
            pairs.append((firsts[sc.name], int(np.argmax(scores))))
    depth = {
        s: [depth_view_scores(sc, s, outs1[sc.name][f"{s}_uncertainty"]) for sc in scenes] for s in ("sparse", "dense")
    }
    active = [active_scores(sc, p) for sc, p in zip(scenes, pairs)]
    return _aggregate(depth, active), {"depth": depth, "active": active, "choices": [list(p) for p in pairs]}


# ----------------------------------------------------------------------------- main

def evaluate(methods_dir: Path, scores_path: Path, details_path: Path | None,
             assets: Path = ASSETS, expected_scenes: int = EXPECTED_SCENES) -> dict:
    if scores_path.exists():
        scores_path.unlink()
    scenes = load_suite(assets, expected_scenes)
    base_results, base_details = baselines(scenes)
    worst, worst_depth, worst_active = worst_values(scenes)
    experiments = {exp: {"results": dict(rows)} for exp, rows in base_results.items()}
    status, details = {}, {"baselines": base_details}
    methods = discover_methods(methods_dir)
    if not methods:
        methods = [(NO_METHOD, None, f"no method registered under {methods_dir}/<name>/predict.json")]
    for name, cmd, error in methods:
        try:
            if cmd is None:
                raise MethodFailure(error)
            agg, det = run_method(cmd, scenes)
            status[name] = {"status": "ok"}
            details[name] = det
        except MethodFailure as exc:
            agg = worst
            status[name] = {"status": "failed", "reason": str(exc)}
            details[name] = {"failed": str(exc)}
            print(f"[evaluate:{SLICE}] {name}: FAILED - {exc}", file=sys.stderr, flush=True)
        for exp, metrics in agg.items():
            experiments[exp]["results"][name] = metrics
    out = {"slice": SLICE, "num_scenes": len(scenes), "experiments": experiments, "_methods": status}
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = scores_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(scores_path)
    if details_path is not None:
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details["_worst"] = {"depth": worst_depth, "active": worst_active}
        details_path.write_text(json.dumps(details) + "\n", encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods-dir", default=os.environ.get("GYM_METHODS_DIR", str(WORKSPACE / "outputs")))
    p.add_argument("--scores", default=str(DEFAULT_SCORES))
    p.add_argument("--details", default="")
    p.add_argument("--assets", default=str(ASSETS), help="audit/smoke use only")
    p.add_argument("--expected-scenes", type=int, default=EXPECTED_SCENES, help="audit/smoke use only")
    args = p.parse_args()
    out = evaluate(Path(args.methods_dir), Path(args.scores), Path(args.details) if args.details else None,
                   Path(args.assets), args.expected_scenes)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
