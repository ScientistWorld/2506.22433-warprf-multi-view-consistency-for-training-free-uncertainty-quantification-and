"""Variants and shortcut payloads used by the hardening audit (not part of the gym).

Each variant speaks the benchmark's prediction interface:
    python -m redteam.hardening.variants --variant NAME --round {1,2} --input DIR --output DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from method import predict as ref
from method.warprf import (
    _relative,
    backproject,
    image_uncertainty,
    pixel_uncertainty,
    project,
    sample_bilinear,
    sample_nearest,
    transform,
)

WORKSPACE = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- building blocks

def masked_pixel_uncertainty(target_depth, target_pose, source_depths, source_poses, k, reduce="mean"):
    """Eq. 8 with in-bounds masking; pixels seen by no source get the view's 95th percentile."""
    h, w = target_depth.shape
    pts = backproject(target_depth, k)
    vals = []
    for sd, sp in zip(source_depths, source_poses):
        s_from_t, t_from_s = _relative(target_pose, sp)
        ins = transform(pts, s_from_t)
        u, v = project(ins, k)
        inb = (u >= 0) & (u <= w) & (v >= 0) & (v <= h) & (ins[..., 2] > 0)
        corr = sample_nearest(transform(backproject(sd, k), t_from_s), u, v)
        vals.append(np.where(inb, np.abs(corr[..., 2] - pts[..., 2]), np.nan))
    stack = np.stack(vals)
    with np.errstate(all="ignore"):
        agg = np.nanmean(stack, axis=0) if reduce == "mean" else np.nanmin(stack, axis=0)
    finite = np.isfinite(agg)
    fill = float(np.percentile(agg[finite], 95)) if np.any(finite) else 1.0
    return np.where(finite, agg, fill)


def photometric_map(target_image, target_depth, target_pose, source_images, source_poses, k):
    """Per-pixel minimum photometric reprojection error (the per-pixel map inside Eq. 9)."""
    pts = backproject(target_depth, k)
    best = None
    for si, sp in zip(source_images, source_poses):
        s_from_t, _ = _relative(target_pose, sp)
        u, v = project(transform(pts, s_from_t), k)
        diff = np.abs(sample_bilinear(si.astype(np.float64), u, v) - target_image).sum(axis=-1)
        best = diff if best is None else np.minimum(best, diff)
    return best


def _norm(x):
    return x / (np.median(x) + 1e-6)


def depth_edges(depth):
    gy, gx = np.gradient(depth.astype(np.float64))
    return np.hypot(gx, gy)


# --------------------------------------------------------------------------- depth variants

def depth_variant(data, setting, kind):
    k = data["intrinsics"]
    sd = ref._depth(data[f"{setting}_source_depth"])
    sp = data[f"{setting}_source_poses"]
    srgb = ref._rgb(data[f"{setting}_source_rgb"])
    out = []
    for t, (td, tp) in enumerate(zip(ref._depth(data[f"{setting}_target_depth"]), data["target_poses"])):
        trgb = ref._rgb(data[f"{setting}_target_rgb"][t])
        if kind == "reference":
            u = pixel_uncertainty(td, tp, sd, sp, k)
        elif kind == "masked":
            u = masked_pixel_uncertainty(td, tp, sd, sp, k, "mean")
        elif kind == "min":
            u = masked_pixel_uncertainty(td, tp, sd, sp, k, "min")
        elif kind == "rgbd":
            u = _norm(pixel_uncertainty(td, tp, sd, sp, k)) + 0.5 * _norm(photometric_map(trgb, td, tp, srgb, sp, k))
        elif kind == "edge":
            u = _norm(pixel_uncertainty(td, tp, sd, sp, k)) + 0.5 * _norm(depth_edges(td))
        elif kind == "rgbd_edge":
            u = (
                _norm(pixel_uncertainty(td, tp, sd, sp, k))
                + 0.5 * _norm(photometric_map(trgb, td, tp, srgb, sp, k))
                + 0.5 * _norm(depth_edges(td))
            )
        elif kind == "depth_prior":
            u = td.astype(np.float64)
        elif kind == "constant":
            u = np.zeros_like(td, dtype=np.float64)
        else:
            raise ValueError(kind)
        out.append(u)
    return np.stack(out).astype(np.float32)


# --------------------------------------------------------------------------- selection variants

def select_variant(k, src_rgb, src_depth, src_poses, cand_rgb, cand_depth, cand_poses, kind):
    srgb = ref._rgb(src_rgb)
    sdep = ref._depth(src_depth)
    scores = []
    for img, dep, pose in zip(ref._rgb(cand_rgb), ref._depth(cand_depth), cand_poses):
        if kind == "reference":
            s = image_uncertainty(img, dep, pose, srgb, src_poses, k)
        elif kind == "masked_mean":
            h, w = dep.shape
            pts = backproject(dep, k)
            best = np.full(dep.shape, np.inf)
            for si, sp in zip(srgb, src_poses):
                s_from_t, _ = _relative(pose, sp)
                ins = transform(pts, s_from_t)
                u, v = project(ins, k)
                inb = (u >= 0) & (u <= w) & (v >= 0) & (v <= h) & (ins[..., 2] > 0)
                diff = np.abs(sample_bilinear(si.astype(np.float64), u, v) - img).sum(-1)
                best = np.minimum(best, np.where(inb, diff, np.inf))
            fin = np.isfinite(best)
            s = float(best[fin].mean()) if np.any(fin) else 0.0
        elif kind == "depth_sum":
            s = float(pixel_uncertainty(dep, pose, sdep, src_poses, k).sum())
        elif kind == "rgbd_sum":
            pu = pixel_uncertainty(dep, pose, sdep, src_poses, k)
            s = float(image_uncertainty(img, dep, pose, srgb, src_poses, k)) / dep.size + float(np.mean(pu / (np.median(pu) + 1e-6))) * 0.05
        elif kind == "constant":
            s = 0.0
        elif kind == "depth_prior":
            s = float(dep.mean())
        else:
            raise ValueError(kind)
        scores.append(s)
    return np.asarray(scores, dtype=np.float64)


VARIANTS = {
    # name: (sparse/dense depth kind, selection kind)
    "warprf_reference_copy": ("reference", "reference"),
    "warprf_masked": ("masked", "masked_mean"),
    "warprf_min": ("min", "reference"),
    "warprf_rgbd": ("rgbd", "reference"),
    "warprf_edge": ("edge", "reference"),
    "warprf_rgbd_edge": ("rgbd_edge", "reference"),
    "warprf_depth_select": ("reference", "depth_sum"),
    "warprf_rgbd_select": ("reference", "rgbd_sum"),
    "weak_depth_prior": ("depth_prior", "depth_prior"),
    "weak_constant": ("constant", "constant"),
}


def run_variant(name, rnd, data):
    dk, sk = VARIANTS[name]
    if rnd == 1:
        return {
            "sparse_uncertainty": depth_variant(data, "sparse", dk),
            "dense_uncertainty": depth_variant(data, "dense", dk),
            "candidate_scores": select_variant(
                data["intrinsics"], data["sparse_source_rgb"], data["sparse_source_depth"], data["sparse_source_poses"],
                data["sparse_candidate_rgb"], data["sparse_candidate_depth"], data["candidate_poses"], sk,
            ),
        }
    return {
        "candidate_scores": select_variant(
            data["intrinsics"], data["source_rgb"], data["source_depth"], data["source_poses"],
            data["candidate_rgb"], data["candidate_depth"], data["candidate_poses"], sk,
        )
    }


# --------------------------------------------------------------------------- shortcut payloads

def _match_public_seed(data, max_seed=2000):
    """Look for a public generator seed whose camera rig matches this instance."""
    from data.scenes import make_scene

    h = int(data["intrinsics"][1, 2] * 2)
    w = int(data["intrinsics"][0, 2] * 2)
    for seed in range(max_seed):
        sc = make_scene(seed, w, h)
        if np.allclose(sc["poses"]["candidate"], data["candidate_poses"], atol=1e-4):
            return sc
    return None


def shortcut_generator_oracle(rnd, scene_name, data, state):
    """Regenerate answers with the public generator; fall back to WarpRF when that fails."""
    from data.scenes import render_view

    sc = state.get(scene_name, "unset")
    if isinstance(sc, str):
        sc = _match_public_seed(data) if rnd == 1 else None
        state[scene_name] = sc
    if rnd == 1:
        out = ref.round1(data)
        if sc is not None:
            for setting in ("sparse", "dense"):
                rendered = ref._depth(data[f"{setting}_target_depth"])
                gt = np.stack([render_view(sc, p)[1] for p in data["target_poses"]])
                out[f"{setting}_uncertainty"] = np.where(np.isfinite(gt), np.abs(rendered - np.where(np.isfinite(gt), gt, 0)), 0).astype(np.float32)
            table = _train_table(sc["seed"])
            if table is not None:
                out["candidate_scores"] = np.nanmax(table, axis=1)
        return out
    out = ref.round2(data)
    seed = state.get(scene_name + "#seed")
    table = _train_table(seed) if seed is not None else None
    if table is not None:
        out["candidate_scores"] = np.nan_to_num(table[int(data["selected_candidates"][0])], nan=-1e9)
    return out


def _train_table(seed):
    if seed is None:
        return None
    for meta in sorted((WORKSPACE / "data" / "train_assets").glob("*/meta.json")):
        info = json.loads(meta.read_text())
        if info.get("seed") == seed:
            with np.load(meta.parent / "answers.npz") as f:
                return f["final_psnr"].astype(np.float64)
    return None


def shortcut_read_hidden_answers(rnd, scene_name, data, state):
    """Read the hidden test answers straight from disk (left open: needs filesystem isolation)."""
    out = ref.round1(data) if rnd == 1 else ref.round2(data)
    for root in (WORKSPACE / "eval" / "test" / "assets", WORKSPACE / "data" / "train_assets"):
        path = root / scene_name / "answers.npz"
        if path.is_file():
            with np.load(path) as f:
                table = f["final_psnr"].astype(np.float64)
                gt = f["target_depth_gt"]
            if rnd == 1:
                for setting in ("sparse", "dense"):
                    rendered = ref._depth(data[f"{setting}_target_depth"])
                    out[f"{setting}_uncertainty"] = np.where(np.isfinite(gt), np.abs(rendered - np.where(np.isfinite(gt), gt, 0)), 0).astype(np.float32)
                out["candidate_scores"] = np.nanmax(table, axis=1)
            else:
                out["candidate_scores"] = np.nan_to_num(table[int(data["selected_candidates"][0])], nan=-1e9)
            break
    return out


def shortcut_fabricated_fields(rnd, scene_name, data, state):
    """Constant predictions plus forged metric, label and constraint fields."""
    out = run_variant("weak_constant", rnd, data)
    out.update(
        {
            "ause": np.array(-1e6),
            "depth_error": np.zeros((6, 48, 48)),
            "candidate_oracle_errors": np.arange(8.0),
            "final_psnr": np.array(1e6),
            "selected_index": np.array(-999999),
            "used_ground_truth_for_uncertainty": np.array(-100),
            "source_view_count": np.array(-100),
        }
    )
    return out


def shortcut_nonfinite(rnd, scene_name, data, state):
    out = ref.round1(data) if rnd == 1 else ref.round2(data)
    if rnd == 1:
        out["sparse_uncertainty"] = np.full_like(out["sparse_uncertainty"], np.nan)
    return out


def shortcut_wrong_shape(rnd, scene_name, data, state):
    out = ref.round1(data) if rnd == 1 else ref.round2(data)
    out["candidate_scores"] = np.append(out["candidate_scores"], 1e9)
    return out


def shortcut_crash(rnd, scene_name, data, state):
    if rnd == 2:
        raise SystemExit(1)
    return ref.round1(data)


SHORTCUTS = {
    "shortcut_generator_oracle": shortcut_generator_oracle,
    "shortcut_read_hidden_answers": shortcut_read_hidden_answers,
    "shortcut_fabricated_fields": shortcut_fabricated_fields,
    "shortcut_nonfinite": shortcut_nonfinite,
    "shortcut_wrong_shape": shortcut_wrong_shape,
    "shortcut_crash": shortcut_crash,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--round", type=int, choices=[1, 2], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((inp / "manifest.json").read_text())
    state_path = out.parent / f".{args.variant}_state.json"
    state = {}
    if args.round == 2 and state_path.is_file():
        state = json.loads(state_path.read_text())
    for scene in manifest["scenes"]:
        with np.load(inp / f"{scene}.npz") as f:
            data = {k: f[k] for k in f.files}
        if args.variant in VARIANTS:
            arrays = run_variant(args.variant, args.round, data)
        else:
            fn = SHORTCUTS[args.variant]
            arrays = fn(args.round, scene, data, state)
            if args.round == 1 and args.variant == "shortcut_generator_oracle":
                sc = state.get(scene)
                state[scene + "#seed"] = None if sc is None else int(sc["seed"])
        np.savez(out / f"{scene}.npz", **arrays)
    if args.round == 1:
        state_path.write_text(json.dumps({k: v for k, v in state.items() if k.endswith("#seed")}))


if __name__ == "__main__":
    main()
