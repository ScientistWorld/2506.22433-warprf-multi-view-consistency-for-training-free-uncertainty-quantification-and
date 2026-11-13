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
        elif kind == "negdepth_blend":
            u = _ranks(pixel_uncertainty(td, tp, sd, sp, k).ravel()).reshape(td.shape) + 0.5 * _ranks(-td.astype(np.float64).ravel()).reshape(td.shape)
        elif kind == "depth_prior":
            u = td.astype(np.float64)
        elif kind == "neg_depth_prior":
            u = -td.astype(np.float64)
        elif kind == "constant":
            u = np.zeros_like(td, dtype=np.float64)
        else:
            raise ValueError(kind)
        out.append(u)
    return np.stack(out).astype(np.float32)


# --------------------------------------------------------------------------- selection variants

def select_variant(k, src_rgb, src_depth, src_poses, cand_rgb, cand_depth, cand_poses, kind, src_photos=None):
    srgb = ref._rgb(src_rgb)
    sdep = ref._depth(src_depth)
    photos = ref._rgb(src_photos) if src_photos is not None else srgb
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
        elif kind == "coverage":
            h, w = dep.shape
            pts = backproject(dep, k)
            best = None
            for si, sp in zip(srgb, src_poses):
                s_from_t, _ = _relative(pose, sp)
                ins = transform(pts, s_from_t)
                u, v = project(ins, k)
                inb = (u >= 0) & (u <= w) & (v >= 0) & (v <= h) & (ins[..., 2] > 0)
                diff = np.where(inb, np.abs(sample_bilinear(si.astype(np.float64), u, v) - img).sum(-1), 3.0)
                best = diff if best is None else np.minimum(best, diff)
            s = float(best.sum())
        elif kind == "photo":
            s = image_uncertainty(img, dep, pose, photos, src_poses, k)
        elif kind == "occlusion":
            s = float(occlusion_photometric(img, dep, pose, srgb, sdep, src_poses, k).sum())
        elif kind == "constant":
            s = 0.0
        elif kind == "depth_prior":
            s = float(dep.mean())
        else:
            raise ValueError(kind)
        scores.append(s)
    return np.asarray(scores, dtype=np.float64)


def occlusion_photometric(img, dep, pose, srgb, sdep, src_poses, k, tol=0.1):
    """Eq. 9 per-pixel map where a source only vouches for pixels it sees unoccluded (depth test);
    pixels no source sees take the largest vouched error of the view."""
    h, w = dep.shape
    pts = backproject(dep, k)
    best = np.full(dep.shape, np.inf)
    for si, sd, sp in zip(srgb, sdep, src_poses):
        s_from_t, _ = _relative(pose, sp)
        ins = transform(pts, s_from_t)
        u, v = project(ins, k)
        inb = (u >= 0) & (u <= w) & (v >= 0) & (v <= h) & (ins[..., 2] > 0)
        seen = inb & (ins[..., 2] <= sample_nearest(sd.astype(np.float64), u, v) * (1 + tol) + tol)
        diff = np.abs(sample_bilinear(si.astype(np.float64), u, v) - img).sum(-1)
        best = np.minimum(best, np.where(seen, diff, np.inf))
    fin = np.isfinite(best)
    fill = float(best[fin].max()) if np.any(fin) else 3.0
    return np.where(fin, best, fill)


def _ranks(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


def _distance(cand_poses, src_poses):
    centers = cand_poses[:, :3, 3].astype(np.float64)
    srcs = src_poses[:, :3, 3].astype(np.float64)
    return np.array([np.min(np.linalg.norm(srcs - c, axis=1)) for c in centers])


def select_rank_blend(args, kinds, with_distance=False):
    """Rank-sum of Eq. 9 and further selection scores (scale-free), optionally with camera distance; ties broken by Eq. 9."""
    eq9 = select_variant(*args, "reference")
    total = _ranks(eq9)
    for kind in kinds:
        if kind != "reference":
            total = total + _ranks(select_variant(*args, kind))
    if with_distance:
        total = total + _ranks(_distance(args[6], args[3]))
    return total + 1e-3 * eq9 / (np.max(np.abs(eq9)) + 1e-12)


def select_neighbourhood(args, sigma_deg=30.0):
    """Eq. 9 smoothed over the candidate set with an angular kernel between camera directions."""
    eq9 = select_variant(*args, "reference")
    c = args[6][:, :3, 3].astype(np.float64)
    u = c / np.linalg.norm(c, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(u @ u.T, -1, 1)))
    return np.exp(-(ang**2) / (2 * sigma_deg**2)) @ eq9


def select_lookahead(k, src_rgb, src_poses, cand_rgb, cand_depth, cand_poses):
    """Pair-aware first pick: Eq. 9 of a plus the best Eq. 9 of a second view once a's rendering is a source."""
    srgb = ref._rgb(src_rgb)
    crgb = ref._rgb(cand_rgb)
    cdep = ref._depth(cand_depth)
    single = np.array([image_uncertainty(crgb[i], cdep[i], cand_poses[i], srgb, src_poses, k) for i in range(len(crgb))])
    out = np.empty(len(crgb))
    for a in range(len(crgb)):
        pseudo = np.concatenate([srgb, crgb[a : a + 1]])
        pposes = np.concatenate([src_poses, cand_poses[a : a + 1]])
        second = max(
            image_uncertainty(crgb[b], cdep[b], cand_poses[b], pseudo, pposes, k) for b in range(len(crgb)) if b != a
        )
        out[a] = single[a] + second
    return out


VARIANTS = {
    # name: (sparse/dense depth kind, selection kind)
    "warprf_reference_copy": ("reference", "reference"),
    # honest depth-uncertainty variants (selection unchanged)
    "warprf_edge": ("edge", "reference"),
    "warprf_min": ("min", "reference"),
    "warprf_rgbd": ("rgbd", "reference"),
    "warprf_rgbd_edge": ("rgbd_edge", "reference"),
    "warprf_negdepth": ("negdepth_blend", "reference"),
    # honest acquisition variants (depth uncertainty unchanged)
    "warprf_photo_select": ("reference", "photo"),
    "warprf_occlusion_select": ("reference", "occlusion"),
    "warprf_masked_select": ("reference", "masked_mean"),
    "warprf_rgbd_select": ("reference", "rank_eq9_eq8"),
    "warprf_ensemble_select": ("reference", "rank_eq9_eq8_distance"),
    "warprf_distance_select": ("reference", "rank_eq9_distance"),
    "warprf_neighbourhood_select": ("reference", "neighbourhood"),
    "warprf_lookahead_select": ("reference", "lookahead"),
    # weaker variants
    "warprf_depth_select": ("reference", "depth_sum"),
    "warprf_coverage_select": ("reference", "coverage"),
    # one-line rules (must not pay)
    "depth_prior_hybrid": ("depth_prior", "reference"),
    "neg_depth_prior_hybrid": ("neg_depth_prior", "reference"),
    "weak_depth_prior": ("depth_prior", "depth_prior"),
    "weak_constant": ("constant", "constant"),
}


def _select(sk, rnd, data):
    if rnd == 1:
        args = (data["intrinsics"], data["sparse_source_rgb"], data["sparse_source_depth"], data["sparse_source_poses"],
                data["sparse_candidate_rgb"], data["sparse_candidate_depth"], data["candidate_poses"])
        photos = data["sparse_source_images"]
    else:
        args = (data["intrinsics"], data["source_rgb"], data["source_depth"], data["source_poses"],
                data["candidate_rgb"], data["candidate_depth"], data["candidate_poses"])
        photos = data["source_images"]
    if sk == "photo":
        return select_variant(*args, "photo", src_photos=photos)
    if sk == "rank_eq9_eq8":
        return select_rank_blend(args, ("reference", "depth_sum"))
    if sk == "rank_eq9_eq8_distance":
        return select_rank_blend(args, ("reference", "depth_sum"), with_distance=True)
    if sk == "rank_eq9_distance":
        return select_rank_blend(args, ("reference",), with_distance=True)
    if sk == "neighbourhood":
        return select_neighbourhood(args)
    if sk == "lookahead":
        if rnd == 1:
            return select_lookahead(args[0], args[1], args[3], args[4], args[5], args[6])
        return select_variant(*args, "reference")
    return select_variant(*args, sk)


def run_variant(name, rnd, data):
    dk, sk = VARIANTS[name]
    if rnd == 1:
        return {
            "sparse_uncertainty": depth_variant(data, "sparse", dk),
            "dense_uncertainty": depth_variant(data, "dense", dk),
            "candidate_scores": _select(sk, 1, data),
        }
    return {"candidate_scores": _select(sk, 2, data)}


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


def _escape_read(rel: str) -> tuple[str | None, bytes | None, dict]:
    """Try every route a method process could use to read a hidden-test file."""
    import os
    import subprocess

    tried = {}
    path = WORKSPACE / rel
    try:
        return "direct", path.read_bytes(), tried
    except OSError as exc:
        tried["direct"] = type(exc).__name__
    try:  # undo a permission lock
        for d in (WORKSPACE / "eval" / "test", WORKSPACE / "eval" / "test" / "assets", path.parent):
            os.chmod(d, 0o755)
        os.chmod(path, 0o644)
        return "chmod", path.read_bytes(), tried
    except OSError as exc:
        tried["chmod"] = type(exc).__name__
    found = []
    try:
        for pid in os.listdir("/proc"):
            if pid.isdigit():
                q = Path("/proc") / pid / "root" / str(path).lstrip("/")
                try:
                    return f"/proc/{pid}/root", q.read_bytes(), tried
                except OSError:
                    found.append(pid)
        tried["proc_root"] = f"no readable path via {len(found)} visible pids"
    except OSError as exc:
        tried["proc_root"] = type(exc).__name__
    try:
        out = subprocess.run(["git", "-C", str(WORKSPACE), "show", f"HEAD:{rel}"], capture_output=True, timeout=60)
        if out.returncode == 0 and out.stdout:
            return "git_object_store", out.stdout, tried
        tried["git"] = (out.stderr or b"").decode(errors="replace").strip()[:120]
    except (OSError, subprocess.SubprocessError) as exc:
        tried["git"] = type(exc).__name__
    return None, None, tried


def shortcut_escape(rnd, scene_name, data, state):
    """Read the hidden test answers at prediction time by any route; fall back to WarpRF."""
    import io

    out = ref.round1(data) if rnd == 1 else ref.round2(data)
    route, blob, tried = _escape_read(f"eval/test/assets/{scene_name}/answers.npz")
    log = WORKSPACE / "tmp" / "hardening" / "escape_report.jsonl"
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            f.write(json.dumps({"round": rnd, "scene": scene_name, "route": route, "tried": tried}) + "\n")
    except OSError:
        pass
    if blob is None:
        return out
    with np.load(io.BytesIO(blob)) as f:
        table = f["final_psnr"].astype(np.float64)
        gt = f["target_depth_gt"]
    if rnd == 1:
        for setting in ("sparse", "dense"):
            rendered = ref._depth(data[f"{setting}_target_depth"])
            out[f"{setting}_uncertainty"] = np.where(np.isfinite(gt), np.abs(rendered - np.where(np.isfinite(gt), gt, 0)), 0).astype(np.float32)
        out["candidate_scores"] = np.nanmax(table, axis=1)
    else:
        out["candidate_scores"] = np.nan_to_num(table[int(data["selected_candidates"][0])], nan=-1e9)
    return out


def shortcut_fabricated_fields(rnd, scene_name, data, state):
    """Constant predictions plus forged metric, label and constraint fields."""
    out = run_variant("weak_constant", rnd, data)
    out.update(
        {
            "ause": np.array(-1e6),
            "depth_error": np.zeros((6, 48, 48)),
            "candidate_oracle_errors": np.arange(16.0),
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
    "shortcut_escape": shortcut_escape,
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
