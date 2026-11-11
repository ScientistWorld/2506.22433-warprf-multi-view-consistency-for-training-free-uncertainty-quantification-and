"""Build frozen benchmark instances (renderings + answers) for a list of scenes.

For every scene this trains, with one fixed backbone recipe and identical
initialisation:

* ``M0``      - sparse model on the 3 initial views (steps 0-1500)
* ``D``       - dense model on 8 views (steps 0-3000)
* ``M1[c]``   - M0 continued after acquiring candidate c (steps 1500-2000)
* ``M2[a,b]`` - M1[a] continued after acquiring candidate b (steps 2000-3000)

and stores the renderings a method needs (inputs) separately from everything
that grades it (answers): target ground-truth depth, the hidden candidate
images, and the held-out novel-view quality of every acquisition sequence.

Usage: python -m data.build_assets --seeds 0-15 --out data/train_assets
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from data import backbone as bb
from data.scenes import ROLE_COUNTS, ROLE_ORDER, camera_rays, make_scene, render_all

STEPS = {"m0": 1500, "m1": 500, "m2": 1000, "dense": 3000}
INIT_SEED = 20250627
BASE = {}
_offset = 0
for _role in ROLE_ORDER:
    BASE[_role] = _offset
    _offset += ROLE_COUNTS[_role]
VIEWS_PER_SCENE = _offset
S = ROLE_COUNTS["sparse_source"]
C = ROLE_COUNTS["candidate"]
T = ROLE_COUNTS["target"]
E = ROLE_COUNTS["eval"]
DENSE = [BASE["sparse_source"] + i for i in range(S)] + [BASE["dense_extra"] + i for i in range(ROLE_COUNTS["dense_extra"])]


def _log(msg: str) -> None:
    print(f"[build {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _gid(scene: int, role: str, i: int) -> int:
    return scene * VIEWS_PER_SCENE + BASE[role] + i


def _ray_table(scenes: list[dict], gts: list[dict], device) -> dict:
    origins, dirs, cos, rgb = [], [], [], []
    for sc, gt in zip(scenes, gts):
        for v in range(VIEWS_PER_SCENE):
            o, d, c = camera_rays(gt["poses"][v], sc["intrinsics"], sc["width"], sc["height"])
            origins.append(o[0].reshape(-1, 3))
            dirs.append(d[0].reshape(-1, 3))
            cos.append(c[0].reshape(-1))
            rgb.append(gt["images"][v].reshape(-1, 3))
    as_t = lambda xs: torch.as_tensor(np.stack(xs), dtype=torch.float32, device=device)
    return {"origins": as_t(origins), "dirs": as_t(dirs), "cos": as_t(cos), "rgb": as_t(rgb)}


def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean SSIM per image for [N,H,W,3] tensors in [0,1] (11x11 Gaussian, sigma 1.5, valid)."""
    coords = torch.arange(11, dtype=pred.dtype, device=pred.device) - 5.0
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    win = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
    x = pred.permute(0, 3, 1, 2).contiguous()
    y = gt.permute(0, 3, 1, 2).contiguous()
    f = lambda z: torch.nn.functional.conv2d(z, win, groups=3)
    mx, my = f(x), f(y)
    sxx = f(x * x) - mx * mx
    syy = f(y * y) - my * my
    sxy = f(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return s.mean(dim=(1, 2, 3))


def _quality(rgb: torch.Tensor, depth: torch.Tensor, gt_rgb: torch.Tensor, gt_depth: torch.Tensor, h: int, w: int):
    """rgb [M,K,P,3], depth [M,K,P] vs gt [M,K,P,3]/[M,K,P] -> psnr, ssim, depth_mae per model."""
    m, k = rgb.shape[:2]
    mse = torch.mean((rgb - gt_rgb) ** 2, dim=(2, 3)).clamp_min(1e-10)
    psnr = (-10.0 * torch.log10(mse)).mean(dim=1)
    ssim = _ssim(rgb.reshape(m * k, h, w, 3), gt_rgb.reshape(m * k, h, w, 3)).reshape(m, k).mean(dim=1)
    valid = torch.isfinite(gt_depth)
    err = torch.where(valid, (depth - torch.where(valid, gt_depth, 0.0)).abs(), 0.0)
    mae = err.sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)
    return psnr.cpu().numpy(), ssim.cpu().numpy(), mae.cpu().numpy()


def _pad(lists: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(x) for x in lists)
    padded = torch.as_tensor([x + [x[0]] * (width - len(x)) for x in lists], dtype=torch.long)
    return padded, torch.as_tensor([len(x) for x in lists], dtype=torch.long)


def _run_stage(name, jobs, rays, stream_seeds, steps, group, render_lists, eval_fn, device, keep_states=False):
    """Train a stage in groups of ``group`` models.

    jobs: list of dicts with keys state (single-model state on CPU), views, scene.
    render_lists: function job -> list of global view ids to render after training.
    Returns per-job render tensors (CPU numpy) and optionally trained states.
    """
    out_renders, out_states, out_eval = [None] * len(jobs), [None] * len(jobs), [None] * len(jobs)
    t0 = time.time()
    start = 0
    g = group
    while start < len(jobs):
        chunk = jobs[start : start + g]
        try:
            state = bb.to_device(bb.concat([j["state"] for j in chunk]), device)
            views, counts = _pad([j["views"] for j in chunk])
            streams = torch.as_tensor([j["scene"] for j in chunk])
            bb.train(state, rays, views, counts, streams, stream_seeds, steps)
            rl = [render_lists(j) for j in chunk]
            rgb, depth, _ = bb.render_views(state, rays, torch.as_tensor(rl, dtype=torch.long))
            ev = eval_fn(chunk, rgb, depth) if eval_fn is not None else [None] * len(chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if g == 1:
                raise
            g = max(1, g // 2)
            _log(f"{name}: OOM, retrying with group size {g}")
            continue
        for i, j in enumerate(chunk):
            idx = start + i
            out_renders[idx] = (
                np.clip(np.round(rgb[i].cpu().numpy() * 255.0), 0, 255).astype(np.uint8),
                depth[i].cpu().numpy().astype(np.float16),
            )
            out_eval[idx] = ev[i]
            if keep_states:
                out_states[idx] = bb.to_device(bb.take(state, [i]), "cpu")
        start += len(chunk)
        _log(f"{name}: {start}/{len(jobs)} models trained ({time.time() - t0:.0f}s)")
    return out_renders, out_states, out_eval


def build(seeds: list[int], out_dir: Path, names: list[str] | None = None, group: int = 32,
          width: int = 48, height: int = 48, steps: dict | None = None, include_seed: bool = True,
          device: str | None = None) -> None:
    steps = dict(STEPS if steps is None else steps)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    names = names or [f"scene_{i:03d}" for i in range(len(seeds))]
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"device={device} scenes={len(seeds)} steps={steps} out={out_dir}")
    scenes = [make_scene(s, width, height) for s in seeds]
    gts = [render_all(sc) for sc in scenes]
    for gt in gts:
        tgt = gt["depths"][BASE["target"] : BASE["target"] + T]
        if np.any(np.isfinite(tgt).reshape(T, -1).sum(axis=1) < 50):
            _log("warning: a target view sees fewer than 50 surface pixels")
    rays = _ray_table(scenes, gts, device)
    gt_depth_t = torch.as_tensor(np.stack([gt["depths"] for gt in gts]).reshape(len(scenes) * VIEWS_PER_SCENE, -1), device=device)
    n = len(scenes)
    h, w = height, width
    stage_tag = {"m0": 11, "dense": 23, "m1": 37, "m2": 53}
    stream = lambda tag: {i: 1000003 * (i + 1) + stage_tag[tag] for i in range(n)}
    init = bb.init_state(1, INIT_SEED, torch.device("cpu"))

    def eval_views(scene):
        return [_gid(scene, "eval", e) for e in range(E)]

    def quality_fn(chunk, rgb, depth):
        ids = torch.as_tensor([eval_views(j["scene"]) for j in chunk], device=device)
        k = len(eval_views(chunk[0]["scene"]))
        rgb_e, depth_e = rgb[:, -k:], depth[:, -k:]
        gt_rgb = rays["rgb"][ids]
        gt_d = gt_depth_t[ids]
        p, s, m = _quality(rgb_e, depth_e, gt_rgb, gt_d, h, w)
        return list(zip(p.tolist(), s.tolist(), m.tolist()))

    sparse_ids = lambda sc: [_gid(sc, "sparse_source", i) for i in range(S)]
    cand_ids = lambda sc: [_gid(sc, "candidate", i) for i in range(C)]
    tgt_ids = lambda sc: [_gid(sc, "target", i) for i in range(T)]
    dense_ids = lambda sc: [sc * VIEWS_PER_SCENE + v for v in DENSE]

    # Stage A: sparse model M0.
    jobs = [{"state": init, "views": sparse_ids(sc), "scene": sc} for sc in range(n)]
    r0, s0, q0 = _run_stage(
        "M0", jobs, rays, stream("m0"), steps["m0"], group,
        lambda j: sparse_ids(j["scene"]) + tgt_ids(j["scene"]) + cand_ids(j["scene"]) + eval_views(j["scene"]),
        quality_fn, device, keep_states=True,
    )
    # Stage B: dense model D.
    jobs = [{"state": init, "views": dense_ids(sc), "scene": sc} for sc in range(n)]
    rd, _, _ = _run_stage(
        "D", jobs, rays, stream("dense"), steps["dense"], group,
        lambda j: dense_ids(j["scene"]) + tgt_ids(j["scene"]), None, device,
    )
    # Stage C: first acquisition M1[c].
    jobs = [
        {"state": s0[sc], "views": sparse_ids(sc) + [_gid(sc, "candidate", c)], "scene": sc, "c": c}
        for sc in range(n)
        for c in range(C)
    ]
    r1, s1, q1 = _run_stage(
        "M1", jobs, rays, stream("m1"), steps["m1"], group,
        lambda j: sparse_ids(j["scene"]) + [_gid(j["scene"], "candidate", j["c"])] + cand_ids(j["scene"]) + eval_views(j["scene"]),
        quality_fn, device, keep_states=True,
    )
    # Stage D: second acquisition M2[a, b].
    jobs = []
    for sc in range(n):
        for a in range(C):
            for b2 in range(C):
                if a == b2:
                    continue
                jobs.append(
                    {
                        "state": s1[sc * C + a],
                        "views": sparse_ids(sc) + [_gid(sc, "candidate", a), _gid(sc, "candidate", b2)],
                        "scene": sc,
                        "a": a,
                        "b": b2,
                    }
                )
    _, _, q2 = _run_stage("M2", jobs, rays, stream("m2"), steps["m2"], group, lambda j: eval_views(j["scene"]), quality_fn, device)

    for sc in range(n):
        gt = gts[sc]
        scene = scenes[sc]
        d = out_dir / names[sc]
        d.mkdir(parents=True, exist_ok=True)
        to_img = lambda x: x.reshape(-1, h, w, 3)
        to_dep = lambda x: x.reshape(-1, h, w)
        u8 = lambda x: np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
        rgb0, dep0 = r0[sc]
        rgbd, depd = rd[sc]
        i_src, i_tgt, i_cand = slice(0, S), slice(S, S + T), slice(S + T, S + T + C)
        nd = len(DENSE)
        step2_src_rgb = np.stack([to_img(r1[sc * C + c][0][: S + 1]) for c in range(C)])
        step2_src_dep = np.stack([to_dep(r1[sc * C + c][1][: S + 1]) for c in range(C)])
        step2_cand_rgb = np.stack([to_img(r1[sc * C + c][0][S + 1 : S + 1 + C]) for c in range(C)])
        step2_cand_dep = np.stack([to_dep(r1[sc * C + c][1][S + 1 : S + 1 + C]) for c in range(C)])
        poses = scene["poses"]
        inputs = {
            "intrinsics": scene["intrinsics"],
            "sparse_source_poses": poses["sparse_source"],
            "sparse_source_images": u8(gt["images"][BASE["sparse_source"] : BASE["sparse_source"] + S]),
            "sparse_source_rgb": to_img(rgb0[i_src]),
            "sparse_source_depth": to_dep(dep0[i_src]),
            "target_poses": poses["target"],
            "sparse_target_rgb": to_img(rgb0[i_tgt]),
            "sparse_target_depth": to_dep(dep0[i_tgt]),
            "candidate_poses": poses["candidate"],
            "sparse_candidate_rgb": to_img(rgb0[i_cand]),
            "sparse_candidate_depth": to_dep(dep0[i_cand]),
            "dense_source_poses": np.concatenate([poses["sparse_source"], poses["dense_extra"]]),
            "dense_source_images": u8(gt["images"][DENSE]),
            "dense_source_rgb": to_img(rgbd[:nd]),
            "dense_source_depth": to_dep(depd[:nd]),
            "dense_target_rgb": to_img(rgbd[nd : nd + T]),
            "dense_target_depth": to_dep(depd[nd : nd + T]),
            "step2_source_rgb": step2_src_rgb,
            "step2_source_depth": step2_src_dep,
            "step2_candidate_rgb": step2_cand_rgb,
            "step2_candidate_depth": step2_cand_dep,
        }
        final = np.full((3, C, C), np.nan)
        for j, job in enumerate(jobs):
            if job["scene"] == sc:
                final[:, job["a"], job["b"]] = q2[j]
        step1 = np.array([q1[sc * C + c] for c in range(C)]).T
        answers = {
            "target_depth_gt": gt["depths"][BASE["target"] : BASE["target"] + T].astype(np.float32),
            "candidate_images": u8(gt["images"][BASE["candidate"] : BASE["candidate"] + C]),
            "final_psnr": final[0],
            "final_ssim": final[1],
            "final_depth_mae": final[2],
            "step1_psnr": step1[0],
            "step1_ssim": step1[1],
            "step1_depth_mae": step1[2],
            "step0_quality": np.array(q0[sc]),
            "eval_poses": poses["eval"],
        }
        np.savez_compressed(d / "inputs.npz", **inputs)
        np.savez_compressed(d / "answers.npz", **answers)
        meta = {"name": names[sc], "steps": steps, "width": w, "height": h, "init_seed": INIT_SEED}
        if include_seed:
            meta["seed"] = int(seeds[sc])
        (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "scenes": names,
        "counts": {"source": S, "dense_source": len(DENSE), "target": T, "candidate": C, "eval": E, "acquisitions": 2},
        "width": w,
        "height": h,
        "steps": steps,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _log(f"wrote {n} scenes to {out_dir}")


def _parse_seeds(text: str) -> list[int]:
    if "-" in text and "," not in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0-15")
    p.add_argument("--out", default="data/train_assets")
    p.add_argument("--group", type=int, default=32)
    p.add_argument("--steps-scale", type=float, default=1.0, help="smoke-test only: scale every stage's steps")
    p.add_argument("--size", type=int, default=48)
    args = p.parse_args()
    steps = {k: max(1, int(round(v * args.steps_scale))) for k, v in STEPS.items()}
    build(_parse_seeds(args.seeds), Path(args.out), group=args.group, steps=steps, width=args.size, height=args.size)


if __name__ == "__main__":
    main()
