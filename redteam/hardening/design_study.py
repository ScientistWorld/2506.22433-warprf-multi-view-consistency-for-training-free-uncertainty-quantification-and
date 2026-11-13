"""Design study for the rebuilt acquisition experiment (one GPU, ~30 min).

Questions answered on 24 public scenes (seeds 7000-7023, disjoint from every
benchmark split):

1. Acquisition: in which candidate / held-out pose design does a better
   uncertainty estimate translate into a better acquisition? For three
   candidate laws (the old wide law, candidates drawn from the held-out views'
   own law, and a wider shared law) every one-step branch M1[c] = M0 + candidate
   c is trained, and each selector's pick is scored by the held-out PSNR of its
   branch. Selectors: WarpRF Eq. 9, honest variants, farthest camera, random,
   and the true candidate rendering error (an oracle uncertainty).
2. Depth convention: AUSE of WarpRF Eq. 8 against one-line depth rules when
   rendered depth composites the background at the far bound (old gym), uses
   paper Eq. 2 literally, or is normalised by accumulated opacity.
3. Throughput (ms per model-step) for planning the full rebuild.
4. Whether the evaluator's namespace isolation works in the job container.

Writes redteam/hardening/design_study.json and prints a summary.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import backbone as bb  # noqa: E402
from data.scenes import BACKGROUND_RGB, BOUND_RADIUS, _sample_camera, camera_rays, make_scene, render_view  # noqa: E402
from method.warprf import _relative, backproject, image_uncertainty, pixel_uncertainty, project, sample_bilinear, transform  # noqa: E402

SEEDS = list(range(7000, 7024))
H = W = 48
INIT_SEED = 20250627
OUT = ROOT / "redteam" / "hardening" / "design_study.json"
GOLDEN = 0.6180339887498949
# view layout per scene
LAYOUT = [("sparse", 3), ("dense_extra", 5), ("target", 6), ("cand_old", 8), ("eval_old", 8),
          ("cand_narrow", 16), ("cand_wide", 16), ("eval_wide", 8)]
OFF = {}
_o = 0
for _n, _k in LAYOUT:
    OFF[_n] = _o
    _o += _k
NV = _o
CAND_SETS = ("cand_old", "cand_narrow", "cand_wide")


def log(msg):
    print(f"[study {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cameras(seed):
    sc = make_scene(seed, W, H)
    rng = np.random.default_rng([seed, 99])
    narrow = [_sample_camera(rng, rng.uniform(0, 2 * math.pi), (2.8, 3.2), (15.0, 35.0), 0.1) for _ in range(16)]
    wide = [_sample_camera(rng, rng.uniform(0, 2 * math.pi), (2.6, 3.5), (5.0, 50.0), 0.25) for _ in range(16)]
    base = rng.uniform(0, 2 * math.pi)
    ev = [_sample_camera(rng, base + math.radians(k * 45.0 + rng.uniform(-15, 15)), (2.6, 3.5), (5.0, 50.0), 0.25) for k in range(8)]
    p = sc["poses"]
    poses = np.concatenate([p["sparse_source"], p["dense_extra"], p["target"], p["candidate"], p["eval"],
                            np.stack(narrow), np.stack(wide), np.stack(ev)]).astype(np.float32)
    return sc, poses


# ----------------------------------------------------------------------------- rendering with all depth conventions

def _render(params, o, d, cs):
    b = torch.sum(o * d, dim=-1)
    c = torch.sum(o * o, dim=-1) - BOUND_RADIUS**2
    disc = b * b - c
    hit = disc > 0.0
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    t_near = torch.where(hit, torch.clamp(-b - sq, min=0.0), torch.full_like(b, bb.MISS_DISTANCE))
    t_far = torch.where(hit, -b + sq, torch.full_like(b, bb.MISS_DISTANCE))
    m, r = b.shape
    u = ((torch.arange(bb.SAMPLES, device=o.device, dtype=o.dtype) + 0.5) / bb.SAMPLES).expand(m, r, bb.SAMPLES)
    span = (t_far - t_near)[..., None]
    t = t_near[..., None] + span * u
    delta = span / bb.SAMPLES
    pts = o[..., None, :] + d[..., None, :] * t[..., None]
    sigma, rgb = bb.query(params, pts.reshape(m, r * bb.SAMPLES, 3),
                          d[..., None, :].expand(m, r, bb.SAMPLES, 3).reshape(m, r * bb.SAMPLES, 3))
    sigma = sigma.reshape(m, r, bb.SAMPLES)
    rgb = rgb.reshape(m, r, bb.SAMPLES, 3)
    alpha = 1.0 - torch.exp(-sigma * delta)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    w = alpha * trans
    acc = w.sum(-1)
    bg = torch.as_tensor(BACKGROUND_RGB, device=o.device, dtype=o.dtype)
    color = torch.sum(w[..., None] * rgb, dim=-2) + (1.0 - acc)[..., None] * bg
    swt = torch.sum(w * t, dim=-1)
    da = (swt + (1.0 - acc) * t_far) * cs
    db = swt * cs
    dc = torch.where(acc > 1e-6, swt / acc.clamp_min(1e-6), t_far) * cs
    return color, torch.stack([da, db, dc], dim=-1), acc


@torch.no_grad()
def render(state, rays, view_ids, chunk=2304):
    m, k = view_ids.shape
    P = rays["origins"].shape[1]
    dev = rays["origins"].device
    vid_all = view_ids.to(dev)
    rgb = torch.empty(m, k, P, 3, device=dev)
    dep = torch.empty(m, k, P, 3, device=dev)
    acc = torch.empty(m, k, P, device=dev)
    for j in range(k):
        vid = vid_all[:, j]
        for s in range(0, P, chunk):
            sl = slice(s, min(s + chunk, P))
            c, d, a = _render(state["params"], rays["origins"][vid][:, sl], rays["dirs"][vid][:, sl], rays["cos"][vid][:, sl])
            rgb[:, j, sl] = c
            dep[:, j, sl] = d
            acc[:, j, sl] = a
    return rgb.cpu().numpy(), dep.cpu().numpy(), acc.cpu().numpy()


# ----------------------------------------------------------------------------- metrics (copied from the evaluator)

def _trap(y, x):
    return float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(x)))


def ause(unc, err, tb):
    n = err.size
    ratios = np.linspace(0.0, 1.0, 100, endpoint=False)
    keep = np.maximum(np.array([int((1.0 - r) * n) for r in ratios]), 1)
    oracle = np.cumsum(np.sort(err))[keep - 1] / keep
    order = np.lexsort((tb, unc))
    by = np.cumsum(err[order])[keep - 1] / keep
    s = float(np.mean(err))
    return 0.0 if s <= 0 else _trap((by - oracle) / s, ratios)


def psnr(a, b):
    return float(-10.0 * np.log10(max(np.mean((a - b) ** 2), 1e-10)))


# ----------------------------------------------------------------------------- acquisition selectors

def masked_eq9(target_image, target_depth, target_pose, source_images, source_poses, k):
    """Eq. 9 restricted to pixels that reproject in front of and inside >= 1 source; returns (sum, mean, coverage)."""
    k = k.astype(np.float64)
    pts = backproject(target_depth, k)
    tgt = target_image.astype(np.float64)
    best = np.full(target_depth.shape, np.inf)
    for img, pose in zip(source_images, source_poses):
        s_from_t, _ = _relative(target_pose, pose)
        ps = transform(pts, s_from_t)
        u, v = project(ps, k)
        ok = (ps[..., 2] > 1e-6) & (u >= 0) & (u <= W) & (v >= 0) & (v <= H)
        diff = np.abs(sample_bilinear(img.astype(np.float64), u, v) - tgt).sum(-1)
        best = np.where(ok, np.minimum(best, diff), best)
    seen = np.isfinite(best)
    cov = float(seen.mean())
    return float(np.where(seen, best, 0).sum()), float(best[seen].mean()) if seen.any() else 0.0, cov


def angle_kernel(centers, sigma_deg=30.0):
    u = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    cosang = np.clip(u @ u.T, -1, 1)
    ang = np.degrees(np.arccos(cosang))
    return np.exp(-(ang**2) / (2 * sigma_deg**2))


def main():
    t_all = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={dev} {torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''}")
    result = {"device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}

    # ---- isolation probe (cheap, do first)
    iso = {}
    for split in ("train", "test"):
        launcher = ROOT / "eval" / split / "isolate.py"
        if not launcher.is_file():
            continue
        probe = ("import os,sys;"
                 "print('hidden_test_listing', os.listdir('/home/user/eval/test'));"
                 "print('git_listing', os.listdir('/home/user/.git') if os.path.isdir('/home/user/.git') else None);"
                 "print('pids', sorted(int(p) for p in os.listdir('/proc') if p.isdigit()));"
                 "import numpy; print('numpy_ok', numpy.__version__);"
                 "open('/home/user/tmp/iso_probe_write','w').write('x'); print('write_ok')")
        cmd = [sys.executable, "-I", str(launcher), "--hide", str(ROOT / "eval" / "test"), "--hide", str(ROOT / ".git"),
               "--", sys.executable, "-c", probe]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=120)
        iso[split] = {"rc": p.returncode, "stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:]}
        p2 = subprocess.run([sys.executable, "-I", str(launcher), "--hide", str(ROOT / "eval" / "test"), "--",
                             "sh", "-c", "umount /home/user/eval/test; ls /home/user/eval/test | wc -l"],
                            capture_output=True, text=True, cwd=str(ROOT), timeout=60)
        iso[split]["umount_attempt"] = {"rc": p2.returncode, "stdout": p2.stdout, "stderr": p2.stderr[-500:]}
    result["isolation"] = iso
    log(f"isolation probe: {json.dumps(iso)[:1500]}")

    # ---- scenes and ground truth
    scenes, poses_all, gt_rgb, gt_dep = [], [], [], []
    for s in SEEDS:
        sc, poses = cameras(s)
        scenes.append(sc)
        poses_all.append(poses)
        imgs, deps = zip(*(render_view(sc, p) for p in poses))
        gt_rgb.append(np.stack(imgs))
        gt_dep.append(np.stack(deps))
    log(f"ground truth rendered ({time.time() - t_all:.0f}s)")
    n = len(SEEDS)
    K = scenes[0]["intrinsics"]
    O, D, C, R = [], [], [], []
    for i in range(n):
        for v in range(NV):
            o, d, c = camera_rays(poses_all[i][v], K, W, H)
            O.append(o[0].reshape(-1, 3)); D.append(d[0].reshape(-1, 3)); C.append(c[0].reshape(-1)); R.append(gt_rgb[i][v].reshape(-1, 3))
    T = lambda xs: torch.as_tensor(np.stack(xs), dtype=torch.float32, device=dev)
    rays = {"origins": T(O), "dirs": T(D), "cos": T(C), "rgb": T(R)}
    gid = lambda i, role, j: i * NV + OFF[role] + j
    init = bb.init_state(1, INIT_SEED, torch.device("cpu"))
    timing = {}

    def train_group(states, view_lists, scene_ids, tag, steps):
        state = bb.to_device(bb.concat(states), dev)
        width = max(len(v) for v in view_lists)
        views = torch.as_tensor([v + [v[0]] * (width - len(v)) for v in view_lists], dtype=torch.long)
        counts = torch.as_tensor([len(v) for v in view_lists], dtype=torch.long)
        seeds = {i: 1000003 * (i + 1) + tag for i in set(scene_ids)}
        t0 = time.time()
        bb.train(state, rays, views, counts, torch.as_tensor(scene_ids), seeds, steps)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return state, time.time() - t0

    # ---- M0 and D (all scenes in one group each)
    m0, dt = train_group([init] * n, [[gid(i, "sparse", j) for j in range(3)] for i in range(n)], list(range(n)), 11, 1500)
    timing["m0_ms_per_model_step"] = 1000 * dt / (n * 1500)
    log(f"M0 trained ({dt:.0f}s)")
    dense_views = lambda i: [gid(i, "sparse", j) for j in range(3)] + [gid(i, "dense_extra", j) for j in range(5)]
    dd, dt = train_group([init] * n, [dense_views(i) for i in range(n)], list(range(n)), 23, 3000)
    timing["dense_ms_per_model_step"] = 1000 * dt / (n * 3000)
    log(f"D trained ({dt:.0f}s)")

    all_ids = lambda i: [[i * NV + v for v in range(NV)]]
    r0 = [render(bb.take(m0, [i]), rays, torch.as_tensor(all_ids(i))) for i in range(n)]
    rd = [render(bb.take(dd, [i]), rays, torch.as_tensor(all_ids(i))) for i in range(n)]
    m0_cpu = [bb.to_device(bb.take(m0, [i]), "cpu") for i in range(n)]
    del m0, dd
    log("M0/D rendered")

    # ---- M1 branches: 40 candidates per scene, one scene per group
    cand_list = [(cs, j) for cs in CAND_SETS for j in range(8 if cs == "cand_old" else 16)]
    step1 = {cs: np.zeros((n, 16, 2)) for cs in CAND_SETS}  # [scene, cand, eval_set]
    t_m1 = 0.0
    eval_ids = {"eval_old": [], "eval_wide": []}
    for i in range(n):
        views = [[gid(i, "sparse", j) for j in range(3)] + [gid(i, cs, j)] for cs, j in cand_list]
        st, dt = train_group([m0_cpu[i]] * len(cand_list), views, [i] * len(cand_list), 37, 500)
        t_m1 += dt
        ev = [gid(i, "eval_old", j) for j in range(8)] + [gid(i, "eval_wide", j) for j in range(8)]
        rgb, _, _ = render(st, rays, torch.as_tensor([ev] * len(cand_list)))
        gt = np.stack([gt_rgb[i][OFF["eval_old"] + j] for j in range(8)] + [gt_rgb[i][OFF["eval_wide"] + j] for j in range(8)]).reshape(16, -1, 3)
        for m, (cs, j) in enumerate(cand_list):
            pred = np.clip(np.round(rgb[m] * 255), 0, 255) / 255.0
            step1[cs][i, j, 0] = np.mean([psnr(pred[e], gt[e]) for e in range(8)])
            step1[cs][i, j, 1] = np.mean([psnr(pred[e], gt[e]) for e in range(8, 16)])
        del st
        if i == 0 or (i + 1) % 6 == 0:
            log(f"M1 scene {i + 1}/{n} ({t_m1:.0f}s)")
    timing["m1_ms_per_model_step"] = 1000 * t_m1 / (n * len(cand_list) * 500)
    timing["m1_group_size"] = len(cand_list)
    result["timing"] = timing
    log(f"timing {timing}")

    # ---- analysis A: depth uncertainty under three depth conventions
    conv_names = ["far_composite_old", "paper_eq2", "opacity_normalised"]
    depth_res = {}
    for setting, rr, srcs in (("sparse", r0, lambda i: [OFF["sparse"] + j for j in range(3)]),
                              ("dense", rd, lambda i: [OFF["sparse"] + j for j in range(3)] + [OFF["dense_extra"] + j for j in range(5)])):
        for ci, cname in enumerate(conv_names):
            rows = {k: [] for k in ("warprf", "depth", "neg_depth", "random", "one_minus_acc")}
            mae = []
            for i in range(n):
                rgb, dep, acc = rr[i]
                dep = dep[0, :, :, ci].reshape(NV, H, W).astype(np.float16).astype(np.float32)
                accv = acc[0].reshape(NV, H, W)
                sp = poses_all[i]
                per = {k: [] for k in rows}
                for t in range(6):
                    v = OFF["target"] + t
                    g = gt_dep[i][v]
                    valid = np.isfinite(g)
                    err = np.abs(dep[v][valid] - g[valid])
                    mae.append(float(err.mean()))
                    tb = np.mod((np.arange(valid.sum()) + 1.0) * GOLDEN + 0.37 * t, 1.0)
                    s_idx = srcs(i)
                    u = pixel_uncertainty(dep[v], sp[v], dep[s_idx], sp[s_idx], K)
                    rnd = np.mod(np.arange(H * W) * 0.7548776662466927 + 0.5698402909980532 * (t + 1), 1.0).reshape(H, W)
                    cues = {"warprf": u, "depth": dep[v], "neg_depth": -dep[v], "random": rnd, "one_minus_acc": 1 - accv[v]}
                    for k2, cue in cues.items():
                        per[k2].append(ause(cue[valid].astype(np.float64), err.astype(np.float64), tb))
                for k2 in rows:
                    rows[k2].append(float(np.mean(per[k2])))
            summ = {k2: float(np.mean(v)) for k2, v in rows.items()}
            for k2 in ("depth", "neg_depth", "one_minus_acc"):
                d = np.array(rows["warprf"]) - np.array(rows[k2])  # positive = cue better than warprf
                summ[f"{k2}_gain_over_warprf"] = float(d.mean())
                summ[f"{k2}_gain_se"] = float(d.std(ddof=1) / math.sqrt(n))
            summ["depth_mae"] = float(np.mean(mae))
            depth_res[f"{setting}/{cname}"] = summ
            log(f"depth {setting}/{cname}: {json.dumps({k: round(v, 4) for k, v in summ.items()})}")
    result["depth"] = depth_res

    # ---- analysis B: one-step acquisition
    acq = {}
    for ci, cname in ((0, "far_composite_old"), (2, "opacity_normalised")):
        for cs in CAND_SETS:
            ncand = 8 if cs == "cand_old" else 16
            sc_scores = {k: np.zeros((n, ncand)) for k in ("warprf", "warprf_masked_sum", "warprf_masked_mean", "warprf_nbr", "farthest", "oracle_unc", "coverage")}
            for i in range(n):
                rgb, dep, _ = r0[i]
                rgbv = np.clip(np.round(rgb[0] * 255), 0, 255).reshape(NV, H, W, 3) / 255.0
                depv = dep[0, :, :, ci].reshape(NV, H, W).astype(np.float16).astype(np.float32)
                sp = poses_all[i]
                src = [OFF["sparse"] + j for j in range(3)]
                for j in range(ncand):
                    v = OFF[cs] + j
                    sc_scores["warprf"][i, j] = image_uncertainty(rgbv[v], depv[v], sp[v], rgbv[src], sp[src], K)
                    ms, mm, cov = masked_eq9(rgbv[v], depv[v], sp[v], rgbv[src], sp[src], K)
                    sc_scores["warprf_masked_sum"][i, j] = ms
                    sc_scores["warprf_masked_mean"][i, j] = mm
                    sc_scores["coverage"][i, j] = 1 - cov
                    cen = sp[v][:3, 3]
                    sc_scores["farthest"][i, j] = min(np.linalg.norm(cen - sp[s][:3, 3]) for s in src)
                    sc_scores["oracle_unc"][i, j] = np.abs(rgbv[v] - gt_rgb[i][v]).sum()
                centers = sp[OFF[cs]: OFF[cs] + ncand, :3, 3]
                sc_scores["warprf_nbr"][i] = angle_kernel(centers) @ sc_scores["warprf"][i]
            for k_sub in sorted({8, ncand}):
                for ei, ename in ((0, "eval_old"), (1, "eval_wide")):
                    P = step1[cs][:, :k_sub, ei]
                    key = f"{cname}/{cs}/k{k_sub}/{ename}"
                    picks = {}
                    for sel, S in sc_scores.items():
                        idx = np.argmax(S[:, :k_sub], axis=1)
                        picks[sel] = P[np.arange(n), idx]
                    rep = {sel: float(v.mean()) for sel, v in picks.items()}
                    rep["random"] = float(P.mean())
                    rep["best"] = float(P.max(axis=1).mean())
                    for sel in picks:
                        if sel == "warprf":
                            continue
                        d = picks[sel] - picks["warprf"]
                        rep[f"{sel}_minus_warprf"] = float(d.mean())
                        rep[f"{sel}_minus_warprf_se"] = float(d.std(ddof=1) / math.sqrt(n))
                    d = picks["warprf"] - P.mean(axis=1)
                    rep["warprf_minus_random"] = float(d.mean())
                    rep["warprf_minus_random_se"] = float(d.std(ddof=1) / math.sqrt(n))

                    def sp_corr(S):
                        out = []
                        for i in range(n):
                            a = np.argsort(np.argsort(S[i, :k_sub])); b = np.argsort(np.argsort(P[i]))
                            out.append(np.corrcoef(a, b)[0, 1])
                        return float(np.nanmean(out))

                    for sel in ("warprf", "oracle_unc", "farthest", "warprf_nbr"):
                        rep[f"spearman_{sel}_vs_gain"] = sp_corr(sc_scores[sel])
                    acq[key] = rep
                    log(f"acq {key}: " + json.dumps({k: round(v, 3) for k, v in rep.items()}))
    result["acquisition"] = acq
    result["elapsed_s"] = time.time() - t_all
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    log(f"done in {time.time() - t_all:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
