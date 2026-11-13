"""Procedural posed-view scenes with analytic RGB images and camera-z depth.

Each scene is drawn from a seeded random family: a textured floor plate plus
three to six ellipsoids and oriented boxes with procedural textures, Lambertian
shading and optional mild specular highlights. Cameras are sampled around the
scene for five fixed roles:

* ``sparse_source``  - 3 initial training views spread over a ~150 degree arc
* ``dense_extra``    - 5 extra training views (dense setting = sparse + extra)
* ``target``         - 6 held-out views used to score pixel uncertainty
* ``candidate``      - 16 candidate next views for active view selection
* ``eval``           - 8 held-out views used to score novel-view quality

Candidate and held-out (eval) cameras are drawn from one pose law (radius
2.8-3.2, elevation 15-35 degrees, look-at jitter 0.1): candidates i.i.d. with
uniform azimuth, held-out views stratified in azimuth around the scene. As in
the paper's benchmarks, where the candidate pool and the test views come from
the same capture, the views a method may acquire are the kind of views it is
judged on.

Camera convention (OpenCV): camera-to-world 4x4 poses, camera x right, y down,
z forward. Pixel (row i, col j) has its centre at continuous image coordinates
(u, v) = (j + 0.5, i + 0.5); intrinsics use cx = W / 2, cy = H / 2. Depth is the
camera-space z coordinate of the first surface hit (``inf`` for background).
"""

from __future__ import annotations

import math

import numpy as np

BOUND_RADIUS = 1.9  # every primitive lies inside this sphere around the origin
BACKGROUND_RGB = np.array([0.08, 0.08, 0.08], dtype=np.float32)
ROLE_COUNTS = {"sparse_source": 3, "dense_extra": 5, "target": 6, "candidate": 16, "eval": 8}
VIEW_LAW = {"radius": (2.8, 3.2), "elevation_deg": (15.0, 35.0), "jitter": 0.1}  # candidates and eval views
ROLE_ORDER = ("sparse_source", "dense_extra", "target", "candidate", "eval")
PATTERNS = ("stripes", "checker", "rings", "gradient")


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OpenCV-style camera-to-world matrix looking from ``eye`` at ``target``."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w.astype(np.float32)


def intrinsics_for(width: int, height: int) -> np.ndarray:
    focal = 1.0 * width
    return np.array([[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)


def camera_rays(pose: np.ndarray, intrinsics: np.ndarray, width: int, height: int, offsets=((0.5, 0.5),)):
    """World-space ray origins/unit directions and per-ray cos(angle to optical axis).

    Returns arrays shaped [len(offsets), H, W, 3], [len(offsets), H, W, 3], [len(offsets), H, W].
    """
    pose = pose.astype(np.float64)
    out_o, out_d, out_c = [], [], []
    for ox, oy in offsets:
        xs, ys = np.meshgrid(np.arange(width, dtype=np.float64) + ox, np.arange(height, dtype=np.float64) + oy)
        cam = np.stack(
            [(xs - intrinsics[0, 2]) / intrinsics[0, 0], (ys - intrinsics[1, 2]) / intrinsics[1, 1], np.ones_like(xs)],
            axis=-1,
        )
        norm = np.linalg.norm(cam, axis=-1, keepdims=True)
        cam_unit = cam / norm
        dirs = cam_unit @ pose[:3, :3].T
        out_o.append(np.broadcast_to(pose[:3, 3], dirs.shape).copy())
        out_d.append(dirs)
        out_c.append(cam_unit[..., 2])
    return np.stack(out_o), np.stack(out_d), np.stack(out_c)


def _rotation(rng: np.random.Generator, tilt: float) -> np.ndarray:
    yaw = rng.uniform(0.0, 2.0 * math.pi)
    ax, az = rng.uniform(-tilt, tilt, size=2)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    cx, sx = math.cos(ax), math.sin(ax)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    cz, sz = math.cos(az), math.sin(az)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_y @ rot_x @ rot_z


def _texture(rng: np.random.Generator) -> dict:
    base = rng.uniform(0.15, 0.95, size=3)
    other = np.clip(base + rng.uniform(-0.6, 0.6, size=3), 0.05, 1.0)
    return {
        "pattern": PATTERNS[int(rng.integers(len(PATTERNS)))],
        "color_a": base,
        "color_b": other,
        "freq": rng.uniform(3.0, 11.0, size=3) * rng.choice([-1.0, 1.0], size=3),
        "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        "specular": float(rng.uniform(0.05, 0.3)) if rng.uniform() < 0.5 else 0.0,
        "shininess": float(rng.uniform(8.0, 40.0)),
    }


def _sample_camera(rng, azimuth, radius_range, elev_range_deg, jitter):
    radius = rng.uniform(*radius_range)
    elev = math.radians(rng.uniform(*elev_range_deg))
    eye = np.array([radius * math.cos(elev) * math.sin(azimuth), radius * math.sin(elev), radius * math.cos(elev) * math.cos(azimuth)])
    target = np.array([0.0, -0.15, 0.0]) + rng.uniform(-jitter, jitter, size=3) * np.array([1.0, 0.5, 1.0])
    return look_at(eye, target)


def make_scene(seed: int, width: int = 48, height: int = 48) -> dict:
    """Deterministically sample one scene and its camera roles from ``seed``."""
    rng = np.random.default_rng(int(seed))
    floor_y = -0.65
    primitives = [
        {
            "kind": "ellipsoid",
            "center": np.array([0.0, floor_y - 0.07, 0.0]),
            "rotation": _rotation(rng, 0.0),
            "scale": np.array([1.35, 0.07, 1.35]),
            "texture": _texture(rng),
        }
    ]
    count = int(rng.integers(3, 7))
    placed = []
    for _ in range(count):
        kind = "ellipsoid" if rng.uniform() < 0.5 else "box"
        scale = rng.uniform(0.16, 0.4, size=3)
        if kind == "ellipsoid" and rng.uniform() < 0.4:
            scale[:] = scale[0]
        for _attempt in range(20):
            r = 0.85 * math.sqrt(rng.uniform())
            phi = rng.uniform(0.0, 2.0 * math.pi)
            xz = np.array([r * math.cos(phi), r * math.sin(phi)])
            if all(np.linalg.norm(xz - p) > 0.35 for p in placed):
                break
        placed.append(xz)
        lift = rng.uniform(0.0, 0.35) if rng.uniform() < 0.3 else 0.0
        center = np.array([xz[0], floor_y + scale[1] * 0.9 + lift, xz[1]])
        primitives.append(
            {
                "kind": kind,
                "center": center,
                "rotation": _rotation(rng, 0.35),
                "scale": scale,
                "texture": _texture(rng),
            }
        )
    light = np.array([rng.uniform(-0.6, 0.6), rng.uniform(0.6, 1.0), rng.uniform(-0.6, 0.6)])
    light = light / np.linalg.norm(light)
    ambient = float(rng.uniform(0.3, 0.45))

    base_az = rng.uniform(0.0, 2.0 * math.pi)
    poses = {}
    sparse_offsets = np.array([0.0, 75.0, 150.0]) + rng.uniform(-12.0, 12.0, size=3)
    poses["sparse_source"] = [
        _sample_camera(rng, base_az + math.radians(a), (2.7, 3.3), (12.0, 38.0), 0.15) for a in sparse_offsets
    ]
    dense_offsets = np.array([37.5, 112.5, 195.0, 255.0, 315.0]) + rng.uniform(-10.0, 10.0, size=5)
    poses["dense_extra"] = [
        _sample_camera(rng, base_az + math.radians(a), (2.7, 3.3), (10.0, 40.0), 0.15) for a in dense_offsets
    ]
    poses["target"] = [
        _sample_camera(rng, rng.uniform(0.0, 2.0 * math.pi), (2.5, 3.6), (5.0, 45.0), 0.25) for _ in range(6)
    ]
    law = (VIEW_LAW["radius"], VIEW_LAW["elevation_deg"], VIEW_LAW["jitter"])
    poses["candidate"] = [
        _sample_camera(rng, rng.uniform(0.0, 2.0 * math.pi), *law) for _ in range(ROLE_COUNTS["candidate"])
    ]
    eval_offsets = np.arange(ROLE_COUNTS["eval"]) * (360.0 / ROLE_COUNTS["eval"]) + rng.uniform(-15.0, 15.0, size=ROLE_COUNTS["eval"])
    poses["eval"] = [_sample_camera(rng, base_az + math.radians(a), *law) for a in eval_offsets]
    return {
        "seed": int(seed),
        "width": int(width),
        "height": int(height),
        "intrinsics": intrinsics_for(width, height),
        "primitives": primitives,
        "light": light,
        "ambient": ambient,
        "poses": {role: np.stack(poses[role]).astype(np.float32) for role in ROLE_ORDER},
    }


def _intersect(prim: dict, origins: np.ndarray, dirs: np.ndarray):
    """Ray/primitive intersection. Returns (t, local_point, world_normal)."""
    rot = prim["rotation"]
    o_l = (origins - prim["center"]) @ rot
    d_l = dirs @ rot
    scale = prim["scale"]
    n = origins.shape[0]
    t = np.full(n, np.inf)
    if prim["kind"] == "ellipsoid":
        os_ = o_l / scale
        ds_ = d_l / scale
        a = np.sum(ds_ * ds_, axis=-1)
        b = 2.0 * np.sum(os_ * ds_, axis=-1)
        c = np.sum(os_ * os_, axis=-1) - 1.0
        disc = b * b - 4.0 * a * c
        ok = disc > 0.0
        sq = np.sqrt(np.maximum(disc, 0.0))
        t0 = (-b - sq) / (2.0 * a)
        t1 = (-b + sq) / (2.0 * a)
        tt = np.where(t0 > 1e-4, t0, t1)
        hit = ok & (tt > 1e-4)
        t[hit] = tt[hit]
        p_l = o_l + d_l * np.where(hit, t, 0.0)[:, None]
        n_l = p_l / (scale * scale)
    else:
        safe = np.where(np.abs(d_l) < 1e-9, 1e-9, d_l)
        t_a = (-scale - o_l) / safe
        t_b = (scale - o_l) / safe
        t_min = np.minimum(t_a, t_b)
        t_max = np.maximum(t_a, t_b)
        t_near = np.max(t_min, axis=-1)
        t_far = np.min(t_max, axis=-1)
        hit = (t_near <= t_far) & (t_far > 1e-4)
        tt = np.where(t_near > 1e-4, t_near, t_far)
        t[hit] = tt[hit]
        axis = np.argmax(t_min, axis=-1)
        n_l = np.zeros_like(o_l)
        n_l[np.arange(n), axis] = -np.sign(safe[np.arange(n), axis])
        p_l = o_l + d_l * np.where(hit, t, 0.0)[:, None]
    n_w = n_l @ rot.T
    n_w = n_w / np.maximum(np.linalg.norm(n_w, axis=-1, keepdims=True), 1e-9)
    return t, p_l, n_w


def _pattern_value(tex: dict, p_l: np.ndarray) -> np.ndarray:
    f = tex["freq"]
    ph = tex["phase"]
    if tex["pattern"] == "stripes":
        v = 0.5 + 0.5 * np.sin(p_l @ f + ph)
    elif tex["pattern"] == "checker":
        s = np.sin(f[0] * p_l[:, 0] + ph) * np.sin(f[1] * p_l[:, 1]) * np.sin(f[2] * p_l[:, 2] + 0.5 * ph)
        v = 0.5 + 0.5 * np.tanh(6.0 * s)
    elif tex["pattern"] == "rings":
        v = 0.5 + 0.5 * np.sin(np.abs(f[0]) * 1.5 * np.linalg.norm(p_l, axis=-1) + ph)
    else:
        v = np.clip(0.5 + 0.8 * (p_l @ (f / np.linalg.norm(f))), 0.0, 1.0)
    return v


def trace(scene: dict, origins: np.ndarray, dirs: np.ndarray):
    """Shade a flat batch of rays. Returns rgb [N,3] and ray distance t [N] (inf = miss)."""
    best_t = np.full(origins.shape[0], np.inf)
    best_rgb = np.broadcast_to(BACKGROUND_RGB.astype(np.float64), origins.shape).copy()
    light = scene["light"]
    for prim in scene["primitives"]:
        t, p_l, n_w = _intersect(prim, origins, dirs)
        closer = t < best_t
        if not np.any(closer):
            continue
        tex = prim["texture"]
        v = _pattern_value(tex, p_l[closer])[:, None]
        albedo = (1.0 - v) * tex["color_a"] + v * tex["color_b"]
        n_c = n_w[closer]
        # Shade with the normal facing the camera.
        facing = np.sum(n_c * dirs[closer], axis=-1, keepdims=True) > 0.0
        n_c = np.where(facing, -n_c, n_c)
        diffuse = np.clip(n_c @ light, 0.0, 1.0)[:, None]
        rgb = albedo * (scene["ambient"] + (1.0 - scene["ambient"]) * diffuse)
        if tex["specular"] > 0.0:
            half = light[None, :] - dirs[closer]
            half = half / np.maximum(np.linalg.norm(half, axis=-1, keepdims=True), 1e-9)
            spec = np.clip(np.sum(n_c * half, axis=-1), 0.0, 1.0) ** tex["shininess"]
            rgb = rgb + tex["specular"] * spec[:, None]
        best_t[closer] = t[closer]
        best_rgb[closer] = rgb
    return np.clip(best_rgb, 0.0, 1.0), best_t


_SUPERSAMPLE = ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))


def render_view(scene: dict, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth RGB (2x2 supersampled) and camera-z depth (pixel-centre ray)."""
    w, h, k = scene["width"], scene["height"], scene["intrinsics"]
    o, d, _ = camera_rays(pose, k, w, h, offsets=_SUPERSAMPLE)
    rgb, _ = trace(scene, o.reshape(-1, 3), d.reshape(-1, 3))
    rgb = rgb.reshape(len(_SUPERSAMPLE), h, w, 3).mean(axis=0)
    oc, dc, cc = camera_rays(pose, k, w, h)
    _, t = trace(scene, oc.reshape(-1, 3), dc.reshape(-1, 3))
    depth = (t * cc.reshape(-1)).reshape(h, w)
    return rgb.astype(np.float32), depth.astype(np.float32)


def all_views(scene: dict) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Stack every role's poses in ROLE_ORDER; returns poses [V,4,4] and (role, index) labels."""
    poses, labels = [], []
    for role in ROLE_ORDER:
        for i, pose in enumerate(scene["poses"][role]):
            poses.append(pose)
            labels.append((role, i))
    return np.stack(poses), labels


def render_all(scene: dict) -> dict[str, np.ndarray]:
    poses, _ = all_views(scene)
    images, depths = zip(*(render_view(scene, p) for p in poses))
    return {"poses": poses, "images": np.stack(images), "depths": np.stack(depths)}
