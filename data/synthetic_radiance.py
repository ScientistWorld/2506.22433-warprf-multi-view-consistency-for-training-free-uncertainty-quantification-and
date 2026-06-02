"""Small posed-view scene used to exercise radiance-field uncertainty.

The scene is analytic so it can provide RGB images, depth maps, and camera poses
without downloading a benchmark dataset. It is intentionally not trivial: views
that are far from the training cameras expose parts of the object with different
colors and depth gradients, leaving room for uncertainty ranking to matter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraSet:
    poses: np.ndarray
    intrinsics: np.ndarray
    width: int
    height: int


def look_at(eye: np.ndarray, target: np.ndarray = np.zeros(3)) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def orbit_cameras(num_views: int, width: int, height: int, radius: float = 3.0) -> CameraSet:
    poses = []
    for i in range(num_views):
        theta = 2.0 * math.pi * i / num_views
        elev = 0.25 * math.sin(2.0 * theta)
        eye = np.array([radius * math.sin(theta), elev, radius * math.cos(theta)], dtype=np.float32)
        poses.append(look_at(eye))
    focal = 0.9 * width
    intrinsics = np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return CameraSet(np.stack(poses), intrinsics, width, height)


def rays_for_camera(pose: np.ndarray, intrinsics: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    dirs = np.stack(
        [(xs - intrinsics[0, 2]) / intrinsics[0, 0], -(ys - intrinsics[1, 2]) / intrinsics[1, 1], np.ones_like(xs)],
        axis=-1,
    )
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    world_dirs = dirs @ pose[:3, :3].T
    origins = np.broadcast_to(pose[:3, 3], world_dirs.shape)
    return origins.astype(np.float32), world_dirs.astype(np.float32)


def _sphere_intersect(origins: np.ndarray, dirs: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    oc = origins - center
    b = np.sum(oc * dirs, axis=-1)
    c = np.sum(oc * oc, axis=-1) - radius * radius
    disc = b * b - c
    t = np.full(disc.shape, np.inf, dtype=np.float32)
    valid = disc > 0
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    near = -b - sqrt_disc
    far = -b + sqrt_disc
    hit = valid & (far > 0)
    t[hit] = np.where(near[hit] > 0, near[hit], far[hit])
    return t


def render_view(pose: np.ndarray, intrinsics: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    origins, dirs = rays_for_camera(pose, intrinsics, width, height)
    t1 = _sphere_intersect(origins, dirs, np.array([-0.35, 0.0, 0.0], dtype=np.float32), 0.82)
    t2 = _sphere_intersect(origins, dirs, np.array([0.55, -0.08, 0.18], dtype=np.float32), 0.48)
    depth = np.minimum(t1, t2)
    hit = np.isfinite(depth)
    pts = origins + dirs * np.where(hit, depth, 0.0)[..., None]
    normals1 = (pts - np.array([-0.35, 0.0, 0.0], dtype=np.float32)) / 0.82
    normals2 = (pts - np.array([0.55, -0.08, 0.18], dtype=np.float32)) / 0.48
    normals = np.where((t1 <= t2)[..., None], normals1, normals2)
    color = 0.5 + 0.5 * normals
    stripes = 0.18 * np.sin(10.0 * pts[..., 0] + 6.0 * pts[..., 1])[..., None]
    light = np.clip(np.sum(normals * np.array([0.25, 0.75, 0.55], dtype=np.float32), axis=-1, keepdims=True), 0.0, 1.0)
    image = np.clip(color * (0.45 + 0.55 * light) + stripes, 0.0, 1.0)
    background = np.stack(
        [0.05 + 0.08 * dirs[..., 1], 0.07 + 0.04 * dirs[..., 0], 0.11 + 0.05 * dirs[..., 2]], axis=-1
    )
    image = np.where(hit[..., None], image, background)
    depth = np.where(hit, depth, 6.0).astype(np.float32)
    return image.astype(np.float32), depth


def render_dataset(num_views: int = 24, width: int = 48, height: int = 48) -> dict[str, np.ndarray]:
    cameras = orbit_cameras(num_views, width, height)
    images, depths = [], []
    for pose in cameras.poses:
        image, depth = render_view(pose, cameras.intrinsics, width, height)
        images.append(image)
        depths.append(depth)
    return {
        "images": np.stack(images),
        "depths": np.stack(depths),
        "poses": cameras.poses,
        "intrinsics": cameras.intrinsics,
        "width": np.array(width),
        "height": np.array(height),
    }
