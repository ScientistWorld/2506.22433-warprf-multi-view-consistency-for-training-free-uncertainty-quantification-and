"""WarpRF uncertainty computation from rendered RGB-D views."""

from __future__ import annotations

import numpy as np


def _bilinear_sample(array: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = array.shape[:2]
    valid = (xs >= 0) & (ys >= 0) & (xs <= w - 1) & (ys <= h - 1)
    x0 = np.floor(np.clip(xs, 0, w - 1)).astype(np.int32)
    y0 = np.floor(np.clip(ys, 0, h - 1)).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = xs - x0
    wy = ys - y0
    if array.ndim == 2:
        wx_e, wy_e = wx, wy
    else:
        wx_e, wy_e = wx[..., None], wy[..., None]
    sampled = (
        array[y0, x0] * (1.0 - wx_e) * (1.0 - wy_e)
        + array[y0, x1] * wx_e * (1.0 - wy_e)
        + array[y1, x0] * (1.0 - wx_e) * wy_e
        + array[y1, x1] * wx_e * wy_e
    )
    return sampled, valid


def _target_to_source_grid(
    target_depth: np.ndarray,
    target_pose: np.ndarray,
    source_pose: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = target_depth.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x_cam = (xs - intrinsics[0, 2]) / intrinsics[0, 0] * target_depth
    y_cam = -(ys - intrinsics[1, 2]) / intrinsics[1, 1] * target_depth
    z_cam = target_depth
    points_target = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)
    points_world = points_target @ target_pose.T
    source_w2c = np.linalg.inv(source_pose)
    points_source = points_world @ source_w2c.T
    z_source = points_source[..., 2]
    u_source = intrinsics[0, 0] * (points_source[..., 0] / np.maximum(z_source, 1e-6)) + intrinsics[0, 2]
    v_source = -intrinsics[1, 1] * (points_source[..., 1] / np.maximum(z_source, 1e-6)) + intrinsics[1, 2]
    return u_source, v_source, z_source, points_world


def pixel_uncertainty_depth(
    target_depth: np.ndarray,
    target_pose: np.ndarray,
    source_depths: np.ndarray,
    source_poses: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Equation 8: average absolute target/source depth inconsistency."""
    accum = np.zeros_like(target_depth, dtype=np.float32)
    counts = np.zeros_like(target_depth, dtype=np.float32)
    target_w2c = np.linalg.inv(target_pose)
    for source_depth, source_pose in zip(source_depths, source_poses):
        u, v, z_source_from_target, _ = _target_to_source_grid(target_depth, target_pose, source_pose, intrinsics)
        sampled_source_depth, in_bounds = _bilinear_sample(source_depth, u, v)
        valid = in_bounds & (z_source_from_target > 0.0) & (sampled_source_depth > 0.0)
        x_source = (u - intrinsics[0, 2]) / intrinsics[0, 0] * sampled_source_depth
        y_source = -(v - intrinsics[1, 2]) / intrinsics[1, 1] * sampled_source_depth
        pts_source = np.stack([x_source, y_source, sampled_source_depth, np.ones_like(sampled_source_depth)], axis=-1)
        pts_world = pts_source @ source_pose.T
        pts_target = pts_world @ target_w2c.T
        reprojected_target_depth = pts_target[..., 2]
        diff = np.abs(target_depth - reprojected_target_depth)
        accum += np.where(valid & (reprojected_target_depth > 0.0), diff, 0.0)
        counts += valid.astype(np.float32)
    return accum / np.maximum(counts, 1.0)


def image_uncertainty(
    target_image: np.ndarray,
    target_depth: np.ndarray,
    target_pose: np.ndarray,
    source_images: np.ndarray,
    source_depths: np.ndarray,
    source_poses: np.ndarray,
    intrinsics: np.ndarray,
) -> float:
    """Equation 9: sum of minimum per-pixel RGB reprojection difference."""
    per_source = []
    for source_image, source_depth, source_pose in zip(source_images, source_depths, source_poses):
        u, v, z_source_from_target, _ = _target_to_source_grid(target_depth, target_pose, source_pose, intrinsics)
        warped_image, in_bounds = _bilinear_sample(source_image, u, v)
        sampled_source_depth, _ = _bilinear_sample(source_depth, u, v)
        valid = in_bounds & (z_source_from_target > 0.0) & (sampled_source_depth > 0.0)
        diff = np.mean(np.abs(target_image - warped_image), axis=-1)
        per_source.append(np.where(valid, diff, np.inf))
    stacked = np.stack(per_source)
    best = np.min(stacked, axis=0)
    finite = np.isfinite(best)
    return float(np.sum(np.where(finite, best, 0.0)) / max(np.sum(finite), 1))
