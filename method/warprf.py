"""WarpRF uncertainty (paper Sec. 3.2) from rendered RGB-D views and poses.

Follows the paper's equations and the authors' released implementation:

* Eq. 8 (pixel-wise depth uncertainty): back-project the target rendered depth,
  move the points into each source camera, project, look up the source
  rendered depth (nearest sample, border padding), lift it to 3D and move it
  back into the target camera; the uncertainty is the mean |z| disagreement over
  the source views whose projection has positive depth. Pixels with no such
  source take the maximum uncertainty of the view.
* Eq. 9 (image-level uncertainty): backward-warp every rendered source image
  into the target view with the target rendered depth (bilinear, border
  padding), take the per-pixel minimum over sources of the channel-summed
  absolute difference to the target rendering, and sum over all pixels.

Conventions: OpenCV cameras (x right, y down, z forward), camera-to-world 4x4
poses, depth = camera-space z, pixel (i, j) centred at (j + 0.5, i + 0.5).
"""

from __future__ import annotations

import numpy as np


def _pixel_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float64) + 0.5, np.arange(height, dtype=np.float64) + 0.5)
    return xs, ys


def backproject(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Camera-space points [H, W, 3] of a z-depth map."""
    h, w = depth.shape
    xs, ys = _pixel_grid(h, w)
    z = depth.astype(np.float64)
    x = (xs - intrinsics[0, 2]) / intrinsics[0, 0] * z
    y = (ys - intrinsics[1, 2]) / intrinsics[1, 1] * z
    return np.stack([x, y, z], axis=-1)


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def project(points: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points[..., 2]
    z_safe = np.where(np.abs(z) < 1e-8, 1e-8, z)
    u = intrinsics[0, 0] * points[..., 0] / z_safe + intrinsics[0, 2]
    v = intrinsics[1, 1] * points[..., 1] / z_safe + intrinsics[1, 2]
    return u, v


def _border_coords(u: np.ndarray, v: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.nan_to_num(u - 0.5, nan=0.0, posinf=width - 1.0, neginf=0.0)
    y = np.nan_to_num(v - 0.5, nan=0.0, posinf=height - 1.0, neginf=0.0)
    return np.clip(x, 0.0, width - 1.0), np.clip(y, 0.0, height - 1.0)


def sample_nearest(array: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = array.shape[:2]
    x, y = _border_coords(u, v, h, w)
    return array[np.rint(y).astype(np.int64), np.rint(x).astype(np.int64)]


def sample_bilinear(array: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = array.shape[:2]
    x, y = _border_coords(u, v, h, w)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = x - x0
    wy = y - y0
    if array.ndim == 3:
        wx = wx[..., None]
        wy = wy[..., None]
    return (
        array[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + array[y0, x1] * wx * (1.0 - wy)
        + array[y1, x0] * (1.0 - wx) * wy
        + array[y1, x1] * wx * wy
    )


def _relative(target_pose: np.ndarray, source_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(source <- target, target <- source) camera transforms."""
    src_from_tgt = np.linalg.inv(source_pose.astype(np.float64)) @ target_pose.astype(np.float64)
    tgt_from_src = np.linalg.inv(src_from_tgt)
    return src_from_tgt, tgt_from_src


def pixel_uncertainty(
    target_depth: np.ndarray,
    target_pose: np.ndarray,
    source_depths: np.ndarray,
    source_poses: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Eq. 8: mean absolute depth disagreement between target and warped sources."""
    k = intrinsics.astype(np.float64)
    target_points = backproject(target_depth, k)
    total = np.zeros(target_depth.shape, dtype=np.float64)
    count = np.zeros(target_depth.shape, dtype=np.float64)
    for source_depth, source_pose in zip(source_depths, source_poses):
        src_from_tgt, tgt_from_src = _relative(target_pose, source_pose)
        in_source = transform(target_points, src_from_tgt)
        u, v = project(in_source, k)
        source_in_target = transform(backproject(source_depth, k), tgt_from_src)
        corresponding = sample_nearest(source_in_target, u, v)
        visible = in_source[..., 2] > 0.0
        total += np.where(visible, np.abs(corresponding[..., 2] - target_points[..., 2]), 0.0)
        count += visible
    uncertainty = np.full(target_depth.shape, np.nan)
    np.divide(total, count, out=uncertainty, where=count > 0)
    finite = np.isfinite(uncertainty)
    fill = float(np.max(uncertainty[finite])) if np.any(finite) else 0.0
    return np.where(finite, uncertainty, fill).astype(np.float32)


def image_uncertainty(
    target_image: np.ndarray,
    target_depth: np.ndarray,
    target_pose: np.ndarray,
    source_images: np.ndarray,
    source_poses: np.ndarray,
    intrinsics: np.ndarray,
) -> float:
    """Eq. 9: sum over pixels of the minimum per-source photometric reprojection error."""
    k = intrinsics.astype(np.float64)
    target_points = backproject(target_depth, k)
    target = target_image.astype(np.float64)
    best = None
    for source_image, source_pose in zip(source_images, source_poses):
        src_from_tgt, _ = _relative(target_pose, source_pose)
        u, v = project(transform(target_points, src_from_tgt), k)
        warped = sample_bilinear(source_image.astype(np.float64), u, v)
        diff = np.abs(warped - target).sum(axis=-1)
        best = diff if best is None else np.minimum(best, diff)
    return float(best.sum())
