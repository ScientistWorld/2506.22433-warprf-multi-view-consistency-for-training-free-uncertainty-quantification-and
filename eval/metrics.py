"""Evaluation code for WarpRF artifacts.

This module deliberately does not import from method/. It scores saved outputs
so future methods can be evaluated by writing the same artifact fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.synthetic_radiance import render_dataset


def _scalar_int(data: np.lib.npyio.NpzFile, key: str, default: int | None = None) -> int:
    if key not in data.files:
        if default is None:
            raise KeyError(f"artifact missing required scalar field '{key}'")
        return default
    return int(np.asarray(data[key]).reshape(()))


def _depth_errors_from_artifact(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute held-out depth errors from method predictions and data/.

    New artifacts write rendered depths plus view indices. The evaluator
    reconstructs the analytic benchmark labels itself so methods cannot
    directly supply their own error maps. Legacy artifacts from earlier runs are
    still readable, but new submissions should not rely on those fields.
    """
    if "test_rendered_depths" in data.files:
        test_depths = data["test_rendered_depths"].astype(np.float32)
        test_indices = data["test_indices"].astype(np.int64)
        height, width = test_depths.shape[-2:]
        num_views = _scalar_int(data, "num_views", int(np.max(test_indices)) + 1)
        dataset = render_dataset(num_views=num_views, width=width, height=height)
        gt_depths = dataset["depths"][test_indices]
        valid_mask = gt_depths < 5.95
        return np.abs(test_depths - gt_depths), valid_mask

    if "depth_error" in data.files:
        valid_mask = data["valid_depth_mask"] if "valid_depth_mask" in data.files else None
        return data["depth_error"], valid_mask

    raise KeyError("artifact must contain test_rendered_depths or legacy depth_error")


def _candidate_oracle_errors_from_artifact(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """Score candidate views using evaluator-computed ground-truth RGB error."""
    if "candidate_rendered_images" in data.files:
        rendered = data["candidate_rendered_images"].astype(np.float32)
        candidate_indices = data["candidate_indices"].astype(np.int64)
        height, width = rendered.shape[1:3]
        num_views = _scalar_int(data, "num_views", int(np.max(candidate_indices)) + 1)
        dataset = render_dataset(num_views=num_views, width=width, height=height)
        return np.mean(np.abs(rendered - dataset["images"][candidate_indices]), axis=(1, 2, 3)).astype(np.float32)

    if "candidate_oracle_errors" in data.files:
        return data["candidate_oracle_errors"]

    raise KeyError("artifact must contain candidate_rendered_images or legacy candidate_oracle_errors")


def ause(uncertainty: np.ndarray, error: np.ndarray, valid_mask: np.ndarray | None = None, fractions: int = 101) -> float:
    u = uncertainty.reshape(-1)
    e = error.reshape(-1)
    valid = np.isfinite(u) & np.isfinite(e)
    if valid_mask is not None:
        valid &= valid_mask.reshape(-1).astype(bool)
    u = u[valid]
    e = e[valid]
    if u.size < 2:
        return float("nan")
    order_u = np.argsort(-u)
    order_e = np.argsort(-e)
    curve_u, curve_e = [], []
    for frac in np.linspace(0.0, 0.99, fractions):
        keep = max(1, int(round((1.0 - frac) * u.size)))
        curve_u.append(float(np.mean(e[order_u[-keep:]])))
        curve_e.append(float(np.mean(e[order_e[-keep:]])))
    norm = max(curve_u[0], 1e-8)
    y = (np.array(curve_u) - np.array(curve_e)) / norm
    dx = 0.99 / (fractions - 1)
    return float(np.sum((y[:-1] + y[1:]) * 0.5 * dx))


def spearman_rank(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid_mask is not None:
        valid &= valid_mask.reshape(-1).astype(bool)
    a = a[valid]
    b = b[valid]
    ar = np.argsort(np.argsort(a)).astype(np.float64)
    br = np.argsort(np.argsort(b)).astype(np.float64)
    ar -= ar.mean()
    br -= br.mean()
    denom = np.sqrt(np.sum(ar * ar) * np.sum(br * br))
    return float(np.sum(ar * br) / max(denom, 1e-12))


def top_half_error_lift(scores: np.ndarray, oracle_errors: np.ndarray) -> float:
    if scores.size == 0:
        return float("nan")
    count = max(1, scores.size // 2)
    top = np.argsort(-scores)[:count]
    scale = max(float(np.mean(oracle_errors)), 1e-8)
    return float(np.mean(oracle_errors[top]) / scale)


def evaluate_artifact(path: Path, experiment_prefix: str = "tiny_nerf") -> dict:
    data = np.load(path)
    uncertainty = data["uncertainty"]
    depth_error, valid_mask = _depth_errors_from_artifact(data)
    warprf_ause = ause(uncertainty, depth_error, valid_mask)
    # Deterministic artifact-shaped baseline; evaluation still reads only the
    # submitted method output and does not rerun any rendering or warping code.
    random_uncertainty = np.mod(np.arange(uncertainty.size, dtype=np.float32) * 0.61803398875, 1.0).reshape(
        uncertainty.shape
    )
    random_ause = ause(random_uncertainty, depth_error, valid_mask)
    mean_error = float(np.mean(depth_error[valid_mask.astype(bool)] if valid_mask is not None else depth_error))
    rank_corr = spearman_rank(uncertainty, depth_error, valid_mask)

    scores = data["candidate_scores"]
    oracle_errors = _candidate_oracle_errors_from_artifact(data)
    selected = int(np.argmax(scores))
    selected_oracle_error = float(oracle_errors[selected])
    random_oracle_error = float(np.mean(oracle_errors))
    best_oracle_error = float(np.max(oracle_errors))
    selection_regret = best_oracle_error - selected_oracle_error
    oracle_scale = max(random_oracle_error, 1e-8)
    random_scores = np.mod(np.arange(scores.size, dtype=np.float32) * 0.61803398875, 1.0)

    return {
        "experiments": {
            f"{experiment_prefix}_depth_uncertainty": {
                "results": {
                    "random_uncertainty": {
                        "ause": random_ause,
                        "spearman_depth_error": spearman_rank(random_uncertainty, depth_error, valid_mask),
                        "mean_depth_error": mean_error,
                        "source_view_count": int(data["source_view_count"]),
                        "used_ground_truth_for_uncertainty": 0,
                    },
                    "warprf": {
                        "ause": warprf_ause,
                        "spearman_depth_error": rank_corr,
                        "mean_depth_error": mean_error,
                        "source_view_count": int(data["source_view_count"]),
                        "used_ground_truth_for_uncertainty": int(data["used_ground_truth_for_uncertainty"]),
                    },
                }
            },
            f"{experiment_prefix}_active_view_proxy": {
                "results": {
                    "random_view": {
                        "selected_candidate_error": random_oracle_error,
                        "selected_error_lift": 1.0,
                        "selection_regret": best_oracle_error - random_oracle_error,
                        "selection_regret_fraction": (best_oracle_error - random_oracle_error) / oracle_scale,
                        "candidate_score_spearman": spearman_rank(random_scores, oracle_errors),
                        "top_half_error_lift": top_half_error_lift(random_scores, oracle_errors),
                        "view_budget": int(data["source_view_count"] + 1),
                    },
                    "warprf": {
                        "selected_candidate_error": selected_oracle_error,
                        "selected_error_lift": selected_oracle_error / oracle_scale,
                        "selection_regret": selection_regret,
                        "selection_regret_fraction": selection_regret / oracle_scale,
                        "candidate_score_spearman": spearman_rank(scores, oracle_errors),
                        "top_half_error_lift": top_half_error_lift(scores, oracle_errors),
                        "view_budget": int(data["source_view_count"] + 1),
                    },
                }
            },
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="outputs/warprf/warprf_outputs.npz")
    parser.add_argument("--scores", default="scoring/scores.json")
    parser.add_argument("--experiment-prefix", default="tiny_nerf")
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()
    results = evaluate_artifact(Path(args.artifact), args.experiment_prefix)
    out = Path(args.scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.merge_existing and out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        existing.setdefault("experiments", {}).update(results["experiments"])
        results = existing
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
