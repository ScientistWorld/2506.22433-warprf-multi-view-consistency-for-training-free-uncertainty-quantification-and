"""Evaluation code for WarpRF artifacts.

This module deliberately does not import from method/. It scores saved outputs
so future methods can be evaluated by writing the same artifact fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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


def evaluate_artifact(path: Path, experiment_prefix: str = "tiny_nerf") -> dict:
    data = np.load(path)
    uncertainty = data["uncertainty"]
    depth_error = data["depth_error"]
    valid_mask = data["valid_depth_mask"] if "valid_depth_mask" in data.files else None
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
    oracle_errors = data["candidate_oracle_errors"]
    selected = int(np.argmax(scores))
    selected_oracle_error = float(oracle_errors[selected])
    random_oracle_error = float(np.mean(oracle_errors))
    best_oracle_error = float(np.max(oracle_errors))
    selection_regret = best_oracle_error - selected_oracle_error

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
                        "selection_regret": best_oracle_error - random_oracle_error,
                        "view_budget": int(data["source_view_count"] + 1),
                    },
                    "warprf": {
                        "selected_candidate_error": selected_oracle_error,
                        "selection_regret": selection_regret,
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
