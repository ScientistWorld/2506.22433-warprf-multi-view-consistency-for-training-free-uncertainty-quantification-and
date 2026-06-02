"""Held-out test-slice scorer for saved WarpRF-style artifacts.

This evaluator is method-agnostic: it reads prediction artifacts from disk and
never imports from method/. The split is over held-out view positions and
candidate-view positions, using the odd positions for the hidden test slice.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from data.synthetic_radiance import render_dataset


SLICE_NAME = "test"
SLICE_PHASE = 1


def _scalar_int(data: np.lib.npyio.NpzFile, key: str, default: int | None = None) -> int:
    if key not in data.files:
        if default is None:
            raise KeyError(f"artifact missing required scalar field '{key}'")
        return default
    return int(np.asarray(data[key]).reshape(()))


def _split_mask(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(length, dtype=bool)
    mask = (np.arange(length) % 2) == SLICE_PHASE
    if not np.any(mask):
        mask[:] = True
    return mask


def _load_metadata(artifact: Path) -> dict:
    meta_path = artifact.parent / "metadata.json"
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _safe_name(value: str) -> str:
    value = value.strip()
    if value.startswith("tiny_nerf_"):
        value = value[len("tiny_nerf_") :]
    value = re.sub(r"_(dense|shifted|sparse)$", "", value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "method"


def _method_name(artifact: Path, metadata: dict) -> str:
    method = metadata.get("method")
    if isinstance(method, str) and method.strip():
        return _safe_name(method)
    return _safe_name(artifact.parent.name)


def _experiment_prefix(metadata: dict, fallback: str | None) -> str:
    if fallback:
        return fallback
    if metadata.get("split_mode") == "dense":
        return "tiny_nerf_dense"
    if int(metadata.get("split_offset", 0)) == 1:
        return "tiny_nerf_shifted"
    return "tiny_nerf"


def _discover_artifacts(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"artifact path does not exist: {path}")
    direct = path / "warprf_outputs.npz"
    if direct.is_file():
        return [direct]
    artifacts = sorted(path.glob("*/warprf_outputs.npz"))
    if artifacts:
        return artifacts
    return sorted(path.glob("**/*.npz"))


def _load_expected_methods(path: Path) -> dict[str, set[str]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("experiments", data)
    expected: dict[str, set[str]] = {}
    for exp_name, exp_data in raw.items():
        if exp_name.startswith("_") or not isinstance(exp_data, dict):
            continue
        methods = exp_data.get("results", exp_data)
        if isinstance(methods, dict):
            expected[exp_name] = {name for name in methods if not name.startswith("_")}
    return expected


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
    if a.size < 2:
        return float("nan")
    ar = np.argsort(np.argsort(a)).astype(np.float64)
    br = np.argsort(np.argsort(b)).astype(np.float64)
    ar -= ar.mean()
    br -= br.mean()
    denom = np.sqrt(np.sum(ar * ar) * np.sum(br * br))
    return float(np.sum(ar * br) / max(denom, 1e-12))


def top_half_error_lift(scores: np.ndarray, oracle_errors: np.ndarray) -> float:
    scores = scores.reshape(-1)
    oracle_errors = oracle_errors.reshape(-1)
    valid = np.isfinite(scores) & np.isfinite(oracle_errors)
    scores = scores[valid]
    oracle_errors = oracle_errors[valid]
    if scores.size == 0:
        return float("nan")
    keep = max(1, scores.size // 2)
    top = np.argsort(scores)[-keep:]
    return float(np.mean(oracle_errors[top]) / max(float(np.mean(oracle_errors)), 1e-8))


def _depth_arrays(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    uncertainty = data["uncertainty"].astype(np.float32)
    if "test_rendered_depths" in data.files:
        rendered = data["test_rendered_depths"].astype(np.float32)
        indices = data["test_indices"].astype(np.int64)
        mask = _split_mask(len(indices))
        rendered = rendered[mask]
        uncertainty = uncertainty[mask]
        indices = indices[mask]
        height, width = rendered.shape[-2:]
        num_views = _scalar_int(data, "num_views", int(np.max(indices)) + 1)
        dataset = render_dataset(num_views=num_views, width=width, height=height)
        gt_depths = dataset["depths"][indices]
        valid_mask = gt_depths < 5.95
        return uncertainty, np.abs(rendered - gt_depths), valid_mask
    if "depth_error" in data.files:
        error = data["depth_error"].astype(np.float32)
        if error.shape[:1] == uncertainty.shape[:1]:
            mask = _split_mask(error.shape[0])
            error = error[mask]
            uncertainty = uncertainty[mask]
            valid_mask = data["valid_depth_mask"][mask] if "valid_depth_mask" in data.files else None
        else:
            valid_mask = data["valid_depth_mask"] if "valid_depth_mask" in data.files else None
        return uncertainty, error, valid_mask
    raise KeyError("artifact must contain test_rendered_depths or legacy depth_error")


def _candidate_arrays(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    scores = data["candidate_scores"].astype(np.float32)
    if "candidate_rendered_images" in data.files:
        rendered = data["candidate_rendered_images"].astype(np.float32)
        indices = data["candidate_indices"].astype(np.int64)
        mask = _split_mask(len(indices))
        rendered = rendered[mask]
        scores = scores[mask]
        indices = indices[mask]
        height, width = rendered.shape[1:3]
        num_views = _scalar_int(data, "num_views", int(np.max(indices)) + 1)
        dataset = render_dataset(num_views=num_views, width=width, height=height)
        oracle_errors = np.mean(np.abs(rendered - dataset["images"][indices]), axis=(1, 2, 3)).astype(np.float32)
        return scores, oracle_errors
    if "candidate_oracle_errors" in data.files:
        oracle_errors = data["candidate_oracle_errors"].astype(np.float32)
        if len(oracle_errors) == len(scores):
            mask = _split_mask(len(scores))
            scores = scores[mask]
            oracle_errors = oracle_errors[mask]
        return scores, oracle_errors
    raise KeyError("artifact must contain candidate_rendered_images or legacy candidate_oracle_errors")


def _score_artifact(artifact: Path, prefix: str, method: str) -> dict:
    data = np.load(artifact)
    uncertainty, depth_error, valid_mask = _depth_arrays(data)
    random_uncertainty = np.mod(
        np.arange(uncertainty.size, dtype=np.float32) * 0.61803398875,
        1.0,
    ).reshape(uncertainty.shape)
    if valid_mask is not None:
        valid_errors = depth_error[valid_mask.astype(bool)]
    else:
        valid_errors = depth_error
    mean_error = float(np.mean(valid_errors))

    scores, oracle_errors = _candidate_arrays(data)
    selected = int(np.argmax(scores))
    selected_oracle_error = float(oracle_errors[selected])
    random_oracle_error = float(np.mean(oracle_errors))
    best_oracle_error = float(np.max(oracle_errors))
    random_scores = np.mod(np.arange(scores.size, dtype=np.float32) * 0.75487766625, 1.0).reshape(scores.shape)

    return {
        f"{prefix}_depth_uncertainty": {
            "random_uncertainty": {
                "ause": ause(random_uncertainty, depth_error, valid_mask),
                "spearman_depth_error": spearman_rank(random_uncertainty, depth_error, valid_mask),
                "mean_depth_error": mean_error,
                "source_view_count": _scalar_int(data, "source_view_count"),
                "used_ground_truth_for_uncertainty": 0,
            },
            method: {
                "ause": ause(uncertainty, depth_error, valid_mask),
                "spearman_depth_error": spearman_rank(uncertainty, depth_error, valid_mask),
                "mean_depth_error": mean_error,
                "source_view_count": _scalar_int(data, "source_view_count"),
                "used_ground_truth_for_uncertainty": _scalar_int(data, "used_ground_truth_for_uncertainty", 0),
            },
        },
        f"{prefix}_active_view_proxy": {
            "random_view": {
                "selected_candidate_error": random_oracle_error,
                "selection_regret": best_oracle_error - random_oracle_error,
                "candidate_score_spearman": spearman_rank(random_scores, oracle_errors),
                "top_half_error_lift": top_half_error_lift(random_scores, oracle_errors),
                "view_budget": _scalar_int(data, "source_view_count") + 1,
            },
            method: {
                "selected_candidate_error": selected_oracle_error,
                "selection_regret": best_oracle_error - selected_oracle_error,
                "candidate_score_spearman": spearman_rank(scores, oracle_errors),
                "top_half_error_lift": top_half_error_lift(scores, oracle_errors),
                "view_budget": _scalar_int(data, "source_view_count") + 1,
            },
        },
    }


def _filter_to_expected(experiments: dict[str, dict], expected: dict[str, set[str]]) -> dict[str, dict]:
    if not expected:
        return experiments
    filtered: dict[str, dict] = {}
    for exp_name, method_names in expected.items():
        if exp_name not in experiments:
            raise KeyError(f"test slice could not score train experiment '{exp_name}'")
        available = experiments[exp_name]["results"]
        missing = sorted(method_names - set(available))
        if missing:
            raise KeyError(f"test slice missing train method(s) for {exp_name}: {missing}")
        filtered[exp_name] = {
            "_slice": SLICE_NAME,
            "results": {name: available[name] for name in sorted(method_names)},
        }
    return filtered


def evaluate_artifacts(input_path: Path, experiment_prefix: str | None = None, expected_scores: Path | None = None) -> dict:
    artifacts = _discover_artifacts(input_path)
    if not artifacts:
        raise FileNotFoundError(f"no .npz artifacts found under {input_path}")

    experiments: dict[str, dict] = {}
    for artifact in artifacts:
        metadata = _load_metadata(artifact)
        prefix = _experiment_prefix(metadata, experiment_prefix)
        method = _method_name(artifact, metadata)
        scored = _score_artifact(artifact, prefix, method)
        for exp_name, methods in scored.items():
            exp = experiments.setdefault(exp_name, {"_slice": SLICE_NAME, "results": {}})
            for method_name, metrics in methods.items():
                if method_name.startswith("random_") and method_name in exp["results"]:
                    continue
                exp["results"][method_name] = metrics

    expected = _load_expected_methods(expected_scores) if expected_scores else {}
    experiments = _filter_to_expected(experiments, expected)
    return {"slice": SLICE_NAME, "experiments": experiments}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="outputs")
    parser.add_argument("--scores", default="scoring/scores_test.json")
    parser.add_argument("--experiment-prefix", default="")
    parser.add_argument("--merge-existing", action="store_true")
    parser.add_argument("--expected-scores", default="scoring/scores_train.json")
    args = parser.parse_args()

    expected_scores = Path(args.expected_scores) if args.expected_scores else None
    results = evaluate_artifacts(Path(args.artifact), args.experiment_prefix or None, expected_scores)
    out = Path(args.scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.merge_existing and out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        existing.setdefault("experiments", {}).update(results["experiments"])
        existing["slice"] = SLICE_NAME
        results = existing
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
