"""Hardening audit: score the reference, variants and shortcuts through both evaluators.

Every case is registered exactly like a future method (``<dir>/<name>/predict.json``)
and scored by the unmodified ``eval/train/evaluate.py`` and ``eval/test/evaluate.py``.
Writes ``redteam/hardening/results.json`` and ``redteam/hardening/results.md``.

Usage: python redteam/hardening/run_checks.py [--assets-train DIR --assets-test DIR --expected N N]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
sys.path.insert(0, str(WORKSPACE))

from redteam.hardening.variants import SHORTCUTS, VARIANTS  # noqa: E402

REFERENCE = "warprf"
PRIMARY = {"depth_uncertainty_sparse": "ause", "depth_uncertainty_dense": "ause", "active_view_selection": "final_psnr"}


def register(root: Path, name: str, command: list[str]) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "predict.json").write_text(json.dumps({"command": command}) + "\n")


def run_eval(split: str, methods: Path, scores: Path, details: Path | None, extra: list[str]) -> int:
    cmd = [sys.executable, str(WORKSPACE / "eval" / split / "evaluate.py"), "--methods-dir", str(methods), "--scores", str(scores)]
    if details is not None:
        cmd += ["--details", str(details)]
    t0 = time.time()
    proc = subprocess.run(cmd + extra, cwd=str(WORKSPACE), capture_output=True, text=True)
    print(f"[checks] {split}: evaluator rc={proc.returncode} in {time.time() - t0:.0f}s", flush=True)
    if proc.stderr.strip():
        print(proc.stderr.strip()[-4000:], flush=True)
    return proc.returncode


def cohort_rank(scores: dict, reference: dict, names: list[str]) -> dict:
    exps = reference["experiments"]
    res = scores["experiments"]
    ranks = {}
    stds = {}
    for exp, spec in exps.items():
        if exp not in res:
            continue
        for metric in spec["metrics"]:
            vals = [res[exp]["results"][n][metric] for n in names if n in res[exp]["results"]]
            stds[(exp, metric)] = float(np.std(vals)) if len(vals) > 1 else 0.0
    for n in names:
        total, terms = 0.0, {}
        for exp, spec in exps.items():
            if exp not in res or n not in res[exp]["results"]:
                continue
            for metric, mdef in spec["metrics"].items():
                std = stds[(exp, metric)]
                if std <= 0:
                    continue
                direction = 1.0 if mdef["higher_is_better"] else -1.0
                val = res[exp]["results"][n][metric]
                base = res[exp]["results"][REFERENCE][metric]
                term = spec["weight"] * abs(mdef["coefficient"]) * direction * (val - base) / std
                terms[f"{exp}/{metric}"] = term
                total += term
        ranks[n] = {"rank": total, "terms": terms}
    return ranks


def paired(details: dict, name: str, n_boot: int = 2000) -> dict:
    """Paired per-scene improvement of ``name`` over the reference, with a scene bootstrap."""
    rng = np.random.default_rng(0)
    out = {}
    ref, var = details.get(REFERENCE, {}), details.get(name, {})
    if "depth" not in ref or "depth" not in var:
        return out
    for exp, setting in (("depth_uncertainty_sparse", "sparse"), ("depth_uncertainty_dense", "dense")):
        # per scene: mean over its target views of (ref AUSE - variant AUSE): positive = variant better
        diffs = np.array([np.mean([r[0] - v[0] for r, v in zip(rs, vs)]) for rs, vs in zip(ref["depth"][setting], var["depth"][setting])])
        out[exp] = _boot(diffs, rng, n_boot)
    diffs = np.array([v["psnr"] - r["psnr"] for r, v in zip(ref["active"], var["active"])])
    out["active_view_selection"] = _boot(diffs, rng, n_boot)
    return out


def _boot(diffs: np.ndarray, rng, n_boot: int) -> dict:
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    means = diffs[idx].mean(axis=1)
    return {
        "mean_improvement": float(diffs.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "scenes_better": int((diffs > 1e-12).sum()),
        "scenes_worse": int((diffs < -1e-12).sum()),
    }


def diagnostics(assets: Path, choices: dict) -> dict:
    """Bounds of the acquisition table, oracle selectors, and the percentile of each method's chosen pair.

    ``oracle_uncertainty`` acquires, at each step, the candidate whose current rendering is
    farthest from its (hidden) photograph: the selection rule every uncertainty method
    approximates, with a perfect uncertainty. ``greedy_oracle`` acquires the candidate with
    the best held-out quality after one step, then the best second view.
    """
    manifest = json.loads((assets / "manifest.json").read_text())
    tables, unc_oracle, greedy = [], [], []
    for name in manifest["scenes"]:
        with np.load(assets / name / "answers.npz") as f:
            a = {k: f[k] for k in f.files}
        with np.load(assets / name / "inputs.npz") as f:
            x = {k: f[k] for k in ("sparse_candidate_rgb", "step2_candidate_rgb")}
        P = a["final_psnr"].astype(np.float64)
        tables.append((P, a["step1_psnr"].astype(np.float64), a["step0_quality"].astype(np.float64)))
        photo = a["candidate_images"].astype(np.float64)
        e1 = np.abs(x["sparse_candidate_rgb"].astype(np.float64) - photo).sum(axis=(1, 2, 3))
        f1 = int(np.argmax(e1))
        e2 = np.abs(x["step2_candidate_rgb"][f1].astype(np.float64) - photo).sum(axis=(1, 2, 3))
        e2[f1] = -np.inf
        unc_oracle.append(P[f1, int(np.argmax(e2))])
        g1 = int(np.argmax(a["step1_psnr"]))
        row = P[g1].copy()
        row[g1] = -np.inf
        greedy.append(float(row.max()))
    off = ~np.eye(tables[0][0].shape[0], dtype=bool)
    out = {
        "final_psnr_best_pair_mean": float(np.mean([t[0][off].max() for t in tables])),
        "final_psnr_worst_pair_mean": float(np.mean([t[0][off].min() for t in tables])),
        "final_psnr_random_mean": float(np.mean([t[0][off].mean() for t in tables])),
        "final_psnr_oracle_uncertainty_mean": float(np.mean(unc_oracle)),
        "final_psnr_greedy_oracle_mean": float(np.mean(greedy)),
        "final_psnr_within_scene_std_mean": float(np.mean([t[0][off].std() for t in tables])),
        "step1_psnr_best_mean": float(np.mean([t[1].max() for t in tables])),
        "step1_psnr_mean": float(np.mean([t[1].mean() for t in tables])),
        "step0_psnr_mean": float(np.mean([t[2][0] for t in tables])),
        "chosen_pair_percentile": {},
        "per_scene": {"oracle_uncertainty": [float(v) for v in unc_oracle], "greedy_oracle": greedy,
                      "best_pair": [float(t[0][off].max()) for t in tables]},
    }
    for name, ch in choices.items():
        if not ch:
            continue
        pct = []
        for (a_, b_), t in zip(ch, tables):
            vals = t[0][off]
            pct.append(float((vals < t[0][a_, b_]).mean()))
        out["chosen_pair_percentile"][name] = float(np.mean(pct))
    return out


def reference_vs_baselines(details: dict, diag: dict) -> dict:
    """Paired per-scene gain of the reference over each baseline and of the oracles over the reference."""
    rng = np.random.default_rng(1)
    ref = details[REFERENCE]
    base = details["baselines"]
    out = {}
    for exp, setting in (("depth_uncertainty_sparse", "sparse"), ("depth_uncertainty_dense", "dense")):
        diffs = np.array([np.mean([b[0] - r[0] for r, b in zip(rs, bs)])
                          for rs, bs in zip(ref["depth"][setting], base["random_uncertainty"][setting])])
        out[f"{exp}: reference AUSE gain over random_uncertainty"] = _boot(diffs, rng, 2000)
    ref_psnr = np.array([r["psnr"] for r in ref["active"]])
    out["active: reference PSNR gain over random_view"] = _boot(ref_psnr - np.array([r["psnr"] for r in base["random_view"]]), rng, 2000)
    out["active: reference PSNR gain over farthest_view"] = _boot(ref_psnr - np.array([r["psnr"] for r in base["farthest_view"]["active"]]), rng, 2000)
    for key in ("oracle_uncertainty", "greedy_oracle", "best_pair"):
        out[f"active: {key} PSNR gain over reference"] = _boot(np.array(diag["per_scene"][key]) - ref_psnr, rng, 2000)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--assets-train", default="")
    p.add_argument("--assets-test", default="")
    p.add_argument("--expected", nargs=2, type=int, default=None)
    p.add_argument("--work", default=str(WORKSPACE / "tmp" / "hardening"))
    p.add_argument("--out", default=str(HERE))
    args = p.parse_args()
    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    extra = {"train": [], "test": []}
    if args.assets_train:
        extra["train"] += ["--assets", args.assets_train]
    if args.assets_test:
        extra["test"] += ["--assets", args.assets_test]
    if args.expected:
        extra["train"] += ["--expected-scenes", str(args.expected[0])]
        extra["test"] += ["--expected-scenes", str(args.expected[1])]
    reference = json.loads((WORKSPACE / "scoring" / "reference.json").read_text())

    methods = work / "methods"
    register(methods, REFERENCE, [sys.executable, "-m", "method.predict"])
    for name in list(VARIANTS) + list(SHORTCUTS):
        register(methods, name, [sys.executable, "-m", "redteam.hardening.variants", "--variant", name])
    (methods / "broken_registration").mkdir()
    (methods / "broken_registration" / "predict.json").write_text('{"command": "not a list"}')

    report = {"splits": {}, "special": {}}
    for split in ("train", "test"):
        scores, details = work / f"scores_{split}.json", work / f"details_{split}.json"
        rc = run_eval(split, methods, scores, details, extra[split])
        if rc != 0:
            raise SystemExit(f"{split} evaluator failed")
        sc = json.loads(scores.read_text())
        det = json.loads(details.read_text())
        names = sorted(sc["_methods"])
        report["splits"][split] = {
            "scores": sc["experiments"],
            "status": sc["_methods"],
            "cohort_rank": cohort_rank(sc, reference, names),
            "paired_vs_reference": {n: paired(det, n) for n in names if n != REFERENCE},
            "choices": {n: det[n].get("choices") for n in names if isinstance(det.get(n), dict)},
            "baseline_choices": det["baselines"]["farthest_view"]["choices"],
        }
        assets = Path(args.assets_train or WORKSPACE / "data" / "train_assets") if split == "train" else Path(
            args.assets_test or WORKSPACE / "eval" / "test" / "assets")
        chosen = dict(report["splits"][split]["choices"])
        chosen["farthest_view"] = report["splits"][split]["baseline_choices"]
        report["splits"][split]["diagnostics"] = diagnostics(assets, chosen)
        report["splits"][split]["reference_vs_baselines"] = reference_vs_baselines(det, report["splits"][split]["diagnostics"])
        report["splits"][split]["isolation"] = sc.get("_isolation")

        # Special cases on the same evaluator.
        empty = work / f"empty_{split}"
        empty.mkdir()
        forged = work / f"forged_{split}.json"
        forged.write_text(json.dumps({"experiments": {"active_view_selection": {"results": {"forged": {"final_psnr": 1e9}}}}}))
        rc = run_eval(split, empty, forged, None, extra[split])
        after = json.loads(forged.read_text()) if forged.exists() else None
        no_method = after["experiments"]["active_view_selection"]["results"].get("no_method_registered") if after else None
        stale = work / f"stale_{split}.json"
        stale.write_text('{"forged": true}')
        rc_stale = run_eval(split, empty, stale, None, ["--assets", str(work / "missing_assets")])
        report["special"][split] = {
            "empty_registry_rc": rc,
            "forged_scores_overwritten": bool(after) and "forged" not in after["experiments"]["active_view_selection"]["results"],
            "no_method_entry": no_method,
            "failed_evaluator_rc": rc_stale,
            "stale_scores_removed": not stale.exists(),
        }

    esc = work / "escape_report.jsonl"
    routes = {}
    if esc.is_file():
        for line in esc.read_text().splitlines():
            r = json.loads(line)
            key = "test" if r["scene"].startswith("test_") else "train"
            routes.setdefault(key, {}).setdefault(str(r["route"]), 0)
            routes[key][str(r["route"])] += 1
            routes.setdefault(f"{key}_example_attempts", r["tried"])
    report["escape_routes"] = routes

    out = Path(args.out)
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["| case | split | sparse AUSE | dense AUSE | final PSNR | final SSIM | final depth MAE | cohort rank | status |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for split, rep in report["splits"].items():
        res = rep["scores"]
        rows = sorted(rep["cohort_rank"].items(), key=lambda kv: -kv[1]["rank"])
        for name, rk in rows:
            g = lambda e, m: res[e]["results"][name][m]
            lines.append(
                f"| {name} | {split} | {g('depth_uncertainty_sparse', 'ause'):.4f} | {g('depth_uncertainty_dense', 'ause'):.4f} | "
                f"{g('active_view_selection', 'final_psnr'):.3f} | {g('active_view_selection', 'final_ssim'):.4f} | "
                f"{g('active_view_selection', 'final_depth_mae'):.4f} | {rk['rank']:+.3f} | {rep['status'][name]['status']} |"
            )
        for bname in ("random_uncertainty",):
            lines.append(
                f"| {bname} (eval) | {split} | {res['depth_uncertainty_sparse']['results'][bname]['ause']:.4f} | "
                f"{res['depth_uncertainty_dense']['results'][bname]['ause']:.4f} | | | | | baseline |"
            )
        for bname in ("random_view", "farthest_view"):
            r = res["active_view_selection"]["results"][bname]
            lines.append(f"| {bname} (eval) | {split} | | | {r['final_psnr']:.3f} | {r['final_ssim']:.4f} | {r['final_depth_mae']:.4f} | | baseline |")
    lines.append("")
    for split in ("test", "train"):
        lines.append(f"Paired improvement over the reference ({split} split; AUSE: reference minus variant, PSNR: variant minus reference; 95% scene-bootstrap CI):")
        lines.append("")
        lines.append("| case | sparse AUSE gain [CI] | dense AUSE gain [CI] | PSNR gain dB [CI] |")
        lines.append("|---|---|---|---|")
        for name, pv in sorted(report["splits"][split]["paired_vs_reference"].items()):
            if not pv:
                continue
            f = lambda e: f"{pv[e]['mean_improvement']:+.4f} [{pv[e]['ci95'][0]:+.4f}, {pv[e]['ci95'][1]:+.4f}]"
            lines.append(f"| {name} | {f('depth_uncertainty_sparse')} | {f('depth_uncertainty_dense')} | {f('active_view_selection')} |")
        lines.append("")
    for split, rep in report["splits"].items():
        lines.append(f"Reference vs baselines and oracles ({split}; mean [95% CI], scenes better/worse):")
        lines.append("")
        for k, v in rep["reference_vs_baselines"].items():
            lines.append(f"- {k}: {v['mean_improvement']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] ({v['scenes_better']}/{v['scenes_worse']})")
        lines.append("")
        diag = {k: v for k, v in rep["diagnostics"].items() if k != "per_scene"}
        lines.append(f"Diagnostics ({split}, isolation={rep['isolation']}): " + json.dumps(diag))
        lines.append("")
    lines.append("Escape routes taken by shortcut_escape (None = no route worked): " + json.dumps(report["escape_routes"]))
    lines.append("")
    lines.append("Special cases: " + json.dumps(report["special"]))
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
