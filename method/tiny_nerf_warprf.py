"""Train a tiny NeRF-like radiance field and run WarpRF on its renderings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.synthetic_radiance import rays_for_camera, render_dataset
from method.warprf import image_uncertainty, pixel_uncertainty_depth


class TinyNeRF(nn.Module):
    def __init__(self, hidden: int = 96, freqs: int = 6):
        super().__init__()
        self.freqs = freqs
        in_dim = 3 * (1 + 2 * freqs) + 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, 4),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x]
        for i in range(self.freqs):
            f = 2.0**i * math.pi
            parts.extend([torch.sin(f * x), torch.cos(f * x)])
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor, dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(torch.cat([self.encode(x), dirs], dim=-1))
        rgb = torch.sigmoid(raw[..., :3])
        sigma = torch.nn.functional.softplus(raw[..., 3])
        return rgb, sigma


def render_rays(model: TinyNeRF, origins: torch.Tensor, dirs: torch.Tensor, samples: int = 48) -> tuple[torch.Tensor, torch.Tensor]:
    n = origins.shape[0]
    t_vals = torch.linspace(0.6, 5.5, samples, device=origins.device)
    pts = origins[:, None, :] + dirs[:, None, :] * t_vals[None, :, None]
    dirs_rep = dirs[:, None, :].expand(n, samples, 3)
    rgb, sigma = model(pts.reshape(-1, 3), dirs_rep.reshape(-1, 3))
    rgb = rgb.reshape(n, samples, 3)
    sigma = sigma.reshape(n, samples)
    deltas = torch.cat([t_vals[1:] - t_vals[:-1], torch.tensor([1e10], device=origins.device)])
    alpha = 1.0 - torch.exp(-sigma * deltas[None, :])
    trans = torch.cumprod(torch.cat([torch.ones((n, 1), device=origins.device), 1.0 - alpha + 1e-10], dim=1), dim=1)[:, :-1]
    weights = alpha * trans
    image = torch.sum(weights[..., None] * rgb, dim=1) + (1.0 - weights.sum(dim=1, keepdim=True)) * 0.08
    depth = torch.sum(weights * t_vals[None, :], dim=1) + (1.0 - weights.sum(dim=1)) * 5.8
    return image, depth


def _flatten_rays(dataset: dict[str, np.ndarray], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origins, dirs, pixels = [], [], []
    for idx in indices:
        o, d = rays_for_camera(dataset["poses"][idx], dataset["intrinsics"], int(dataset["width"]), int(dataset["height"]))
        origins.append(o.reshape(-1, 3))
        dirs.append(d.reshape(-1, 3))
        pixels.append(dataset["images"][idx].reshape(-1, 3))
    return np.concatenate(origins), np.concatenate(dirs), np.concatenate(pixels)


def train_model(dataset: dict[str, np.ndarray], train_indices: np.ndarray, steps: int, batch: int, device: torch.device) -> TinyNeRF:
    model = TinyNeRF().to(device)
    origins, dirs, pixels = _flatten_rays(dataset, train_indices)
    origins_t = torch.from_numpy(origins).to(device)
    dirs_t = torch.from_numpy(dirs).to(device)
    pixels_t = torch.from_numpy(pixels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    gen = torch.Generator(device=device).manual_seed(7)
    for step in range(steps):
        idx = torch.randint(0, origins_t.shape[0], (batch,), generator=gen, device=device)
        pred, _ = render_rays(model, origins_t[idx], dirs_t[idx])
        loss = torch.mean((pred - pixels_t[idx]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in {int(steps * 0.55), int(steps * 0.8)}:
            for group in opt.param_groups:
                group["lr"] *= 0.45
    return model


@torch.no_grad()
def render_model_views(model: TinyNeRF, dataset: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    h, w = int(dataset["height"]), int(dataset["width"])
    images, depths = [], []
    for idx in indices:
        o, d = rays_for_camera(dataset["poses"][idx], dataset["intrinsics"], w, h)
        o_t = torch.from_numpy(o.reshape(-1, 3)).to(device)
        d_t = torch.from_numpy(d.reshape(-1, 3)).to(device)
        chunks_img, chunks_depth = [], []
        for start in range(0, o_t.shape[0], 4096):
            img, dep = render_rays(model, o_t[start : start + 4096], d_t[start : start + 4096])
            chunks_img.append(img.cpu().numpy())
            chunks_depth.append(dep.cpu().numpy())
        images.append(np.concatenate(chunks_img).reshape(h, w, 3).astype(np.float32))
        depths.append(np.concatenate(chunks_depth).reshape(h, w).astype(np.float32))
    return np.stack(images), np.stack(depths)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = render_dataset(num_views=args.num_views, width=args.width, height=args.height)
    offset = args.split_offset % args.num_views
    if args.split_mode == "sparse":
        train_bases = np.array([0, 6, 12, 18], dtype=np.int64)
        candidate_bases = np.array([2, 4, 8, 10, 14, 16, 20, 22], dtype=np.int64)
        test_bases = np.array([3, 9, 15, 21], dtype=np.int64)
    elif args.split_mode == "dense":
        train_bases = np.array([0, 3, 6, 9, 12, 15, 18, 21], dtype=np.int64)
        candidate_bases = np.array([1, 2, 4, 5, 7, 8, 10, 11], dtype=np.int64)
        test_bases = np.array([13, 14, 16, 17], dtype=np.int64)
    else:
        raise ValueError(f"unknown split mode: {args.split_mode}")
    train_indices = (train_bases + offset) % args.num_views
    candidate_indices = (candidate_bases + offset) % args.num_views
    test_indices = (test_bases + offset) % args.num_views
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = train_model(dataset, train_indices, args.steps, args.batch_size, device)
    source_images, source_depths = render_model_views(model, dataset, train_indices, device)
    test_images, test_depths = render_model_views(model, dataset, test_indices, device)
    candidate_images, candidate_depths = render_model_views(model, dataset, candidate_indices, device)

    uncertainties, errors, valid_masks = [], [], []
    for i, view_idx in enumerate(test_indices):
        u = pixel_uncertainty_depth(
            test_depths[i], dataset["poses"][view_idx], source_depths, dataset["poses"][train_indices], dataset["intrinsics"]
        )
        uncertainties.append(u)
        errors.append(np.abs(test_depths[i] - dataset["depths"][view_idx]))
        valid_masks.append(dataset["depths"][view_idx] < 5.95)

    candidate_scores = []
    for i, view_idx in enumerate(candidate_indices):
        candidate_scores.append(
            image_uncertainty(
                candidate_images[i],
                candidate_depths[i],
                dataset["poses"][view_idx],
                source_images,
                source_depths,
                dataset["poses"][train_indices],
                dataset["intrinsics"],
            )
        )
    candidate_scores = np.array(candidate_scores, dtype=np.float32)
    chosen = int(candidate_indices[int(np.argmax(candidate_scores))])
    oracle_errors = np.mean(np.abs(candidate_images - dataset["images"][candidate_indices]), axis=(1, 2, 3))

    np.savez_compressed(
        out_dir / "warprf_outputs.npz",
        uncertainty=np.stack(uncertainties),
        depth_error=np.stack(errors),
        valid_depth_mask=np.stack(valid_masks),
        candidate_scores=candidate_scores,
        candidate_oracle_errors=oracle_errors.astype(np.float32),
        candidate_indices=candidate_indices,
        selected_index=np.array(chosen),
        train_indices=train_indices,
        test_indices=test_indices,
        source_view_count=np.array(len(train_indices)),
        used_ground_truth_for_uncertainty=np.array(0),
    )
    metadata = {
        "method": "tiny_nerf_warprf",
        "device": str(device),
        "steps": args.steps,
        "width": args.width,
        "height": args.height,
        "split_offset": args.split_offset,
        "split_mode": args.split_mode,
        "selected_candidate_view": chosen,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/warprf")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--num-views", type=int, default=24)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--split-offset", type=int, default=0)
    parser.add_argument("--split-mode", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
