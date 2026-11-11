"""Batched tiny NeRF backbone used to build every benchmark instance.

The backbone is a small NeRF: a positional-encoded MLP whose density depends on
position only and whose colour also depends on the viewing direction, rendered
with volume rendering (paper Eq. 1-2). Many independent models are trained in
lockstep as one batched tensor program so that a scene's acquisition branches
can be trained cheaply on one GPU. Models are fully independent (separate
weights and Adam moments); the only thing they share is the ray-sampling random
stream of their scene, which makes branch-to-branch comparisons paired.

Rendered depth is the expected ray-termination distance (Eq. 2) converted to
camera-space z, matching the ground-truth convention in ``data/scenes.py``.
"""

from __future__ import annotations

import math

import torch

from data.scenes import BACKGROUND_RGB, BOUND_RADIUS

HIDDEN = 96
FREQS = 6
DIR_FREQS = 2
SAMPLES = 48
BATCH_RAYS = 1024
LR_START = 5e-3
LR_END = 5e-4
SCHEDULE_STEPS = 3000  # learning-rate horizon of the full acquisition schedule
MISS_DISTANCE = 6.0
POS_DIM = 3 + 6 * FREQS
DIR_DIM = 3 + 6 * DIR_FREQS
LAYERS = {
    "W0": (POS_DIM, HIDDEN),
    "W1": (HIDDEN, HIDDEN),
    "W2": (HIDDEN, HIDDEN),
    "Ws": (HIDDEN, 1),
    "Wc0": (HIDDEN + DIR_DIM, HIDDEN // 2),
    "Wc1": (HIDDEN // 2, 3),
}


def init_state(num_models: int, seed: int, device: torch.device) -> dict:
    """Identical nn.Linear-style initialisation for every model in the batch."""
    gen = torch.Generator().manual_seed(int(seed))
    params, m, v = {}, {}, {}
    for name, (fan_in, fan_out) in LAYERS.items():
        bound = 1.0 / math.sqrt(fan_in)
        weight = (torch.rand(fan_in, fan_out, generator=gen) * 2.0 - 1.0) * bound
        bias = (torch.rand(1, fan_out, generator=gen) * 2.0 - 1.0) * bound
        for key, value in ((name, weight), ("b" + name[1:], bias)):
            params[key] = value.expand(num_models, *value.shape).clone().to(device)
            m[key] = torch.zeros_like(params[key])
            v[key] = torch.zeros_like(params[key])
    return {"params": params, "m": m, "v": v, "step": 0}


def take(state: dict, index) -> dict:
    """Copy a subset of models (weights and optimiser moments)."""
    if not torch.is_tensor(index):
        index = torch.as_tensor(index, dtype=torch.long)
    out = {"step": state["step"]}
    for part in ("params", "m", "v"):
        out[part] = {k: t[index.to(t.device)].clone() for k, t in state[part].items()}
    return out


def to_device(state: dict, device) -> dict:
    out = {"step": state["step"]}
    for part in ("params", "m", "v"):
        out[part] = {k: t.to(device) for k, t in state[part].items()}
    return out


def concat(states: list[dict]) -> dict:
    steps = {s["step"] for s in states}
    if len(steps) != 1:
        raise ValueError("cannot concatenate models at different optimisation steps")
    out = {"step": states[0]["step"]}
    for part in ("params", "m", "v"):
        out[part] = {k: torch.cat([s[part][k] for s in states]) for k in states[0][part]}
    return out


def num_models(state: dict) -> int:
    return int(next(iter(state["params"].values())).shape[0])


def export_weights(state: dict, index: int) -> dict:
    return {k: t[index].detach().cpu().numpy() for k, t in state["params"].items()}


def _encode(x: torch.Tensor, freqs: int) -> torch.Tensor:
    parts = [x]
    for i in range(freqs):
        f = (2.0**i) * math.pi
        parts.append(torch.sin(f * x))
        parts.append(torch.cos(f * x))
    return torch.cat(parts, dim=-1)


def query(params: dict, points: torch.Tensor, dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """points/dirs [M, N, 3] -> density [M, N], rgb [M, N, 3]."""
    h = _encode(points / BOUND_RADIUS, FREQS)
    h = torch.relu(torch.baddbmm(params["b0"], h, params["W0"]))
    h = torch.relu(torch.baddbmm(params["b1"], h, params["W1"]))
    h = torch.relu(torch.baddbmm(params["b2"], h, params["W2"]))
    sigma = torch.nn.functional.softplus(torch.baddbmm(params["bs"], h, params["Ws"])[..., 0] - 1.0)
    c = torch.cat([h, _encode(dirs, DIR_FREQS)], dim=-1)
    c = torch.relu(torch.baddbmm(params["bc0"], c, params["Wc0"]))
    rgb = torch.sigmoid(torch.baddbmm(params["bc1"], c, params["Wc1"]))
    return sigma, rgb


def render_rays(params: dict, origins, dirs, cosines, jitter=None):
    """Volume-render rays [M, R, 3]; returns rgb [M,R,3], camera-z depth [M,R], opacity [M,R]."""
    b = torch.sum(origins * dirs, dim=-1)
    c = torch.sum(origins * origins, dim=-1) - BOUND_RADIUS**2
    disc = b * b - c
    hit = disc > 0.0
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    t_near = torch.where(hit, torch.clamp(-b - sq, min=0.0), torch.full_like(b, MISS_DISTANCE))
    t_far = torch.where(hit, -b + sq, torch.full_like(b, MISS_DISTANCE))
    m, r = b.shape
    base = torch.arange(SAMPLES, device=origins.device, dtype=origins.dtype)
    offs = (base + 0.5) if jitter is None else (base + jitter)
    u = (offs / SAMPLES).expand(m, r, SAMPLES)
    span = (t_far - t_near)[..., None]
    t = t_near[..., None] + span * u
    delta = span / SAMPLES
    pts = origins[..., None, :] + dirs[..., None, :] * t[..., None]
    sigma, rgb = query(
        params,
        pts.reshape(m, r * SAMPLES, 3),
        dirs[..., None, :].expand(m, r, SAMPLES, 3).reshape(m, r * SAMPLES, 3),
    )
    sigma = sigma.reshape(m, r, SAMPLES)
    rgb = rgb.reshape(m, r, SAMPLES, 3)
    alpha = 1.0 - torch.exp(-sigma * delta)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = alpha * trans
    acc = weights.sum(dim=-1)
    bg = torch.as_tensor(BACKGROUND_RGB, device=origins.device, dtype=origins.dtype)
    color = torch.sum(weights[..., None] * rgb, dim=-2) + (1.0 - acc)[..., None] * bg
    distance = torch.sum(weights * t, dim=-1) + (1.0 - acc) * t_far
    return color, distance * cosines, acc


def learning_rate(step: int) -> float:
    frac = min(step, SCHEDULE_STEPS) / SCHEDULE_STEPS
    return LR_START * (LR_END / LR_START) ** frac


def train(state: dict, rays: dict, view_lists: torch.Tensor, counts: torch.Tensor, stream_ids: torch.Tensor,
          stream_seeds: dict[int, int], steps: int, batch: int = BATCH_RAYS) -> dict:
    """Continue training every model for ``steps`` iterations (in place).

    rays: dict with origins/dirs [V,P,3], cos [V,P], rgb [V,P,3] on the device.
    view_lists: [M, L] global view ids each model trains on (padded); counts: [M].
    stream_ids: [M] scene id per model; models of the same scene share one random
    stream (seeded by ``stream_seeds[scene]``), whatever group they are trained in.
    """
    device = rays["origins"].device
    params = state["params"]
    for p in params.values():
        p.requires_grad_(True)
    pixels = rays["origins"].shape[1]
    unique = sorted({int(s) for s in stream_ids.tolist()})
    row_of = torch.as_tensor([unique.index(int(s)) for s in stream_ids.tolist()], device=device)
    gens = {}
    for s in unique:
        g = torch.Generator(device=device)
        g.manual_seed(int(stream_seeds[s]) + 7919 * int(state["step"]))
        gens[s] = g
    counts_d = counts.to(device)
    view_lists = view_lists.to(device)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(steps):
        state["step"] += 1
        u = torch.cat([torch.rand(1, batch, generator=gens[s], device=device) for s in unique])[row_of]
        jit = torch.cat([torch.rand(1, batch, SAMPLES, generator=gens[s], device=device) for s in unique])[row_of]
        flat = torch.minimum((u * (counts_d[:, None] * pixels)).long(), counts_d[:, None] * pixels - 1)
        slot = flat // pixels
        pix = flat % pixels
        view = torch.gather(view_lists, 1, slot)
        o = rays["origins"][view, pix]
        d = rays["dirs"][view, pix]
        cs = rays["cos"][view, pix]
        target = rays["rgb"][view, pix]
        pred, _, _ = render_rays(params, o, d, cs, jitter=jit)
        loss = torch.mean((pred - target) ** 2, dim=(1, 2)).sum()
        grads = torch.autograd.grad(loss, list(params.values()))
        lr = learning_rate(state["step"])
        t = state["step"]
        with torch.no_grad():
            for (name, p), g in zip(params.items(), grads):
                mm = state["m"][name]
                vv = state["v"][name]
                mm.mul_(beta1).add_(g, alpha=1.0 - beta1)
                vv.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                denom = (vv / (1.0 - beta2**t)).sqrt_().add_(eps)
                p.addcdiv_(mm, denom, value=-lr / (1.0 - beta1**t))
    for p in params.values():
        p.requires_grad_(False)
    return state


@torch.no_grad()
def render_views(state: dict, rays: dict, view_ids: torch.Tensor, chunk: int = 2304):
    """Render full views. view_ids [M, K] global ids -> rgb [M,K,P,3], depth [M,K,P], acc [M,K,P]."""
    device = rays["origins"].device
    view_ids = view_ids.to(device)
    m, k = view_ids.shape
    pixels = rays["origins"].shape[1]
    rgb = torch.empty(m, k, pixels, 3, device=device)
    depth = torch.empty(m, k, pixels, device=device)
    acc = torch.empty(m, k, pixels, device=device)
    for j in range(k):
        vid = view_ids[:, j]
        for start in range(0, pixels, chunk):
            sl = slice(start, min(start + chunk, pixels))
            o = rays["origins"][vid][:, sl]
            d = rays["dirs"][vid][:, sl]
            cs = rays["cos"][vid][:, sl]
            c, z, a = render_rays(state["params"], o, d, cs)
            rgb[:, j, sl] = c
            depth[:, j, sl] = z
            acc[:, j, sl] = a
    return rgb, depth, acc
