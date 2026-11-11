"""Actual tiny-NeRF/WarpRF runs with independent initialization seeds.

The geometry variant corrects ray-distance versus optical-axis-depth conversion
in warping; the rendering predictions and training views remain the same.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from data.synthetic_radiance import render_dataset
from method.tiny_nerf_warprf import train_model, render_model_views
from method.warprf import pixel_uncertainty_depth, image_uncertainty

SPLITS = [('warprf', 'sparse', 0), ('warprf_dense', 'dense', 0), ('warprf_shifted', 'sparse', 1)]


def run_one(out: Path, seed: int, steps: int, mode: str, offset: int):
    marker = out / 'complete.json'
    if marker.exists():
        return
    start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = render_dataset(num_views=24, width=48, height=48)
    if mode == 'dense':
        ti = np.array([0,3,6,9,12,15,18,21])
        ci = np.array([1,2,4,5,7,8,10,11])
        vi = np.array([13,14,16,17])
    else:
        ti = np.array([0,6,12,18])
        ci = np.array([2,4,8,10,14,16,20,22])
        vi = np.array([3,9,15,21])
    ti, ci, vi = [(a+offset)%24 for a in (ti,ci,vi)]
    device = torch.device('cpu')
    model = train_model(ds, ti, steps, 1024, device)
    si, sd = render_model_views(model, ds, ti, device)
    _, vd = render_model_views(model, ds, vi, device)
    cr, cd = render_model_views(model, ds, ci, device)
    k, poses = ds['intrinsics'], ds['poses']
    u = np.stack([pixel_uncertainty_depth(d, poses[i], sd, poses[ti], k) for d,i in zip(vd,vi)])
    c = np.array([image_uncertainty(im,d,poses[i],si,sd,poses[ti],k) for im,d,i in zip(cr,cd,ci)], dtype=np.float32)
    common = dict(test_rendered_depths=vd, candidate_rendered_images=cr,
                  candidate_indices=ci, train_indices=ti, test_indices=vi,
                  source_view_count=np.array(len(ti)), used_ground_truth_for_uncertainty=np.array(0),
                  width=np.array(48),height=np.array(48),num_views=np.array(24))
    np.savez_compressed(out/'warprf_outputs.npz', **common, uncertainty=u, candidate_scores=c,
                        selected_index=np.array(ci[np.argmax(c)]))
    meta = dict(method='tiny_nerf_warprf', device='cpu', steps=steps, width=48, height=48,
                split_offset=offset, split_mode=mode, seed=seed)
    (out/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    # All ray directions have the same camera-space z factor for this intrinsics.
    yy,xx = np.mgrid[:48,:48]
    cosine = (1 + ((xx-k[0,2])/k[0,0])**2 + ((yy-k[1,2])/k[1,1])**2)**(-0.5)
    cosine = cosine.astype(np.float32)
    source_z = sd*cosine
    fixed_u = np.stack([pixel_uncertainty_depth(d*cosine,poses[i],source_z,poses[ti],k)/cosine
                        for d,i in zip(vd,vi)])
    fixed_c = np.array([image_uncertainty(im,d*cosine,poses[i],si,source_z,poses[ti],k)
                        for im,d,i in zip(cr,cd,ci)], dtype=np.float32)
    np.savez_compressed(out/'geometry_outputs.npz', **common, uncertainty=fixed_u, candidate_scores=fixed_c,
                        selected_index=np.array(ci[np.argmax(fixed_c)]))
    # Check the fix on camera coordinates without consulting evaluation labels.
    x=(xx-k[0,2])/k[0,0]
    y=-(yy-k[1,2])/k[1,1]
    old_points=np.stack([x*vd,y*vd,vd],axis=-1)
    correct_points=old_points*cosine[...,None]
    original_error=float(np.mean(np.abs(np.linalg.norm(old_points,axis=-1)-vd)))
    fixed_error=float(np.mean(np.abs(np.linalg.norm(correct_points,axis=-1)-vd)))
    marker.write_text(json.dumps(dict(seed=seed,steps=steps,seconds=time.time()-start,
        old_ray_distance_roundtrip_mae=original_error,fixed_ray_distance_roundtrip_mae=fixed_error),indent=2)+'\n')
    print('TRAINED',str(out),marker.read_text(),flush=True)


def suite(name: str):
    if name.startswith('reference_seed_') or name.startswith('geometry_seed_'):
        seed=int(name.rsplit('_',1)[1]); steps=900
    elif name=='more_training':
        seed=11; steps=2700
    elif name=='less_training':
        seed=11; steps=90
    else:
        raise ValueError(name)
    cache=Path('redteam/.cache')/f'train_{steps}_seed_{seed}'
    for folder,mode,offset in SPLITS:
        run_one(cache/folder,seed,steps,mode,offset)
    return cache

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name')
    suite(parser.parse_args().name)
