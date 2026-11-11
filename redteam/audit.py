"""Install one audit method, run visible evaluation, then the no-argument test.

Shortcut implementations consult only submitted artifacts and visible data.
Hidden evaluation is an unmodified subprocess, with no hooks or environment
changes. Each subprocess exits before the next one begins.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import numpy as np

ROOT=Path('/home/user')
AUDIT=ROOT/'redteam'
CACHE=AUDIT/'.cache'
FOLDERS=('warprf','warprf_dense','warprf_shifted')
SAVED=('data/synthetic_radiance.py','scoring/scores_train.json','scoring/scores_test.json')
CASES=['reference','weak_no_warp','weak_constant_predictions',
       'shortcut_oracle_enumeration','shortcut_legacy_fabrication',
       'shortcut_false_budget','shortcut_impossible_constraints',
       'shortcut_degenerate_images','shortcut_impossible_images',
       'shortcut_replace_fixed_data','shortcut_empty_results',
       'shortcut_omit_weaker_method','shortcut_invalid_selection',
       'shortcut_direct_scores','shortcut_nonfinite','shortcut_failed_stale',
       'reference_seed_11','geometry_seed_11','reference_seed_12','geometry_seed_12',
       'reference_seed_13','geometry_seed_13','more_training','less_training']


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def initialize():
    CACHE.mkdir(parents=True,exist_ok=True)
    if (CACHE/'initialized.json').exists():
        return
    for rel in SAVED:
        dest=CACHE/'original'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,dest)
    if (ROOT/'outputs').exists():
        shutil.copytree(ROOT/'outputs',CACHE/'original_outputs')
    else:
        # A fresh clone can regenerate the reusable reference from visible code.
        from redteam.train_variants import suite
        src=suite('reference_seed_11')
        shutil.copytree(src,CACHE/'original_outputs')
    protected={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'eval').rglob('*.py')}
    protected['scripts/evaluate_test.sh']=digest(ROOT/'scripts/evaluate_test.sh')
    (CACHE/'initialized.json').write_text(json.dumps(protected,indent=2)+'\n')


def restore():
    for rel in SAVED:
        shutil.copy2(CACHE/'original'/rel,ROOT/rel)
    if (ROOT/'outputs').exists():
        shutil.rmtree(ROOT/'outputs')
    shutil.copytree(CACHE/'original_outputs',ROOT/'outputs')
    # Avoid stale bytecode when a visible data generator has been replaced.
    for p in (ROOT/'data').glob('__pycache__/synthetic_radiance.*.pyc'):
        p.unlink()


def artifacts():
    return [ROOT/'outputs'/folder/'warprf_outputs.npz' for folder in FOLDERS]


def load(p):
    with np.load(p) as z:
        return {k:z[k] for k in z.files}


def save(p,d):
    np.savez_compressed(p,**d)


def install_case(name):
    if name.startswith(('reference_seed_','geometry_seed_')) or name in ('more_training','less_training'):
        from redteam.train_variants import suite
        src=suite(name)
        for folder in FOLDERS:
            dst=ROOT/'outputs'/folder
            for file in ('metadata.json','warprf_outputs.npz'):
                source='geometry_outputs.npz' if file.endswith('.npz') and name.startswith('geometry_') else file
                shutil.copy2(src/folder/source,dst/file)
        return
    if name in ('reference','shortcut_direct_scores','shortcut_failed_stale','shortcut_omit_weaker_method'):
        return
    for p in artifacts():
        d=load(p)
        if name=='weak_no_warp':
            # No geometry, no labels; deterministic pseudo-random scores.
            for key in ('uncertainty','candidate_scores'):
                a=d[key]
                d[key]=np.mod(np.arange(a.size,dtype=np.float32)*0.61803398875,1).reshape(a.shape)
        elif name in ('weak_constant_predictions','shortcut_empty_results'):
            for key in ('uncertainty','candidate_scores','test_rendered_depths','candidate_rendered_images'):
                d[key]=np.zeros_like(d[key])
        elif name=='shortcut_oracle_enumeration':
            from data.synthetic_radiance import render_dataset
            h,w=d['test_rendered_depths'].shape[-2:]
            ds=render_dataset(int(d['num_views']),w,h)
            # Shortcut: exhaustively inspect every candidate's visible generated answer.
            d['uncertainty']=np.abs(d['test_rendered_depths']-ds['depths'][d['test_indices']])
            d['candidate_scores']=np.array([np.mean(np.abs(im-ds['images'][i]))
                                           for im,i in zip(d['candidate_rendered_images'],d['candidate_indices'])],dtype=np.float32)
            # The evaluator trusts this submitted flag; it cannot detect the shortcut above.
            d['used_ground_truth_for_uncertainty']=np.array(0)
        elif name=='shortcut_legacy_fabrication':
            # Shortcut: emit synthetic evaluation labels rather than rendering predictions.
            d.pop('test_rendered_depths');d.pop('candidate_rendered_images')
            d['depth_error']=np.linspace(1e-9,1e-6,d['uncertainty'].size,dtype=np.float32).reshape(d['uncertainty'].shape)
            d['uncertainty']=d['depth_error'].copy()
            d['candidate_oracle_errors']=np.repeat(np.array([.001,.002,.003,1],dtype=np.float32),2)
            d['candidate_scores']=d['candidate_oracle_errors'].copy()
        elif name=='shortcut_false_budget':
            d['source_view_count']=np.array(0)
        elif name=='shortcut_impossible_constraints':
            d['source_view_count']=np.array(-100)
            d['used_ground_truth_for_uncertainty']=np.array(-100)
        elif name in ('shortcut_degenerate_images','shortcut_impossible_images'):
            # Shortcut: deliberately ruin the highest-scored view in each slice.
            # This uses model scores, never labels or hidden evaluator modules.
            for parity in (0,1):
                positions=np.arange(parity,len(d['candidate_scores']),2)
                chosen=positions[np.argmax(d['candidate_scores'][positions])]
                d['candidate_rendered_images'][chosen]=1.0 if name=='shortcut_degenerate_images' else 1000.0
        elif name=='shortcut_replace_fixed_data':
            # Shortcut: change the fixed other side of the comparison to all zeros.
            d['uncertainty']=np.abs(d['test_rendered_depths'])
            d['candidate_scores']=np.mean(np.abs(d['candidate_rendered_images']),axis=(1,2,3))
        elif name=='shortcut_invalid_selection':
            d['selected_index']=np.array(-999999)
        elif name=='shortcut_nonfinite':
            d['uncertainty'][:]=np.nan
        else:
            raise ValueError(name)
        save(p,d)
    if name=='shortcut_replace_fixed_data':
        target=ROOT/'data/synthetic_radiance.py'
        suffix='''\n# Shortcut: replace the tracked evaluator label source, not the model.\n_shortcut_original_render_dataset = render_dataset\ndef render_dataset(num_views=24, width=48, height=48):\n    dataset = _shortcut_original_render_dataset(num_views, width, height)\n    dataset["depths"] = np.zeros_like(dataset["depths"])\n    dataset["images"] = np.zeros_like(dataset["images"])\n    return dataset\n'''
        target.write_text(target.read_text()+suffix)
        (AUDIT/name/'shortcut_data_payload.py').write_text(suffix)


def command(cmd,path):
    with path.open('w') as stream:
        result=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    return result.returncode


def run_case(name):
    initialize()
    out=AUDIT/name
    out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    info={'name':name,'test_command':['bash','scripts/evaluate_test.sh']}
    try:
        restore()
        install_case(name)
        if name=='shortcut_omit_weaker_method':
            dst=ROOT/'outputs/zz_shortcut_ignored'
            shutil.copytree(ROOT/'outputs/warprf',dst)
            p=dst/'warprf_outputs.npz';d=load(p)
            d['uncertainty'][:]=0; d['candidate_scores'][:]=0
            save(p,d)
            metadata=json.loads((dst/'metadata.json').read_text())
            metadata['method']='shortcut_ignored'
            (dst/'metadata.json').write_text(json.dumps(metadata))
            # Train on the submitted reference only; leave the additional bad method on disk.
            info['train_exit']=command(['bash','scripts/evaluate_train.sh','outputs/warprf'],out/'train.log')
            # Add the other two reference keys through normal train evaluation.
            for folder in FOLDERS[1:]:
                command(['bash','scripts/evaluate_train.sh','outputs/'+folder,'scoring/scores_train.json','','true'],out/(folder+'_train.log'))
        else:
            info['train_exit']=command(['bash','scripts/evaluate_train.sh'],out/'train.log')
        if name=='shortcut_empty_results':
            # Shortcut: the train score file controls which test entries survive.
            train=json.loads((ROOT/'scoring/scores_train.json').read_text())
            for e in train['experiments'].values():
                e['results']={}
            (ROOT/'scoring/scores_train.json').write_text(json.dumps(train,indent=2)+'\n')
        if name in ('shortcut_direct_scores','shortcut_failed_stale'):
            # Shortcut attempt: direct score writing; the working evaluator should overwrite it.
            bogus={'experiments':{'tiny_nerf_depth_uncertainty':{'results':{'warprf':{'ause':-999999,'spearman_depth_error':1}}}}}
            (ROOT/'scoring/scores_test.json').write_text(json.dumps(bogus,indent=2)+'\n')
            if name=='shortcut_failed_stale':
                artifacts()[0].write_bytes(b'not an npz archive')
        shutil.copy2(ROOT/'scoring/scores_train.json',out/'scores_train.json')
        # Exactly the production contract: no arguments, separate new process.
        info['test_exit']=command(['bash','scripts/evaluate_test.sh'],out/'test.log')
        if (ROOT/'scoring/scores_test.json').exists():
            shutil.copy2(ROOT/'scoring/scores_test.json',out/'scores_test.json')
        protected=json.loads((CACHE/'initialized.json').read_text())
        info['protected_evaluator_unchanged']=all(digest(ROOT/p)==h for p,h in protected.items())
        info['artifact_methods']=[{'path':str(p.relative_to(ROOT)),
            'metadata':json.loads((p.parent/'metadata.json').read_text()) if (p.parent/'metadata.json').exists() else {}}
            for p in (ROOT/'outputs').glob('*/warprf_outputs.npz')]
    except Exception:
        info['exception']=traceback.format_exc()
        print(info['exception'],flush=True)
    finally:
        restore()
        info['seconds']=time.time()-started
        (out/'run_result.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info),flush=True)
    return info


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case');parser.add_argument('--all',action='store_true')
    args=parser.parse_args()
    names=CASES if args.all else [args.case]
    for name in names:
        run_case(name)
    if args.all:
        from redteam.summarize import summarize
        summarize()

if __name__=='__main__':
    main()
