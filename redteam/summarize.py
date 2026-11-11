"""Compute audit-cohort normalized rank from actual hidden-test JSON files.

Future rank normalization depends on future attempts and cannot be predicted.
This script makes the chosen audit cohort, directions, absolute coefficients,
experiment weights, missing-value skips and normalization explicit.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent

def getmetrics(scores):
    return {(e,m):v for e,x in scores.get('experiments',{}).items()
            for m,v in x.get('results',{}).get('warprf',{}).items()
            if isinstance(v,(int,float)) and math.isfinite(v)}

def summarize():
    defs=json.loads((HERE/'reference/reference_definition.json').read_text())['experiments']
    cohort={}
    statuses={}
    for result in HERE.glob('*/run_result.json'):
        status=json.loads(result.read_text());name=result.parent.name
        statuses[name]=status
        p=result.parent/'scores_test.json'
        if status.get('test_exit')!=0 or not p.exists():
            continue
        data=json.loads(p.read_text())
        # Do not silently interpret NaN as an ordinary missing metric.
        if any(not math.isfinite(v) for exp in data.get('experiments',{}).values()
               for metrics in exp.get('results',{}).values() for v in metrics.values()
               if isinstance(v,(int,float))):
            statuses[name]['unrankable_nonfinite']=True
            continue
        cohort[name]=data
    if 'reference' not in cohort:
        raise RuntimeError('Reference hidden-test run did not complete')
    base=getmetrics(cohort['reference'])
    flat={n:getmetrics(s) for n,s in cohort.items()}
    scales={}
    for key in base:
        vals=[d[key] for d in flat.values() if key in d]
        std=statistics.pstdev(vals)
        if std>0:
            scales[key]=std
    ranks={}
    contributions={}
    for name,values in flat.items():
        terms={}
        for (e,m),baseline in base.items():
            metric=defs[e]['metrics'].get(m)
            if metric is None or (e,m) not in values or (e,m) not in scales:
                continue
            direction=1 if metric['higher_is_better'] else -1
            terms[e+'/'+m]=(defs[e]['weight']*abs(metric['coefficient'])*direction*
                            (values[e,m]-baseline)/scales[e,m])
        contributions[name]=terms
        ranks[name]=sum(terms.values())
    details={}
    for name,data in cohort.items():
        vals=getmetrics(data)
        changes=[(1 if defs[e]['metrics'][m]['higher_is_better'] else -1)*(v-base[e,m])
                 for (e,m),v in vals.items() if (e,m) in base
                 and abs(defs[e]['metrics'][m]['coefficient'])>0]
        details[name]={
            'audit_cohort_rank_delta':ranks[name],
            'all_available_scored_metrics_nondecreasing':bool(changes) and all(v>=-1e-10 for v in changes),
            'strictly_improved_metric_count':sum(v>1e-10 for v in changes),
            'raw_metric_count':len(vals),
            'scored_metric_count':sum((e,m) in base and e in defs and m in defs[e]['metrics']
                                      and abs(defs[e]['metrics'][m]['coefficient'])>0
                                      for e,m in vals),
            'rank_contributions':contributions[name],
            'metrics':{e:data['experiments'][e]['results'].get('warprf',{}) for e in data.get('experiments',{})},
        }
    seeds=[n for n in ('reference_seed_11','reference_seed_12','reference_seed_13') if n in cohort]
    seed_spread={}
    for (e,m) in base:
        values=[flat[n][e,m] for n in seeds if (e,m) in flat[n]]
        if values:
            seed_spread[e+'/'+m]={'min':min(values),'max':max(values),'range':max(values)-min(values)}
    seed_ranks=[ranks[n] for n in seeds]
    comparison={}
    for n in cohort:
        if n.startswith(('geometry_seed_','reference_seed_')) or n in ('reference','more_training','less_training','weak_no_warp','weak_constant_predictions'):
            depth=[]; rgb=[]
            for e,x in cohort[n]['experiments'].items():
                if e.endswith('_depth_uncertainty') and 'warprf' in x['results']:
                    depth.append(x['results']['warprf']['mean_depth_error'])
                if e.endswith('_active_view_proxy') and 'random_view' in x['results']:
                    rgb.append(x['results']['random_view']['selected_candidate_error'])
            comparison[n]={'rank_delta':ranks[n],'mean_test_depth_mae':statistics.mean(depth),
                           'mean_test_candidate_rgb_mae':statistics.mean(rgb)}
    bounds=[]
    for (e,m),v in base.items():
        if m not in defs[e]['metrics'] or abs(defs[e]['metrics'][m]['coefficient'])==0:
            continue
        if (m in ('ause','mean_depth_error','selection_regret_fraction','used_ground_truth_for_uncertainty') and v==0) or (m.endswith('spearman') or m=='spearman_depth_error') and v==1:
            bounds.append({'experiment':e,'metric':m,'value':v})
    random_baseline_rank=0.0
    for e,experiment in cohort['reference']['experiments'].items():
        method='random_uncertainty' if e.endswith('_depth_uncertainty') else 'random_view'
        for m,value in experiment['results'][method].items():
            metric=defs[e]['metrics'][m]
            if (e,m) in scales:
                random_baseline_rank+=(defs[e]['weight']*abs(metric['coefficient'])*
                    (1 if metric['higher_is_better'] else -1)*(value-base[e,m])/scales[e,m])
    report={
        'normalization':'population std across all successful finite audit cases; failed and nonfinite cases excluded; empty results skip all losses',
        'rank_formula':'sum(weight * abs(coefficient) * direction * (value - reference_hidden_test_value) / std)',
        'reference_rank':0,
        'random_baseline_rank_delta':random_baseline_rank,
        'cohort':sorted(cohort),
        'scales':{e+'/'+m:v for (e,m),v in scales.items()},
        'case_results':details,
        'quality_comparison':comparison,
        'reference_seed_rank_range':max(seed_ranks)-min(seed_ranks) if seed_ranks else None,
        'reference_seed_metric_spread':seed_spread,
        'reference_at_intended_bounds':bounds,
        'unsuccessful_cases':{n:s for n,s in statuses.items() if n not in cohort},
    }
    (HERE/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (HERE/'rank_table.csv').open('w') as out:
        writer=csv.writer(out,lineterminator='\n');writer.writerow(['case','audit_cohort_rank_delta','scored_metric_count','dominates_reference'])
        for name,d in sorted(details.items(),key=lambda item:-item[1]['audit_cohort_rank_delta']):
            writer.writerow([name,d['audit_cohort_rank_delta'],d['scored_metric_count'],d['all_available_scored_metrics_nondecreasing']])
    print(json.dumps({'ranks':ranks,'quality':comparison,'seed_rank_spread':report['reference_seed_rank_range']},indent=2),flush=True)

if __name__=='__main__':
    summarize()
