import sys, json, math
sys.path.insert(0, "/home/user/redteam/hardening/analysis")
from npz import load
from pathlib import Path
def mean(v): return sum(v)/len(v)
def rank(v):
    o=sorted(range(len(v)), key=lambda i: v[i]); r=[0]*len(v)
    for k,i in enumerate(o): r[i]=k
    return r
for split, root in [("train","/home/user/data/train_assets"),("test","/home/user/eval/test/assets")]:
    man = json.loads((Path(root)/"manifest.json").read_text())
    picks = {}
    for n in man["scenes"]:
        a = load(Path(root)/n/"answers.npz", {"step1_psnr","candidate_images"})
        x = load(Path(root)/n/"inputs.npz", {"candidate_poses","sparse_source_poses","sparse_candidate_rgb"})
        C=16; s1=a["step1_psnr"][1]; cp=x["candidate_poses"][1]; sp=x["sparse_source_poses"][1]
        cen=[(cp[c*16+3],cp[c*16+7],cp[c*16+11]) for c in range(C)]; scen=[(sp[c*16+3],sp[c*16+7],sp[c*16+11]) for c in range(3)]
        npx=48*48*3; ph=a["candidate_images"][1]; r=x["sparse_candidate_rgb"][1]
        err=[sum(abs(r[c*npx+j]-ph[c*npx+j]) for j in range(npx)) for c in range(C)]
        dist=[min(math.dist(c,s) for s in scen) for c in cen]
        re, rd = rank(err), rank(dist)
        for wd in (0, 0.25, 0.5, 1.0, 2.0, 100):
            sc=[re[c]+wd*rd[c]+1e-3*err[c]/max(err) for c in range(C)]
            picks.setdefault(f"err+{wd}*dist", []).append(s1[max(range(C), key=lambda c: sc[c])])
        picks.setdefault("best", []).append(max(s1)); picks.setdefault("mean", []).append(mean(s1))
        # rank of best candidate in err ordering (15 = top)
        b=max(range(C), key=lambda c: s1[c]); picks.setdefault("best_err_rank", []).append(re[b]); picks.setdefault("best_dist_rank", []).append(rd[b])
    print(split, {k: round(mean(v),3) for k,v in picks.items()})
