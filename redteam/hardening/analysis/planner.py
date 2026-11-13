import sys, json, math
sys.path.insert(0, "/home/user/redteam/hardening/analysis")
from npz import load
from pathlib import Path
def mean(v): return sum(v)/len(v)
def se(d): m=mean(d); return math.sqrt(sum((x-m)**2 for x in d)/(len(d)-1)/len(d))
def maxgap(az):
    s = sorted(a % (2*math.pi) for a in az); g = [s[i+1]-s[i] for i in range(len(s)-1)] + [s[0]+2*math.pi-s[-1]]; return max(g)
for split, root, det in [("train","/home/user/data/train_assets","/home/user/tmp/hardening/details_train.json"),("test","/home/user/eval/test/assets","/home/user/tmp/hardening/details_test.json")]:
    D = json.load(open(det)); man = json.loads((Path(root)/"manifest.json").read_text())
    res = {k: [] for k in ["warprf","farthest","gap_planner","gap_greedy","sumgap_planner"]}
    for i, n in enumerate(man["scenes"]):
        a = load(Path(root)/n/"answers.npz", {"final_psnr"}); x = load(Path(root)/n/"inputs.npz", {"candidate_poses","sparse_source_poses"})
        C=16; P = a["final_psnr"][1]; P=[P[r*C:(r+1)*C] for r in range(C)]
        cp = x["candidate_poses"][1]; sp = x["sparse_source_poses"][1]
        caz = [math.atan2(cp[c*16+3], cp[c*16+11]) for c in range(C)]; saz = [math.atan2(sp[c*16+3], sp[c*16+11]) for c in range(3)]
        w = D["warprf"]["choices"][i]; res["warprf"].append(P[w[0]][w[1]])
        f = D["baselines"]["farthest_view"]["choices"][i]; res["farthest"].append(P[f[0]][f[1]])
        best = min(((maxgap(saz+[caz[p],caz[q]]), p, q) for p in range(C) for q in range(C) if p!=q))
        # order: first = the one that alone gives the smaller gap
        p,q = best[1], best[2]
        if maxgap(saz+[caz[q]]) < maxgap(saz+[caz[p]]): p,q = q,p
        res["gap_planner"].append(P[p][q])
        g1 = min(range(C), key=lambda c: maxgap(saz+[caz[c]])); g2 = min((c for c in range(C) if c!=g1), key=lambda c: maxgap(saz+[caz[g1],caz[c]]))
        res["gap_greedy"].append(P[g1][g2])
    print(split, {k: round(mean(v),3) for k,v in res.items() if v})
    for k in ["farthest","gap_planner","gap_greedy"]:
        d=[p-q for p,q in zip(res[k],res["warprf"])]; print("   %s - warprf %+.3f (se %.3f) better %d/%d" % (k, mean(d), se(d), sum(1 for v in d if v>1e-9), len(d)))
