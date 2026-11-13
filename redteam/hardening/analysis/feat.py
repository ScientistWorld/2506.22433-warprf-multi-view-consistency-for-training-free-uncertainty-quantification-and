import sys, json, math
sys.path.insert(0, "/home/user/redteam/hardening/analysis")
from npz import load
from pathlib import Path
def mean(v): return sum(v)/len(v)
def rank(v):
    o=sorted(range(len(v)), key=lambda i: v[i]); r=[0]*len(v)
    for k,i in enumerate(o): r[i]=k
    return r
def spear(a,b):
    ra, rb = rank(a), rank(b); ma, mb = mean(ra), mean(rb)
    num=sum((x-ma)*(y-mb) for x,y in zip(ra,rb)); den=math.sqrt(sum((x-ma)**2 for x in ra)*sum((y-mb)**2 for y in rb)); return num/den if den else 0
for split, root, det in [("train","/home/user/data/train_assets","/home/user/tmp/hardening/details_train.json"),("test","/home/user/eval/test/assets","/home/user/tmp/hardening/details_test.json")]:
    D = json.load(open(det)); man = json.loads((Path(root)/"manifest.json").read_text())
    S = {k: [] for k in ["true_err","dist","azgap","elev","radius","fg_frac","lookat_off","render_dark"]}
    first = {"warprf": [], "oracle_err": [], "best": [], "mean": []}
    for i, n in enumerate(man["scenes"]):
        a = load(Path(root)/n/"answers.npz", {"step1_psnr","candidate_images"})
        x = load(Path(root)/n/"inputs.npz", {"candidate_poses","sparse_source_poses","sparse_candidate_rgb","sparse_candidate_depth"})
        C=16; s1 = a["step1_psnr"][1]; cp = x["candidate_poses"][1]; sp = x["sparse_source_poses"][1]
        cen=[(cp[c*16+3],cp[c*16+7],cp[c*16+11]) for c in range(C)]; scen=[(sp[c*16+3],sp[c*16+7],sp[c*16+11]) for c in range(3)]
        npx=48*48*3; ph=a["candidate_images"][1]; r=x["sparse_candidate_rgb"][1]; dep=x["sparse_candidate_depth"][1]
        err=[sum(abs(r[c*npx+j]-ph[c*npx+j]) for j in range(npx)) for c in range(C)]
        az=[math.atan2(p[0],p[2]) for p in cen]; saz=[math.atan2(p[0],p[2]) for p in scen]
        S["true_err"].append(spear(err,s1))
        S["dist"].append(spear([min(math.dist(c,s) for s in scen) for c in cen], s1))
        S["azgap"].append(spear([min(abs((u-v+math.pi)%(2*math.pi)-math.pi) for v in saz) for u in az], s1))
        S["elev"].append(spear([math.atan2(p[1], math.hypot(p[0],p[2])) for p in cen], s1))
        S["radius"].append(spear([math.sqrt(sum(q*q for q in p)) for p in cen], s1))
        S["fg_frac"].append(spear([sum(1 for j in range(2304) if dep[c*2304+j] > 0.5) for c in range(C)], s1))
        fwd=[(cp[c*16+2],cp[c*16+6],cp[c*16+10]) for c in range(C)]
        # distance of the optical axis from the scene centre (0,-0.15,0)
        def axis_off(c):
            o=cen[c]; d=fwd[c]; v=(0-o[0], -0.15-o[1], 0-o[2]); t=sum(v[k]*d[k] for k in range(3)); return math.sqrt(sum((v[k]-t*d[k])**2 for k in range(3)))
        S["lookat_off"].append(spear([axis_off(c) for c in range(C)], s1))
        S["render_dark"].append(spear([-sum(r[c*npx:(c+1)*npx]) for c in range(C)], s1))
        w = D["warprf"]["choices"][i][0]; first["warprf"].append(s1[w]); first["oracle_err"].append(s1[max(range(C), key=lambda c: err[c])])
        first["best"].append(max(s1)); first["mean"].append(mean(s1))
    print(split, "spearman with step-1 PSNR:", {k: round(mean(v),3) for k,v in S.items()})
    print("   step-1 PSNR of first pick:", {k: round(mean(v),3) for k,v in first.items()})
