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
for split, root in [("train","/home/user/data/train_assets"),("test","/home/user/eval/test/assets")]:
    man = json.loads((Path(root)/"manifest.json").read_text())
    sp_align, sp_err, pick_align = [], [], []
    for n in man["scenes"]:
        a = load(Path(root)/n/"answers.npz", {"step1_psnr","candidate_images","eval_poses"})
        x = load(Path(root)/n/"inputs.npz", {"candidate_poses","sparse_candidate_rgb"})
        C=16; s1=a["step1_psnr"][1]; cp=x["candidate_poses"][1]; ep=a["eval_poses"][1]
        cen=[(cp[c*16+3],cp[c*16+7],cp[c*16+11]) for c in range(C)]; ecen=[(ep[c*16+3],ep[c*16+7],ep[c*16+11]) for c in range(8)]
        align=[-min(math.dist(c,e) for e in ecen) for c in cen]
        npx=48*48*3; ph=a["candidate_images"][1]; r=x["sparse_candidate_rgb"][1]
        err=[sum(abs(r[c*npx+j]-ph[c*npx+j]) for j in range(npx)) for c in range(C)]
        sp_align.append(spear(align,s1)); sp_err.append(spear(err,s1))
        # partial: combine err rank and align rank
        re, ra = rank(err), rank(align)
        pick_align.append(s1[max(range(C), key=lambda c: re[c]+ra[c])])
    print(split, "spearman(eval-alignment, step1)=%.3f  spearman(true err, step1)=%.3f  pick(err+align)=%.3f" % (mean(sp_align), mean(sp_err), mean(pick_align)))
