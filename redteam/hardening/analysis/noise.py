import sys, json, math
sys.path.insert(0, "/home/user/redteam/hardening/analysis")
from npz import load
from pathlib import Path
def mean(v): return sum(v)/len(v)
def corr(a,b):
    ma, mb = mean(a), mean(b); num=sum((x-ma)*(y-mb) for x,y in zip(a,b)); den=math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b)); return num/den
for split, root in [("train","/home/user/data/train_assets"),("test","/home/user/eval/test/assets")]:
    man = json.loads((Path(root)/"manifest.json").read_text())
    cs, diffsd, tot = [], [], []
    for n in man["scenes"]:
        a = load(Path(root)/n/"answers.npz", {"final_psnr"})
        P = a["final_psnr"][1]; C = 16; P = [P[r*C:(r+1)*C] for r in range(C)]
        ab = [P[i][j] for i in range(C) for j in range(C) if i < j]; ba = [P[j][i] for i in range(C) for j in range(C) if i < j]
        cs.append(corr(ab, ba))
        d = [x-y for x,y in zip(ab,ba)]; diffsd.append(math.sqrt(mean([x*x for x in d])/2))
        allv = ab+ba; m = mean(allv); tot.append(math.sqrt(mean([(x-m)**2 for x in allv])))
    print(split, "corr(P[a,b],P[b,a]) mean %.3f min %.3f; noise sd (order) %.3f; total sd %.3f" % (mean(cs), min(cs), mean(diffsd), mean(tot)))
