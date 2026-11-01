import sys, json, warnings; sys.path.insert(0, "/root/poster/calc")
warnings.filterwarnings("ignore")
import numpy as np
from scipy.integrate import trapezoid
from delay import asym
from busch import sumrule
gams = [-3,-2,-1.5,-1,-0.7,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5,0.7,1,1.5,2,3,5,10,20,50,1e4]
w = np.round(np.arange(-6.5, 6.5001, 0.02), 4)
def gauss(x, var=0.5): return np.exp(-x**2/(2*var))/np.sqrt(2*np.pi*var)
out = {"w0": float(w[0]), "dw": 0.02, "n": len(w), "gamma": [], "R": [], "Pb": [], "p": []}
for g in gams:
    k, wk, Pb = asym(g)
    free = trapezoid(wk, k); f = wk/free
    # densidad: pb N(w) + (1-pb) ∫ f(k) [N(w-k)+N(w+k)]/2 dk
    W = w[:, None]; K = k[None, :]
    conv = trapezoid(f[None, :]*0.5*(gauss(W-K)+gauss(W+K)), k, axis=1)
    pb = Pb/(Pb+free)
    p = pb*gauss(w) + (1-pb)*conv
    p /= trapezoid(p, w)
    S = sumrule(g)[2]; R = 0.5 + 2*(S - Pb*(-g*g/2))
    out["gamma"].append(g); out["R"].append(round(float(R), 4)); out["Pb"].append(round(float(pb), 4))
    out["p"].append([float(f"{v:.4g}") for v in p])
    print(g, round(R,4), round(pb,3), "var", round(float(trapezoid(p*w**2, w)),4))
json.dump(out, open("../web/datos.json", "w"), separators=(",", ":"))
import os; print(os.path.getsize("../web/datos.json"))
