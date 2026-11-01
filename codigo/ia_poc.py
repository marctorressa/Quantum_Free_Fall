"""Prueba de concepto: inferir la interacción a partir de histogramas finitos de
tiempos de llegada de un constituyente (vuelo largo, q fijo).
w = G + s*k (libres) o w = G (ligados), G~N(0,1/2), s=±1; T/t_c = sqrt(1+q^2 w^2) - q w.
Se detecta UN átomo por par (eventos iid). Jitter del detector gaussiano."""
import numpy as np, warnings, json, sys
warnings.filterwarnings("ignore")
from scipy.integrate import trapezoid, cumulative_trapezoid
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from delay import asym
from busch import sumrule

rng = np.random.default_rng(7)
Q = 0.225; JIT = 0.01                       # q de la fila 1; jitter = 1 % de t_c
EDGES = np.linspace(0.35, 2.6, 46)          # 45 bins en T/t_c

# ---- tabla de distribuciones asintóticas en una malla de gamma ----
phis = np.linspace(-0.75, 0.995, 45)        # phi = (2/pi) arctan(gamma)
gams = np.tan(phis*np.pi/2)
tab = []
for g in gams:
    k, wk, Pb = asym(g)
    free = trapezoid(wk, k)
    cdf = cumulative_trapezoid(wk, k, initial=0)/free
    S = sumrule(g)[2]; R = 0.5 + 2*(S - Pb*(-g*g/2))
    tab.append((k, cdf, Pb/(Pb+free), R))
R_true = np.array([t[3] for t in tab])

def sample_T(i, n, odd=False):
    """n tiempos de llegada T/t_c; odd=True -> triplete/fermiones polarizados (=Tonks-Girardeau)."""
    k, cdf, pb, _ = tab[-1] if odd else tab[i]
    if odd: pb = 0.0
    G = rng.normal(0, np.sqrt(0.5), n)
    kk = np.interp(rng.random(n), cdf, k)
    s = rng.choice([-1, 1], n)
    w = np.where(rng.random(n) < pb, G, G + s*kk)
    T = np.sqrt(1 + Q**2*w**2) - Q*w
    return T + rng.normal(0, JIT, n)

def feat(T):
    h, _ = np.histogram(T, EDGES); return h/len(T)

def moment_est(T, ref_var):
    v = T.var()
    return np.interp(v, ref_var, R_true)     # var monótona en R

# varianza de referencia (muestras grandes) para el estimador clásico
ref_var = np.array([sample_T(i, 400000).var() for i in range(len(gams))])
ref_var = np.maximum.accumulate(ref_var)

out = {"N": [], "rmse_nn": [], "rmse_mom": [], "acc_nn": [], "acc_lr": []}
Ns = [50, 100, 200, 500, 1000, 2000, 5000]
i_g1 = int(np.argmin(abs(gams - 1.0)))       # singlete con gamma ~ 1
for N in Ns:
    # --- regresión de R = delta/delta_1 ---
    ntr = 120
    X, y = [], []
    for i in range(len(gams)):
        for _ in range(ntr):
            X.append(feat(sample_T(i, N))); y.append(R_true[i])
    X, y = np.array(X), np.array(y)
    reg = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=600,
                        early_stopping=True, random_state=0)).fit(X, y)
    Xt, yt, ym = [], [], []
    for i in range(len(gams)):
        for _ in range(30):
            T = sample_T(i, N); Xt.append(feat(T)); yt.append(R_true[i]); ym.append(moment_est(T, ref_var))
    pn = reg.predict(np.array(Xt)); yt = np.array(yt)
    out["rmse_nn"].append(float(np.sqrt(np.mean((pn-yt)**2))))
    out["rmse_mom"].append(float(np.sqrt(np.mean((np.array(ym)-yt)**2))))
    # --- clasificación singlete(gamma~1) vs triplete ---
    Xc = [feat(sample_T(i_g1, N)) for _ in range(1500)] + [feat(sample_T(0, N, odd=True)) for _ in range(1500)]
    yc = [0]*1500 + [1]*1500
    clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=600,
                        early_stopping=True, random_state=0)).fit(np.array(Xc), yc)
    Xv = [feat(sample_T(i_g1, N)) for _ in range(800)] + [feat(sample_T(0, N, odd=True)) for _ in range(800)]
    yv = np.array([0]*800 + [1]*800)
    out["acc_nn"].append(float(np.mean(clf.predict(np.array(Xv)) == yv)))
    # referencia: umbral óptimo en la varianza muestral
    vs = np.array([sample_T(i_g1, N).var() for _ in range(800)]); vt = np.array([sample_T(0, N, odd=True).var() for _ in range(800)])
    thr = 0.5*(np.median(vs) + np.median(vt))
    out["acc_lr"].append(float(0.5*(np.mean(vs < thr) + np.mean(vt >= thr))))
    out["N"].append(N)
    print(N, out["rmse_nn"][-1], out["rmse_mom"][-1], out["acc_nn"][-1], out["acc_lr"][-1], flush=True)
out["gamma_singlete"] = float(gams[i_g1]); out["R_singlete"] = float(R_true[i_g1]); out["R_triplete"] = float(R_true[-1])
json.dump(out, open("ia_poc.json", "w"), indent=1)
