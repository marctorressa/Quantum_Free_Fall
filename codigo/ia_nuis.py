"""Caso realista: q incierto (frecuencia de trampa con ±20 %). La red se entrena con q aleatorio;
el estimador clásico de varianza asume q nominal."""
import sys; sys.path.insert(0, "/root/poster/calc")
import numpy as np, json, warnings
warnings.filterwarnings("ignore")
src = open("/root/poster/calc/ia_poc.py").read().split("# varianza de referencia")[0]
exec(src)
Q0 = 0.225
def sample_Tq(i, n, q):
    global Q
    Q = q; T = sample_T(i, n); Q = Q0; return T
ref_var = np.maximum.accumulate(np.array([sample_Tq(i, 400000, Q0).var() for i in range(len(gams))]))
res = {}
for N in [500, 1000, 2000, 5000]:
    X, y = [], []
    for i in range(len(gams)):
        for _ in range(300):
            q = rng.uniform(0.8, 1.2)*Q0; X.append(feat(sample_Tq(i, N, q))); y.append(R_true[i])
    reg = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=800,
                        early_stopping=True, random_state=0)).fit(np.array(X), np.array(y))
    Xt, yt, ym = [], [], []
    for i in range(len(gams)):
        for _ in range(40):
            q = rng.uniform(0.8, 1.2)*Q0; T = sample_Tq(i, N, q)
            Xt.append(feat(T)); yt.append(R_true[i]); ym.append(np.interp(T.var(), ref_var, R_true))
    yt = np.array(yt); pn = reg.predict(np.array(Xt))
    res[N] = (float(np.sqrt(np.mean((pn-yt)**2))), float(np.sqrt(np.mean((np.array(ym)-yt)**2))))
    print(N, res[N], flush=True)
json.dump(res, open("ia_nuis.json", "w"), indent=1)
