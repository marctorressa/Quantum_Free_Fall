import numpy as np, warnings, json
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("pdf")
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from delay import asym, delta
from busch import sumrule

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"

from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation
def _coma(ax):
    for axis in (ax.xaxis, ax.yaxis):
        if axis.get_scale()=="linear":
            axis.set_major_formatter(FuncFormatter(lambda v,p: (f"{v:.1f}" if abs(v-round(v))>1e-9 or True else f"{v:.0f}").replace(".",",")))

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2})

# ---------- Fig A: reproduccion Fig.1 Beau-Martellini desde Ec.(11), v0=0, sigma=tau=1
def mean_T_eq11(q, tc=1e4):
    x = tc/(2*q); g = 1/(q*tc)
    t = np.geomspace(1e-6*tc, 60*max(1, q)*tc, 400000)
    s2 = 1 + t**2
    Pi = (x*t + g*t + 0.5*g*t**3)/s2 * np.exp(-(x - 0.5*g*t**2)**2/(2*s2))/np.sqrt(2*np.pi*s2)
    return trapezoid(t*Pi, t)/trapezoid(Pi, t), tc

qs = np.geomspace(1e-2, 1e2, 41)
d11 = np.array([mean_T_eq11(q)[0]/1e4 - 1 for q in qs])
k0, w0, P0 = asym(0.0)
qh = np.geomspace(1e-2, 1e2, 13)
dh = np.array([delta(q, 0.0, k0, w0, P0) for q in qh])

fig, ax = plt.subplots(figsize=(6.2, 4.3))
ax.loglog(qs, d11, color=C[0], label="Ec. (11), integración numérica")
ax.loglog(qs, qs**2/2, "--", color=C[1], lw=1.6, label=r"Ec. (18): $\delta=q^2/2$")
ax.loglog(qs, np.sqrt(2/np.pi)*qs, ":", color=C[2], lw=2, label=r"Ec. (19): $\delta=\sqrt{2/\pi}\,q$")
ax.loglog(qh, dh, "o", ms=7, mfc="white", mec=INK, mew=1.4, label="Cálculo independiente (vuelo largo)")
ax.set_xlabel(r"parámetro de cuanticidad $q=\hbar/(2m\sigma\sqrt{2gx})$")
ax.set_ylabel(r"retraso relativo $\delta=(\langle T_x\rangle-t_c)/t_c$")
ax.grid(True, which="major", color=GRID, lw=0.8); ax.legend(frameon=False, fontsize=9.5)
fig.tight_layout(); fig.savefig("../fig/figA_reproduccion.pdf"); plt.close(fig)

# ---------- Fig B: cociente delta^(1)(gamma)/delta_1 
gam = np.concatenate([-np.geomspace(3, 0.05, 18), [0.0], np.geomspace(0.05, 300, 26)])
qset = [0.225, 0.713, 10.0]
ref = {q: delta(q, 0.0, k0, w0, P0) for q in qset}
rows = []
for gg in gam:
    k, w, Pb = asym(gg)
    S = sumrule(gg)[2]; rule = 0.5 + 2*(S - Pb*(-gg*gg/2))
    rows.append([gg, Pb, rule] + [delta(q, gg, k, w, Pb)/ref[q] for q in qset])
R = np.array(rows); np.savetxt("ratios.csv", R, delimiter=",",
    header="gamma,P_ligado,regla_suma(q->0),q=0.225,q=0.713,q=10", fmt="%.6f")

fig, ax = plt.subplots(figsize=(6.6, 4.3))
for y in (0.5, 1, 2): ax.axhline(y, color=GRID, lw=1, zorder=0)
ax.plot(R[:,0], R[:,2], color=C[0], label=r"$q\to0$: regla de suma (analítico)")
ax.plot(R[:,0], R[:,3], "--", color=C[1], label=r"$q=0{,}225$ (Tabla I, fila 1)")
ax.plot(R[:,0], R[:,4], "-.", color=C[2], label=r"$q=0{,}713$ (Tabla I, fila 2)")
ax.plot(R[:,0], R[:,5], ":", color=C[3], lw=2.4, label=r"$q=10$ (régimen cuántico)")
ax.set_xscale("symlog", linthresh=0.1); ax.set_xlim(-3, 300); ax.set_ylim(0.35, 2.15)
ax.set_xlabel(r"intensidad de contacto $\gamma=\lambda/(\hbar\omega\,\ell_{\rm rel})$")
ax.set_ylabel(r"$\delta^{(1)}(\gamma)\,/\,\delta_1$"); _coma(ax)
ax.text(-1.6, 0.38, "dímero ligado (cae como $2m$)", fontsize=9, color=INK2, va="bottom", ha="left")
ax.text(0.0, 1.03, "libres", fontsize=9, color=INK2, ha="center", va="bottom")
ax.text(290, 2.03, "Tonks–Girardeau\n= fermiones polarizados", fontsize=9, color=INK2, ha="right", va="bottom")
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig("../fig/figB_cociente.pdf"); plt.close(fig)

json.dump({"delta1_q0225": ref[0.225], "delta1_q0713": ref[0.713],
           "eq11_q0225": float(np.interp(np.log(0.225), np.log(qs), d11))}, open("nums.json","w"), indent=1)
print(open("nums.json").read())
print(R[[0,10,18,25,35,-1]].round(4))
