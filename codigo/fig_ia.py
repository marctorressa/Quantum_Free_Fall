import json, numpy as np
import matplotlib; matplotlib.use("pdf"); import matplotlib.pyplot as plt
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; INK2, GRID = "#5f5e5a", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2})
a = json.load(open("ia_poc.json")); b = json.load(open("ia_nuis.json"))
Nb = np.array([int(k) for k in b]); nn_b = np.array([v[0] for v in b.values()]); mo_b = np.array([v[1] for v in b.values()])
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.loglog(a["N"], a["rmse_mom"], "-", color=C[0], marker="o", ms=6, label=r"clásico (varianza), $q$ conocido")
ax.loglog(a["N"], a["rmse_nn"], "--", color=C[2], marker="s", ms=6, label=r"red neuronal, $q$ conocido")
ax.loglog(Nb, mo_b, "-", color=C[1], marker="o", ms=7, label=r"clásico, $q$ con $\pm20\,\%$ de incertidumbre")
ax.loglog(Nb, nn_b, "--", color=C[3], marker="s", ms=7, lw=2.4, label=r"red neuronal, $q$ con $\pm20\,\%$")
ax.set_xlabel(r"número de átomos detectados $N$")
ax.set_ylabel(r"error (RMS) en $\delta^{(1)}/\delta_1$ inferido")
ax.grid(True, which="major", color=GRID, lw=0.8); ax.legend(frameon=False, fontsize=9.2, loc="lower left")
ax.set_ylim(0.015, 0.4)
fig.tight_layout(); fig.savefig("../fig/figC_ia.pdf")
