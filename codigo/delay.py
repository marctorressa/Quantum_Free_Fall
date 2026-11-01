"""Retraso cuántico del constituyente en caída libre, interacción de contacto.
Unidades relativas hbar=mu=omega=1 (m=2). sigma_v(1 particula libre)=1/2.
w = v/sigma_v ; T/t_c = sqrt(1+q^2 w^2) - q w ;  delta = E[sqrt(1+q^2w^2)] - 1."""
import numpy as np
from scipy.integrate import quad
from busch import nu_ground, state, phi0, sumrule

def overlap_k(k, nu, N, g):
    th = -np.arctan2(g, k)          # tan(theta) = -g/k
    f = lambda r: np.cos(k*r + th)*N*phi0(r, nu)
    return 2*quad(f, 0, 12, limit=400)[0]

def asym(g, kmax=8.0, nk=801):
    nu, N = state(g)
    k = np.linspace(1e-4, kmax, nk)
    wk = np.array([overlap_k(kk, nu, N, g) for kk in k])**2/np.pi
    Pb = 0.0
    if g < 0:
        kap = -g
        Pb = (2*quad(lambda r: np.sqrt(kap)*np.exp(-kap*r)*N*phi0(r,nu), 0, 40, limit=400)[0])**2
    return k, wk, Pb

def delta(q, g, k, wk, Pb, nG=80):
    x, wh = np.polynomial.hermite.hermgauss(nG)
    G = x*np.sqrt(2)*np.sqrt(0.5)          # N(0,1/2)
    pG = wh/np.sqrt(np.pi)
    from scipy.integrate import trapezoid
    tot = 0.0
    # componente ligada: deriva solo con V
    tot += Pb*np.sum(pG*np.sqrt(1+q**2*G**2))
    # componente libre: w = G +/- k  (u/(2 sigma_v) = k)
    for s in (+1, -1):
        W = G[:,None] + s*k[None,:]
        tot += 0.5*trapezoid(np.sum(pG[:,None]*wk[None,:]*np.sqrt(1+q**2*W**2),axis=0), k)
    norm = Pb + trapezoid(wk, k)
    return tot/norm - 1

if __name__ == "__main__":
    for g in [0.0, 1.0, 5.0, 1e4, -1.0, -3.0]:
        k, wk, Pb = asym(g)
        dk = k[1]-k[0]
        norm = np.sum(wk)*dk + Pb
        Ew2_rule = 0.5 + 2*(sumrule(g)[2] - Pb*(-g*g/2))   # ratio semiclasico
        Ew2_num = 0.5*(1-Pb)*0 + 0.5 + np.sum(wk*k**2)*dk   # <G^2>=1/2 (todas) + <k^2> libres
        print(f"g={g:8.1f} norma={norm:.4f} Pb={Pb:.4f}  E[w2]: regla={Ew2_rule:.4f} num={Ew2_num:.4f}"
              f"  d(q=0.225)/d1={delta(0.225,g,k,wk,Pb)/delta(0.225,0,*asym(0)):.4f}")
