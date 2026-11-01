"""Estado de Busch 1D (relativa, unidades hbar=mu=omega=1 => ell=1, m=2) y regla de suma."""
import numpy as np
from scipy.special import pbdv, rgamma
from scipy.optimize import brentq
from scipy.integrate import quad

def nu_ground(g):
    """Rama fundamental par: g*rgamma((1-nu)/2) + 2*rgamma(-nu/2) = 0"""
    if g == 0: return 0.0
    f = lambda n: g*rgamma((1-n)/2) + 2*rgamma(-n/2)
    if g > 0: return brentq(f, 1e-12, 1-1e-12)
    lo = -g*g - 6.0
    return brentq(f, lo, -1e-12)

def phi0(r, nu):
    return pbdv(nu, np.sqrt(2)*np.abs(r))[0]

def state(g):
    nu = nu_ground(g)
    N2 = 2*quad(lambda r: phi0(r,nu)**2, 0, 14, limit=400)[0]
    return nu, 1/np.sqrt(N2)

def sumrule(g):
    """S = <T_rel + U> por virial + Hellmann-Feynman, E = nu+1/2"""
    nu = nu_ground(g); E = nu + 0.5
    h = 1e-5*max(1, abs(g))
    dE = (nu_ground(g+h) - nu_ground(g-h))/(2*h) if g != 0 else (nu_ground(h)-nu_ground(-h))/(2*h)
    U = g*dE
    return E, U, E/2 + U/4

if __name__ == "__main__":
    # verificacion contra diagonalizacion en malla (delta = g/dr en r=0)
    L, n = 14.0, 2801
    r = np.linspace(-L, L, n); dr = r[1]-r[0]
    for g in [-2.0, -0.5, 0.0, 1.0, 5.0, 50.0]:
        main = 1/dr**2 + 0.5*r**2; off = -0.5/dr**2*np.ones(n-1)
        H = np.diag(main) + np.diag(off,1) + np.diag(off,-1)
        H[n//2, n//2] += g/dr
        w, v = np.linalg.eigh(H); psi = v[:,0]/np.sqrt(dr)
        Ugrid = g*psi[n//2]**2
        Tgrid = np.sum(np.conj(psi)*(H@psi - (0.5*r**2)*psi))*dr - Ugrid
        E, U, S = sumrule(g)
        print(f"g={g:6.1f}  E_busch={E:.5f} E_grid={w[0]:.5f} | S_rule={S:.5f} S_grid={Tgrid+Ugrid:.5f}")
