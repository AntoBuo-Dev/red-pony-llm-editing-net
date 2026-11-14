"""Free hidden units by folding them (least squares) into the remaining units of the same layer."""
import numpy as np
from sim_harness import *

def elu(x): return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))

def free_units(L, O, counts, ridge=1e-3, verbose=True):
    """L: list of (W,b); O: observation samples; counts: units to free in hidden layers 1..3.
    Returns new layers (freed units have zero in/out weights) and the freed indices."""
    L = [(W.astype(np.float64).copy(), b.astype(np.float64).copy()) for W, b in L]
    freed = []
    x = O.astype(np.float64)
    for li, k in enumerate(counts):
        W, b = L[li]
        h = elu(x @ W.T + b)                      # activations of this layer (with upstream edits)
        Wn, bn = L[li + 1]
        mu = h.mean(0); hc = h - mu
        C = hc.T @ hc / len(h)
        alive = np.ones(h.shape[1], bool); chosen = []
        for _ in range(k):
            idx = np.where(alive)[0]
            P = np.linalg.inv(C[np.ix_(idx, idx)] + ridge * np.eye(len(idx)))
            resid = 1.0 / np.diag(P)
            score = np.sqrt(resid) * np.linalg.norm(Wn[:, idx], axis=0)
            j = idx[np.argmin(score)]
            # regression of unit j on the other alive units: coef = -P[j,others]/P[j,j]
            jj = np.where(idx == j)[0][0]
            coef = -P[jj] / P[jj, jj]; coef[jj] = 0
            Wn[:, idx] += np.outer(Wn[:, j], coef)
            bn += Wn[:, j] * (mu[j] - coef @ mu[idx])
            Wn[:, j] = 0
            alive[j] = False; chosen.append(int(j))
        W[chosen, :] = 0; b[chosen] = 0
        L[li] = (W, b); L[li + 1] = (Wn, bn)
        freed.append(chosen)
        x = elu(x @ W.T + b)
    return [(W.astype(np.float32), b.astype(np.float32)) for W, b in L], freed

