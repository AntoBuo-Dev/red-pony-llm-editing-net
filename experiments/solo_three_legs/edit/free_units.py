"""Free hidden units by folding them (least squares) into the remaining units of the same layer.

Same method as the H1 heading-hold edit (evaluation/heading_hold/free_units.py in the H1 repo):
for each layer, repeatedly pick the unit whose activation is best predicted by the other alive units
(residual std x norm of its outgoing weights), express it as an affine function of the others and move
its outgoing weights onto them (and onto the next bias). Its incoming and outgoing weights become zero.
"""
import numpy as np


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def free_units(L, X, counts, ridge=1e-4):
    """L: list of (W, b) (float64); X: observation samples; counts: units to free in hidden layers 1..3.
    Returns (new layers, freed indices per layer, residual std of each freed unit)."""
    L = [(W.astype(np.float64).copy(), b.astype(np.float64).copy()) for W, b in L]
    freed, resid_std = [], []
    x = np.asarray(X, np.float64)
    for li, k in enumerate(counts):
        W, b = L[li]
        h = elu(x @ W.T + b)
        Wn, bn = L[li + 1]
        mu = h.mean(0)
        hc = h - mu
        C = hc.T @ hc / len(h)
        alive = np.ones(h.shape[1], bool)
        chosen, rs = [], []
        for _ in range(k):
            idx = np.where(alive)[0]
            P = np.linalg.inv(C[np.ix_(idx, idx)] + ridge * np.eye(len(idx)))
            resid = 1.0 / np.diag(P)                       # residual variance of each unit on the others
            score = np.sqrt(resid) * np.linalg.norm(Wn[:, idx], axis=0)
            jj = int(np.argmin(score)); j = idx[jj]
            coef = -P[jj] / P[jj, jj]; coef[jj] = 0.0      # regression of unit j on the other alive units
            Wn[:, idx] += np.outer(Wn[:, j], coef)
            bn += Wn[:, j] * (mu[j] - coef @ mu[idx])
            Wn[:, j] = 0.0
            alive[j] = False
            chosen.append(int(j)); rs.append(float(np.sqrt(resid[jj])))
        W[chosen, :] = 0.0; b[chosen] = 0.0
        L[li] = (W, b); L[li + 1] = (Wn, bn)
        freed.append(chosen); resid_std.append(rs)
        x = elu(x @ W.T + b)
    return L, freed, resid_std
