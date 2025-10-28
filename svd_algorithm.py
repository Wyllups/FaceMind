# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 23:16:12 2025

@author: Wilfredo
"""

import numpy as np

def custom_svd(A):
    """
    Implementación personalizada de la Descomposición en Valores Singulares (SVD)
    """
    B = np.dot(A.T, A)
    eigvals, eigvecs = np.linalg.eig(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    V = eigvecs
    sigma = np.sqrt(np.clip(eigvals, a_min=0, a_max=None))

    m, n = A.shape
    Sigma = np.zeros((m, n))
    np.fill_diagonal(Sigma, sigma)

    U = np.zeros((m, len(sigma)))
    for i in range(len(sigma)):
        if sigma[i] > 1e-10:
            U[:, i] = (A @ V[:, i]) / sigma[i]

    for i in range(U.shape[1]):
        norm = np.linalg.norm(U[:, i])
        if norm > 0:
            U[:, i] /= norm

    Vt = V.T
    print("✅ SVD calculada con éxito.")
    return U, Sigma, Vt
