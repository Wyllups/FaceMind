# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 23:14:58 2025

@author: Wilfredo
"""

import sqlite3
import numpy as np
from svd_algorithm import custom_svd


class SVDClassifier:
    """Clasificador basado en SVD para distinguir entre 'abiertos' y 'cerrados'."""

    def __init__(self, k_components=10):
        self.k = k_components
        self.media = None
        self.Uk = None

    def fit(self, X_train):
        print("🔹 Entrenando modelo...")
        self.media = np.mean(X_train, axis=1)
        Xc = X_train - self.media[:, None]
        U, S, Vt = custom_svd(Xc)
        self.Uk = U[:, :self.k]
        print(f"✅ Modelo entrenado con {self.k} componentes principales.")

    def reconstruct(self, x):
        x_centrada = x - self.media
        proyeccion = np.dot(self.Uk.T, x_centrada)
        return np.dot(self.Uk, proyeccion) + self.media

    def reconstruction_error(self, x):
        return np.linalg.norm(x - self.reconstruct(x))

    @staticmethod
    def classify(x, modelo_abiertos, modelo_cerrados):
        err_a = modelo_abiertos.reconstruction_error(x)
        err_c = modelo_cerrados.reconstruction_error(x)
        print(f"Error abiertos: {err_a:.4f} | Error cerrados: {err_c:.4f}")
        return "abiertos" if err_a < err_c else "cerrados"


def load_data_from_db(db_path="imagenes.db"):
    with sqlite3.connect(db_path) as conn:
        data = conn.execute("SELECT etiqueta, vector FROM imagenes").fetchall()

    etiquetas, vectores = [], []
    for etiqueta, vector_blob in data:
        vectores.append(np.frombuffer(vector_blob, dtype=np.float32))
        etiquetas.append(etiqueta)

    X = np.array(vectores).T
    y = np.array(etiquetas)

    print(f"✅ {X.shape[1]} imágenes cargadas desde '{db_path}'.")
    return X[:, y == "abiertos"], X[:, y == "cerrados"]
