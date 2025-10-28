# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 23:08:41 2025

@author: Wilfredo
"""

from etl import ImageProcessor
from classifier import SVDClassifier, load_data_from_db

if __name__ == "__main__":
    print("🚀 Iniciando FaceMind...")

    # Paso 1️⃣ ETL
    proc = ImageProcessor(r"C:\\Users\\roger\\OneDrive\\Desktop\\face_mind\\img_sp")
    proc.extract()
    proc.transform()
    proc.load_to_db()
    proc.matrices_por_clase()

    # Paso 2️⃣ Clasificación
    X_abiertos, X_cerrados = load_data_from_db("imagenes.db")

    modelo_abiertos = SVDClassifier(20)
    modelo_cerrados = SVDClassifier(20)
    modelo_abiertos.fit(X_abiertos)
    modelo_cerrados.fit(X_cerrados)

    # Paso 3️⃣ Prueba
    print("\n🧪 Probando con una imagen...")
    x_test = X_abiertos[:, 1]
    pred = SVDClassifier.classify(x_test, modelo_abiertos, modelo_cerrados)

    print("🔎 Resultado final:", pred)
