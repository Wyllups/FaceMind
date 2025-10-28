# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 23:16:56 2025

@author: Wilfredo
"""

# -*- coding: utf-8 -*-
"""
FaceMind – Módulo ETL (Extract, Transform, Load)
Procesa las imágenes, las vectoriza, guarda la información en SQLite y calcula matrices por clase.
"""

import os
import sqlite3
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


class ImageProcessor:
    """
    Pipeline ETL de imágenes:
    - Extrae imágenes desde carpetas 'abiertos' y 'cerrados'
    - Convierte cada imagen en vector plano normalizado
    - Apila los vectores uno al lado del otro → matriz (n_pixeles × n_imágenes)
    - Divide el dataset en 80% entrenamiento y 20% prueba
    - Calcula medias y matrices cuadradas (X·Xᵀ)
    """

    def __init__(self, dataset_path, db_name="imagenes.db",
                 image_size=(200, 200), crop_box=(5, 40, 190, 150),
                 test_size=0.2, random_state=42):

        self.dataset_path = dataset_path
        self.image_size = image_size
        self.crop_box = crop_box
        self.db_name = db_name
        self.test_size = test_size
        self.random_state = random_state

        # Inicialización de variables
        self.image_paths = []
        self.labels = []
        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.X_abiertos_train = None
        self.X_cerrados_train = None
        self.media_abiertos = None
        self.media_cerrados = None

        self._init_db()

    # ===============================================================
    # Inicializar base de datos
    # ===============================================================
    def _init_db(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS imagenes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    etiqueta TEXT,
                    vector BLOB
                )
            """)
            conn.commit()
        print(f"✅ Base de datos '{self.db_name}' lista.")

    # ===============================================================
    # Extracción de imágenes
    # ===============================================================
    def extract(self):
        print("📁 Extrayendo rutas de imágenes...")
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"No existe la ruta: {self.dataset_path}")

        for label in sorted(os.listdir(self.dataset_path)):
            class_dir = os.path.join(self.dataset_path, label)
            if not os.path.isdir(class_dir):
                continue
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, file))
                    self.labels.append(label)

        if not self.image_paths:
            raise ValueError("⚠ No se encontraron imágenes en el dataset.")

        print(f"✅ {len(self.image_paths)} imágenes encontradas.")
        print(f"📋 Clases detectadas: {set(self.labels)}")

    # ===============================================================
    # Transformación de imágenes en vectores
    # ===============================================================
    def transform(self):
        print("🧠 Procesando y vectorizando imágenes...")
        vectores, etiquetas = [], []

        for path, label in zip(self.image_paths, self.labels):
            try:
                img = Image.open(path).convert("L")
                img_resized = img.resize(self.image_size, Image.Resampling.LANCZOS)
                img_crop = img_resized.crop(self.crop_box)
                img_array = np.array(img_crop, dtype=np.float32) / 255.0
                vectores.append(img_array.flatten())
                etiquetas.append(label)
            except Exception as e:
                print(f"⚠ Error al procesar {path}: {e}")

        self.X = np.column_stack(vectores)
        self.y = np.array(etiquetas)

        X_T = self.X.T
        X_train_T, X_test_T, y_train, y_test = train_test_split(
            X_T, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )

        self.X_train = X_train_T.T
        self.X_test = X_test_T.T
        self.y_train = y_train
        self.y_test = y_test

        # Separar por clase
        self.X_abiertos_train = self.X_train[:, self.y_train == 'abiertos']
        self.X_cerrados_train = self.X_train[:, self.y_train == 'cerrados']

        np.save("X_train.npy", self.X_train)
        np.save("X_test.npy", self.X_test)
        np.save("y_train.npy", self.y_train)
        np.save("y_test.npy", self.y_test)

        print(f"✅ Matriz X creada con forma {self.X.shape}")
        print(f"📊 Entrenamiento: {self.X_train.shape[1]} imágenes")
        print(f"🧩 Prueba: {self.X_test.shape[1]} imágenes")

    # ===============================================================
    # Cargar en base de datos
    # ===============================================================
    def load_to_db(self):
        if self.X is None or self.y is None:
            raise RuntimeError("Ejecuta transform() antes de load_to_db().")

        print("💾 Guardando vectores en la base de datos...")
        datos = [(label, vector.tobytes()) for label, vector in zip(self.y, self.X.T)]

        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO imagenes (etiqueta, vector)
                VALUES (?, ?)
            """, datos)
            conn.commit()

        print(f"✅ {len(self.y)} registros insertados en '{self.db_name}'.")

    # ===============================================================
    # Calcular matrices por clase
    # ===============================================================
    def matrices_por_clase(self):
        if self.X_abiertos_train is None or self.X_cerrados_train is None:
            raise RuntimeError("Ejecuta transform() antes de matrices_por_clase().")

        print("📊 Calculando medias y matrices cuadradas (X·Xᵀ)...")

        self.media_abiertos = np.mean(self.X_abiertos_train, axis=1)
        self.media_cerrados = np.mean(self.X_cerrados_train, axis=1)

        Xc_abiertos = self.X_abiertos_train - self.media_abiertos[:, None]
        Xc_cerrados = self.X_cerrados_train - self.media_cerrados[:, None]

        G_abiertos = np.dot(Xc_abiertos, Xc_abiertos.T)
        G_cerrados = np.dot(Xc_cerrados, Xc_cerrados.T)

        np.save("media_abiertos.npy", self.media_abiertos)
        np.save("media_cerrados.npy", self.media_cerrados)
        np.save("G_abiertos.npy", G_abiertos)
        np.save("G_cerrados.npy", G_cerrados)

        print("✅ Medias y matrices cuadradas calculadas y guardadas.")
        return G_abiertos, G_cerrados


if __name__ == "__main__":
    ruta_dataset = r"C:\\Users\\roger\\OneDrive\\Desktop\\face_mind\\img_sp"

    proc = ImageProcessor(ruta_dataset)
    proc.extract()
    proc.transform()
    proc.load_to_db()
    proc.matrices_por_clase()

    print("\n✅ Pipeline completado correctamente.")
