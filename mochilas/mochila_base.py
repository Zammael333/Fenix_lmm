"""
Definición base de Mochila Modular de Pesos y Capacidades Tácticas.
Permite la descarga y montaje en caliente de pesos para microagentes.
"""
import os
import numpy as np

class MochilaBase:
    def __init__(self, name: str, size_floats: int = 1024 * 1024): # ~4MB por defecto
        self.name = name
        self.size_floats = size_floats
        self.is_loaded = False
        self.weights = None

    def initialize_synthetic_weights(self, file_path: str):
        """Genera y guarda un bloque de pesos sintéticos normalizados si no existe."""
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            data = np.random.randn(self.size_floats).astype(np.float32)
            with open(file_path, "wb") as f:
                f.write(data.tobytes())
