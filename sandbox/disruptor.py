"""Agente Disruptor (Red Team): Inyecta fuzzing, ruido y vectores de ataque estocásticos."""
import numpy as np

class Disruptor:
    def __init__(self):
        self.attack_vectors = [
            {"p": 0.0, "r": 0.0, "desc": "Denominador Cero Estricto"},
            {"p": -10.5, "r": 10.5, "desc": "Cancelación Catastrófica (Varianza Negativa)"},
            {"p": 1e12, "r": 1e-12, "desc": "Escala Numérica Extrema"},
            {"p": float("nan"), "r": 1.0, "desc": "Inyección de NaN"},
            {"p": 0.5, "r": 0.5, "desc": "Condición Nominal"},
        ]

    def generate_stress_cases(self):
        return self.attack_vectors
