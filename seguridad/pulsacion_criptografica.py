"""
Módulo de Dinámica de Pulsaciones Criptográficas (Keystroke Dynamics).
Analiza el tiempo de vuelo (Flight Time) y la variabilidad (Jitter) entre
pulsaciones de teclado para validar la autenticidad del operador legítimo
frente a scripts automatizados o intrusos físicos.
"""
import numpy as np
import logging

logger = logging.getLogger("fenix.seguridad.pulsacion")

class PulsacionCriptografica:
    def __init__(self, jitter_tolerance_ms: float = 25.0, baseline_jitter_ms: float = 4.5):
        self.jitter_tolerance_ms = jitter_tolerance_ms
        self.baseline_jitter_ms = baseline_jitter_ms

    def evaluate_keystrokes(self, flight_times_ms: list[float]) -> dict:
        """
        Evalúa un vector de intervalos entre pulsaciones (ms).
        Retorna dict con métricas y veredicto de seguridad:
          - "SOVEREIGN_AUTHORIZED": Operador humano legítimo dentro de la cadencia.
          - "ROBOTIC_INJECTION": Disparado por scripts automatizados (jitter nulo o constante).
          - "ANOMALOUS_OPERATOR": Disparado por intruso (varianza excesiva).
        """
        if len(flight_times_ms) < 3:
            return {
                "verdict": "INSUFFICIENT_DATA",
                "mean_ms": 0.0,
                "jitter_ms": 0.0,
                "authorized": False
            }

        intervals = np.array(flight_times_ms, dtype=np.float64)
        mean_ms = float(np.mean(intervals))
        jitter_ms = float(np.std(intervals))

        # Detección de inyección de script automatizado (sin variabilidad humana natural)
        if jitter_ms < 0.5:
            logger.critical(f"INYECCIÓN AUTOMATIZADA DETECTADA: Jitter sintético nulo ({jitter_ms:.2f} ms)")
            return {
                "verdict": "ROBOTIC_INJECTION",
                "mean_ms": mean_ms,
                "jitter_ms": jitter_ms,
                "authorized": False
            }

        # Detección de intruso (desviación abrupta sobre el umbral de tolerancia)
        if abs(jitter_ms - self.baseline_jitter_ms) > self.jitter_tolerance_ms:
            logger.warning(f"DESVIACIÓN CONDUCTUAL DETECTADA: Jitter anómalo ({jitter_ms:.2f} ms > tolerancia)")
            return {
                "verdict": "ANOMALOUS_OPERATOR",
                "mean_ms": mean_ms,
                "jitter_ms": jitter_ms,
                "authorized": False
            }

        return {
            "verdict": "SOVEREIGN_AUTHORIZED",
            "mean_ms": mean_ms,
            "jitter_ms": jitter_ms,
            "authorized": True
        }

if __name__ == "__main__":
    pc = PulsacionCriptografica()
    # 1. Humano legítimo
    legit = [120.1, 125.4, 118.2, 122.9, 124.0, 119.5]
    print("Legítimo:", pc.evaluate_keystrokes(legit))

    # 2. Script robótico
    robot = [10.0, 10.0, 10.0, 10.0, 10.0]
    print("Robot:", pc.evaluate_keystrokes(robot))

    # 3. Intruso errático
    intruder = [12.0, 350.0, 45.0, 890.0, 20.0]
    print("Intruso:", pc.evaluate_keystrokes(intruder))
