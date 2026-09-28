"""
Clase Base de Microagente Heterogéneo (~100 MB de huella).
Define el ciclo de vida, presupuesto de error y ejecución asíncrona.
"""
import os
import sys
import time
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("fenix.enjambre.agente")

class AgenteBase:
    def __init__(self, agent_id: str, role: str):
        self.agent_id = agent_id
        self.role = role
        self.error_count = 0
        self.total_ops = 0
        self.weights = None
        self.is_active = False

    def attach_weights(self, weights_array):
        """Asocia la vista de pesos mmap cargada por el paginador latente."""
        self.weights = weights_array
        self.is_active = True

    def detach_weights(self):
        """Desasocia los pesos para permitir la evicción de RAM sin retención de referencias."""
        self.weights = None
        self.is_active = False

    def execute_mission(self, task: dict) -> dict:
        """Punto de entrada de misión del agente con registro de telemetría."""
        self.total_ops += 1
        start_t = time.perf_counter()
        try:
            result = self._run_logic(task)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return {
                "agent_id": self.agent_id,
                "role": self.role,
                "status": "SUCCESS",
                "elapsed_ms": round(elapsed_ms, 2),
                "data": result
            }
        except Exception as e:
            self.error_count += 1
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.error(f"Fallo en misión de '{self.agent_id}': {e}")
            return {
                "agent_id": self.agent_id,
                "role": self.role,
                "status": "ERROR",
                "error": str(e),
                "elapsed_ms": round(elapsed_ms, 2)
            }

    def _run_logic(self, task: dict) -> dict:
        raise NotImplementedError("Debe ser implementado por la especialización del agente.")
