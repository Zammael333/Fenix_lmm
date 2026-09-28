"""Microagente Sargento SRE (Site Reliability Engineering & Confinamiento de Recursos)."""
import os
import time
from enjambre.agentes.agente_base import AgenteBase

class AgenteSRE(AgenteBase):
    def __init__(self, agent_id: str = "agente_sre"):
        super().__init__(agent_id, "SRE_SYSTEM_RELIABILITY")

    def _run_logic(self, task: dict) -> dict:
        # Extraer carga de CPU promedio (1m, 5m, 15m)
        load_1, load_5, load_15 = os.getloadavg()
        # Estado térmico preventivo
        thermal_celsius = 45.0
        try:
            # Intentar leer sensor de temperatura térmico en Linux si existe
            thermal_path = "/sys/class/thermal/thermal_zone0/temp"
            if os.path.exists(thermal_path):
                with open(thermal_path, "r") as f:
                    thermal_celsius = float(f.read().strip()) / 1000.0
        except Exception:
            pass

        dpmo = (self.error_count / max(self.total_ops, 1)) * 1_000_000
        return {
            "load_avg": [load_1, load_5, load_15],
            "thermal_celsius": thermal_celsius,
            "total_ops": self.total_ops,
            "error_count": self.error_count,
            "dpmo": round(dpmo, 4),
            "seis_sigmas_compliant": dpmo <= 3.4
        }
