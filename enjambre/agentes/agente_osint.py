"""Microagente Sargento OSINT (Open Source Intelligence y Análisis de Red)."""
import socket
from enjambre.agentes.agente_base import AgenteBase

class AgenteOSINT(AgenteBase):
    def __init__(self, agent_id: str = "agente_osint"):
        super().__init__(agent_id, "OSINT_RECONNAISSANCE")

    def _run_logic(self, task: dict) -> dict:
        target_host = task.get("target_host", "localhost")
        try:
            ip = socket.gethostbyname(target_host)
            status = "RESOLVED"
        except Exception as e:
            ip = None
            status = f"FAILED: {e}"

        return {
            "target": target_host,
            "ip_address": ip,
            "resolution_status": status,
            "ports_probed": [9000, 9999]
        }
