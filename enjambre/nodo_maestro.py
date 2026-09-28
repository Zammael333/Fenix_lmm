"""
Nodo Maestro del Enjambre de Fénix LLM.
Coordina las misiones de los sargentos, sincroniza el paginador latente de memoria
y se conecta mediante cliente asíncrono mTLS con el Kernel Central.
"""
import os
import sys
import ssl
import json
import asyncio
import logging
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from enjambre.agentes.agente_sre import AgenteSRE
from enjambre.agentes.agente_osint import AgenteOSINT
from enjambre.agentes.agente_forense import AgenteForense
from enjambre.agentes.agente_marketing import AgenteMarketing
from mochilas.paginador_latente import PaginadorLatente
from routing.kalman_estimator import KalmanStateEstimator
from seguridad.cert_manager import CertManager

logger = logging.getLogger("fenix.enjambre.maestro")

class NodoMaestro:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.mmap_store_dir = self.config.get("paging", {}).get("mmap_dir", "/tmp/fenix_mmap")
        self.paginador = PaginadorLatente(self.mmap_store_dir, max_resident_agents=2)
        self.kalman = KalmanStateEstimator(num_agents=4)

        # Registro de sargentos base
        self.agentes = {
            "agente_sre": AgenteSRE("agente_sre"),
            "agente_osint": AgenteOSINT("agente_osint"),
            "agente_forense": AgenteForense("agente_forense"),
            "agente_marketing": AgenteMarketing("agente_marketing"),
        }
        self.agent_keys = list(self.agentes.keys())

    def dispatch_mission(self, agent_name: str, task: dict) -> dict:
        """
        Despacha una misión al microagente especificado:
        1. Predice y actualiza probabilidades en el Filtro de Kalman.
        2. Carga/swappea pesos en memoria mediante mmap y evicción guiada.
        3. Ejecuta la misión en el agente.
        4. Desasocia punteros para permitir evicción limpia.
        """
        if agent_name not in self.agentes:
            return {"status": "ERROR", "error": f"Agente '{agent_name}' no reconocido."}

        # 1. Kalman Step: registrar demanda para este agente
        agent_idx = self.agent_keys.index(agent_name)
        u = [0.0] * len(self.agent_keys)
        u[agent_idx] = 1.0
        self.kalman.predict(u=u)
        probs_array = self.kalman.get_probabilities()
        probs_dict = {name: float(probs_array[i]) for i, name in enumerate(self.agent_keys)}

        # 2. Paging Step: Swapping con evicción predictiva
        weights = self.paginador.load_agent(agent_name, agent_probs=probs_dict)
        agente = self.agentes[agent_name]
        agente.attach_weights(weights)

        # 3. Execution Step
        result = agente.execute_mission(task)

        # 4. Detach weights
        agente.detach_weights()

        result["kalman_probabilities"] = probs_dict
        result["resident_in_ram"] = list(self.paginador.active_pages.keys())
        return result

    async def connect_to_kernel(self):
        """Conecta el cliente del sargento al Kernel Central mediante mTLS asíncrono."""
        host = self.config["network"]["host"]
        port = self.config["network"]["port_maestro"]
        cert_dir = os.path.join(PROJECT_ROOT, "certs")

        cm = CertManager(cert_dir)
        certs = cm.generate_all_certs()

        ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=certs["ca_cert"])
        ssl_ctx.load_cert_chain(certfile=certs["client_cert"], keyfile=certs["client_key"])
        # server_hostname=None con context configurado para localhost / 127.0.0.1
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_REQUIRED

        reader, writer = await asyncio.open_connection(host, port, ssl=ssl_ctx)
        logger.info(f"Conexión mTLS establecida exitosamente con el Kernel Central en {host}:{port}")

        # Enviar latido inicial
        heartbeat_msg = json.dumps({"type": "HEARTBEAT", "timestamp": "INITIAL_SYNC"}) + "\n"
        writer.write(heartbeat_msg.encode("utf-8"))
        await writer.drain()

        reply = await reader.readline()
        logger.info(f"Respuesta del Kernel Central: {reply.decode().strip()}")

        writer.close()
        await writer.wait_closed()

if __name__ == "__main__":
    cfg = os.path.join(PROJECT_ROOT, "config.yaml")
    maestro = NodoMaestro(cfg)

    print("--- Probando Despacho de Misión: Agente Marketing ---")
    res1 = maestro.dispatch_mission("agente_marketing", {"prompt": "Por favor podrías optimizar este prompt para el nodo"})
    print("Resultado 1:", json.dumps(res1, indent=2))

    print("\n--- Probando Despacho de Misión: Agente SRE ---")
    res2 = maestro.dispatch_mission("agente_sre", {})
    print("Resultado 2:", json.dumps(res2, indent=2))

    print("\n--- Probando Despacho de Misión: Agente OSINT ---")
    res3 = maestro.dispatch_mission("agente_osint", {"target_host": "127.0.0.1"})
    print("Resultado 3:", json.dumps(res3, indent=2))

    maestro.paginador.close_all()
    print("\nPrueba de Nodo Maestro finalizada con éxito.")
