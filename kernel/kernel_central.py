"""
Nodo Núcleo Central (Kernel Central) de Fénix LLM.
Coordina el clúster heterogéneo mediante un bus de sockets asíncronos TLS con mTLS estricto.
Monitorea el presupuesto de error Seis Sigmas y gestiona el ciclo de vida de los agentes.
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

from routing.capa_cero_router import CapaCeroRouter
from routing.kalman_estimator import KalmanStateEstimator
from seguridad.cert_manager import CertManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [KERNEL_CENTRAL] - %(levelname)s - %(message)s")
logger = logging.getLogger("fenix.kernel")

class KernelCentral:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.host = self.config["network"]["host"]
        self.port = self.config["network"]["port_maestro"]
        self.cert_dir = os.path.join(PROJECT_ROOT, "certs")

        # Asegurar certificados con SAN
        cm = CertManager(self.cert_dir)
        self.certs = cm.generate_all_certs(overwrite=False)

        self.router = CapaCeroRouter(config_path)
        self.kalman = KalmanStateEstimator(num_agents=self.config["kalman"]["state_dimension"])
        self.clients = {} # {addr: writer}
        self.running = False
        self.server = None

    def _build_ssl_context(self) -> ssl.SSLContext:
        """Configura contexto SSL del servidor con autenticación mutua (mTLS)."""
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.load_cert_chain(certfile=self.certs["server_cert"], keyfile=self.certs["server_key"])
        ctx.load_verify_locations(cafile=self.certs["ca_cert"])
        ctx.verify_mode = ssl.CERT_REQUIRED
        return ctx

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")
        logger.info(f"Conexión mTLS autorizada con nodo sargento: {addr}")
        self.clients[addr] = writer

        try:
            while self.running:
                data = await reader.readline()
                if not data:
                    break
                line = data.decode("utf-8").strip()
                if not line:
                    continue

                try:
                    payload = json.loads(line)
                except Exception:
                    payload = {"type": "RAW_TEXT", "content": line}

                msg_type = payload.get("type", "UNKNOWN")

                if msg_type == "HEARTBEAT":
                    resp = {"status": "ACK_HEARTBEAT", "timestamp": payload.get("timestamp")}
                    writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                    await writer.drain()

                elif msg_type == "DEMAND_EVENT":
                    # Actualización de Kalman guiada por evento
                    event_agent_idx = payload.get("agent_idx", 0)
                    u = np.zeros(self.kalman.N)
                    u[event_agent_idx] = 1.0
                    self.kalman.predict(u=u)
                    probs = self.kalman.get_probabilities().tolist()
                    broadcast_msg = json.dumps({"type": "STATE_UPDATE", "probs": probs}) + "\n"
                    # Broadcast a todos los sargentos conectados
                    for client_writer in list(self.clients.values()):
                        try:
                            client_writer.write(broadcast_msg.encode("utf-8"))
                            await client_writer.drain()
                        except Exception:
                            pass

                elif msg_type == "USER_INPUT":
                    content = payload.get("content", "")
                    route_status, route_out = self.router.route(content)
                    resp = {
                        "status": route_status,
                        "output": route_out,
                        "probs": self.kalman.get_probabilities().tolist()
                    }
                    writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                    await writer.drain()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error en socket seguro de {addr}: {e}")
        finally:
            logger.warning(f"Desconectando sargento {addr}")
            if addr in self.clients:
                del self.clients[addr]
            writer.close()
            await writer.wait_closed()

    async def start(self):
        ssl_ctx = self._build_ssl_context()
        self.running = True
        self.server = await asyncio.start_server(
            self.handle_client, self.host, self.port, ssl=ssl_ctx
        )
        logger.info(f"Kernel Central mTLS operativo en {self.host}:{self.port}")
        async with self.server:
            await self.server.serve_forever()

    def stop(self):
        self.running = False
        if self.server:
            self.server.close()

if __name__ == "__main__":
    import numpy as np
    cfg = os.path.join(PROJECT_ROOT, "config.yaml")
    kernel = KernelCentral(cfg)
    try:
        asyncio.run(kernel.start())
    except KeyboardInterrupt:
        logger.info("Kernel Central detenido por el operador.")
