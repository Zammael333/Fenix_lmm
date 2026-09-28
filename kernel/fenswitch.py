"""
Demonio de Kill Switch Out-of-Band en Python puro (< 2MB RAM).
Escucha peticiones en el puerto 9999 para ejecutar cortes de emergencia
con independencia del hilo principal del enjambre o del orquestador.
"""
import os
import sys
import socket
import select
import subprocess
import logging
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.getenv("FENIX_KILL_PORT", "9999"))
TOKEN_KILL = os.getenv("FENIX_KILL_TOKEN", "RED_ALERT_KILL_33")
LOG_DIR = Path(os.getenv("FENIX_AUDIT_DIR", str(PROJECT_ROOT / "audit")))
LOG_FILE = LOG_DIR / "kill_switch.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [KILL_SWITCH] - %(levelname)s - %(message)s")

def log_audit(action: str):
    timestamp = datetime.datetime.now().isoformat()
    line = f"{timestamp} - [AUDIT_IMMUNE] - {action}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    logging.warning(action)

def execute_kill_level(payload: str):
    if "STOP_SANDBOX" in payload:
        log_audit("Nivel 1 Activado: STOP_SANDBOX. Deteniendo procesos de sandbox.")
        subprocess.run(["pkill", "-f", "constructor.py|disruptor.py"], check=False)
        return "NIVEL_1_SANDBOX_STOPPED"

    elif "KILL_ENJAMBRE" in payload:
        log_audit("Nivel 2 Activado: KILL_ENJAMBRE. Terminando todos los microagentes Fénix.")
        subprocess.run(["pkill", "-TERM", "-f", "agente_"], check=False)
        subprocess.run(["pkill", "-TERM", "-f", "nodo_maestro.py"], check=False)
        return "NIVEL_2_ENJAMBRE_KILLED"

    elif "HARD_SHUTDOWN" in payload:
        log_audit("Nivel 3 Activado: HARD_SHUTDOWN. Pánico de hardware e interrupción forzada.")
        subprocess.run(["sync"], check=False)
        subprocess.run(["pkill", "-9", "-f", "fenix"], check=False)
        return "NIVEL_3_HARD_SHUTDOWN_COMMITTED"

    return "UNKNOWN_LEVEL"

def run_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", PORT))
    server.listen(5)
    log_audit(f"Demonio Out-of-Band operativo en puerto {PORT} (RAM < 2MB)")

    while True:
        try:
            readable, _, _ = select.select([server], [], [], 1.0)
            if not readable:
                continue

            conn, addr = server.accept()
            conn.settimeout(2.0)
            data = conn.recv(1024).decode("utf-8", errors="ignore").strip()

            if TOKEN_KILL in data:
                res = execute_kill_level(data)
                conn.sendall(f"ACK_KILL: {res}\n".encode("utf-8"))
            else:
                conn.sendall(b"ERR_INVALID_TOKEN\n")
            conn.close()

        except KeyboardInterrupt:
            log_audit("Kill Switch detenido manualmente.")
            break
        except Exception as e:
            logging.error(f"Error en bucle del switch: {e}")

    server.close()

if __name__ == "__main__":
    run_server()
