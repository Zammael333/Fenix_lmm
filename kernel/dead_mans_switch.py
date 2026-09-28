"""
Dead Man's Switch (Interruptor de Hombre Muerto).
Monitorea la existencia y frescura de un token criptográfico volátil en /dev/shm.
Si el latido cesa (sesión desconectada o coaccionada), ejecuta la purga de memoria volátil.
"""
import os
import sys
import time
import glob
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("fenix.kernel.dead_mans_switch")

class DeadMansSwitch:
    def __init__(self, volatile_dir: str = None, timeout_seconds: float = None):
        default_dir = "/dev/shm/fenix_volatile" if os.path.exists("/dev/shm") else str(PROJECT_ROOT / "data" / "volatile")
        self.volatile_dir = Path(os.getenv("FENIX_VOLATILE_SHM_DIR", volatile_dir or default_dir))
        self.timeout_seconds = float(os.getenv("FENIX_DEAD_MAN_TIMEOUT", timeout_seconds or 10.0))
        self.heartbeat_file = self.volatile_dir / "heartbeat.token"
        self.volatile_dir.mkdir(parents=True, exist_ok=True)

    def ping(self):
        """Actualiza el latido escribiendo la marca de tiempo actual."""
        with open(self.heartbeat_file, "w", encoding="utf-8") as f:
            f.write(str(time.time()))

    def check_and_purge_if_expired(self) -> bool:
        """
        Comprueba el latido. Si caducó, destruye los archivos de la memoria volátil
        escribiendo ceros y retorna True (purga ejecutada).
        """
        if not self.heartbeat_file.exists():
            return False

        try:
            with open(self.heartbeat_file, "r", encoding="utf-8") as f:
                val = float(f.read().strip())
        except Exception:
            val = 0.0

        elapsed = time.time() - val
        if elapsed > self.timeout_seconds:
            logger.critical(f"DEAD MAN'S SWITCH DISPARADO: Latido expirado ({elapsed:.1f}s > {self.timeout_seconds}s). Purgando RAM volátil.")
            self._shred_volatile_storage()
            return True

        return False

    def _shred_volatile_storage(self):
        """Sobrescribe con ceros y elimina los archivos en memoria volátil."""
        for path in self.volatile_dir.glob("*"):
            try:
                size = path.stat().st_size
                with open(path, "wb") as f:
                    f.write(b"\x00" * size)
                path.unlink()
                logger.info(f"Archivo purgado a ceros: {path}")
            except Exception as e:
                logger.error(f"Error purgando {path}: {e}")

if __name__ == "__main__":
    dms = DeadMansSwitch(timeout_seconds=2.0)
    dms.ping()
    print("Latido emitido. Comprobando estado:", dms.check_and_purge_if_expired())
    print("Esperando 2.5s para probar expiración...")
    time.sleep(2.5)
    print("Comprobando tras expiración:", dms.check_and_purge_if_expired())
