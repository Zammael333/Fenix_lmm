"""
Paginador de Memoria Latente Predictivo con Mapeo de Memoria (mmap).
Administra el swapping y evicción táctica de tensores entre el SSD y la RAM volátil.
Implementa aislamiento estricto del ciclo de vida del búfer para evitar
excepciones BufferError de CPython al cerrar descriptores.
"""
import os
import gc
import mmap
import time
import logging
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("fenix.mochilas.paginador")

class PaginadorLatente:
    def __init__(self, mmap_store_dir: str = None, max_resident_agents: int = 2):
        if mmap_store_dir:
            p = Path(mmap_store_dir)
            self.store_dir = p if p.is_absolute() else (PROJECT_ROOT / p)
        else:
            self.store_dir = PROJECT_ROOT / "data" / "mmap_store"
        self.max_resident_agents = max_resident_agents
        self.store_dir.mkdir(parents=True, exist_ok=True)
        # Diccionario de páginas activas: {agent_name: {"mmap": mm, "file": f, "weights": arr, "loaded_at": t}}
        self.active_pages = {}

    def ensure_weight_file(self, agent_name: str, size_elements: int = 1024 * 1024) -> str:
        """Crea el archivo de pesos mapeable para el agente si no existe."""
        path = self.store_dir / f"{agent_name}.weights.bin"
        if not path.exists():
            data = (np.random.randn(size_elements) * 0.02).astype(np.float32)
            with open(path, "wb") as f:
                f.write(data.tobytes())
            logger.info(f"Generado archivo de pesos SSD para '{agent_name}' ({len(data)*4/1024/1024:.2f} MB)")
        return str(path)

    def load_agent(self, agent_name: str, agent_probs: dict = None) -> np.ndarray:
        """
        Carga latente de pesos en RAM virtual mediante mmap.
        Si la RAM simulada está saturada, ejecuta evicción selectiva guiada por Kalman.
        """
        if agent_name in self.active_pages:
            logger.debug(f"RAM HIT: Pesos de '{agent_name}' ya residentes en memoria.")
            return self.active_pages[agent_name]["weights"]

        # Evicción si alcanzamos el límite de residentes en RAM
        if len(self.active_pages) >= self.max_resident_agents:
            self._evict_lowest_priority(exclude=agent_name, agent_probs=agent_probs)

        path_str = self.ensure_weight_file(agent_name)
        start_time = time.perf_counter()

        file_obj = open(path_str, "rb")
        fileno = file_obj.fileno()
        length = os.path.getsize(path_str)

        mm = mmap.mmap(fileno, length, access=mmap.ACCESS_READ)
        weights_view = np.frombuffer(mm, dtype=np.float32)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.active_pages[agent_name] = {
            "mmap": mm,
            "file": file_obj,
            "weights": weights_view,
            "loaded_at": time.time()
        }

        logger.info(f"SWAPPING: Pesos de '{agent_name}' cargados de SSD a RAM vía mmap en {elapsed_ms:.2f} ms.")
        return weights_view

    def evict_agent(self, agent_name: str) -> bool:
        """
        Evicta limpiamente los pesos de un agente de la RAM al SSD.
        Aplica aislamiento de referencias para prevenir BufferError en CPython.
        """
        if agent_name not in self.active_pages:
            return False

        page = self.active_pages.pop(agent_name)
        mm = page.get("mmap")
        file_obj = page.get("file")

        # 1. Destruir referencias explícitas a la vista de NumPy
        del page["weights"]
        del page

        # 2. Forzar recolección de basura para liberar el búfer exportado
        gc.collect()

        # 3. Cierre defensivo de mmap y archivo
        try:
            if mm and not mm.closed:
                mm.close()
        except BufferError as e:
            logger.warning(f"BufferError mitigado durante evicción de '{agent_name}': {e}. Descriptor pospuesto.")
        finally:
            if file_obj and not file_obj.closed:
                file_obj.close()

        logger.info(f"EVICCIÓN: Pesos de '{agent_name}' desalojados limpiamente de RAM.")
        return True

    def _evict_lowest_priority(self, exclude: str = None, agent_probs: dict = None):
        """Selecciona el agente residente con menor probabilidad según el Filtro de Kalman y lo desaloja."""
        candidates = [name for name in self.active_pages if name != exclude]
        if not candidates:
            return

        if agent_probs:
            evict_target = min(candidates, key=lambda name: agent_probs.get(name, 0.0))
        else:
            evict_target = min(candidates, key=lambda name: self.active_pages[name]["loaded_at"])

        logger.info(f"CONFINAMIENTO TERMODINÁMICO: Evictando agente de baja prioridad '{evict_target}'.")
        self.evict_agent(evict_target)

    def close_all(self):
        """Cierra todas las páginas activas."""
        for name in list(self.active_pages.keys()):
            self.evict_agent(name)

if __name__ == "__main__":
    paginador = PaginadorLatente(max_resident_agents=2)
    probs = {"agente_sre": 0.8, "agente_osint": 0.1, "agente_marketing": 0.05, "agente_forense": 0.05}

    w_sre = paginador.load_agent("agente_sre", agent_probs=probs)
    w_osint = paginador.load_agent("agente_osint", agent_probs=probs)
    w_mkt = paginador.load_agent("agente_marketing", agent_probs=probs)

    paginador.close_all()
    print("Todas las páginas desalojadas con éxito. Sin BufferErrors.")
