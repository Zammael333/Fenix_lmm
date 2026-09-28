"""
Puente CTypes para el Acelerador C99 / AVX2 de Fénix LLM.
Acelera operaciones críticas de Capa Cero, RMSNorm y Softmax.
Si la biblioteca compartida no está compilada, ofrece fallback transparente a NumPy.
"""
import os
import ctypes
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SO_PATH = PROJECT_ROOT / "lib" / "libfenix_accelerator.so"

class FenixAceleradorC:
    def __init__(self):
        self.available = False
        self._lib = None

        if SO_PATH.exists():
            try:
                self._lib = ctypes.CDLL(str(SO_PATH))

                # fenix_capa_cero_match(const char*, char*, int) -> int
                self._lib.fenix_capa_cero_match.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_int
                ]
                self._lib.fenix_capa_cero_match.restype = ctypes.c_int

                # fenix_avx2_rmsnorm(const float*, const float*, float*, int, float)
                self._lib.fenix_avx2_rmsnorm.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_float
                ]
                self._lib.fenix_avx2_rmsnorm.restype = None

                # fenix_avx2_softmax(const float*, float*, int, float)
                self._lib.fenix_avx2_softmax.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_float
                ]
                self._lib.fenix_avx2_softmax.restype = None

                self.available = True
            except Exception:
                self.available = False

    def match_capa_cero(self, command: str) -> tuple[bool, str]:
        """Evalúa un comando en el acelerador nativo C99."""
        if not self.available:
            return False, ""
        buf = ctypes.create_string_buffer(256)
        matched = self._lib.fenix_capa_cero_match(command.encode("utf-8"), buf, 256)
        if matched:
            return True, buf.value.decode("utf-8")
        return False, ""

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """RMSNorm acelerado con AVX2."""
        if not self.available or x.ndim != 1:
            # Fallback NumPy
            rms = np.sqrt(np.mean(x ** 2) + eps)
            return (x / rms) * weight

        x_c = x.astype(np.float32)
        w_c = weight.astype(np.float32)
        out_c = np.zeros_like(x_c)

        x_ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        w_ptr = w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._lib.fenix_avx2_rmsnorm(x_ptr, w_ptr, out_ptr, len(x_c), eps)
        return out_c

    def softmax(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax acelerado con AVX2."""
        if not self.available or logits.ndim != 1:
            scaled = logits / max(temperature, 1e-4)
            shifted = scaled - np.max(scaled)
            exp_l = np.exp(shifted)
            return exp_l / np.sum(exp_l)

        l_c = logits.astype(np.float32)
        out_c = np.zeros_like(l_c)
        l_ptr = l_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._lib.fenix_avx2_softmax(l_ptr, out_ptr, len(l_c), temperature)
        return out_c

if __name__ == "__main__":
    acc = FenixAceleradorC()
    print("Acelerador C99 / AVX2 disponible:", acc.available)
    if acc.available:
        matched, resp = acc.match_capa_cero("/status")
        print("Capa Cero C99 Match:", matched, resp)
        x = np.ones(64, dtype=np.float32)
        w = np.ones(64, dtype=np.float32)
        print("RMSNorm C99 resultado[0]:", acc.rmsnorm(x, w)[0])
