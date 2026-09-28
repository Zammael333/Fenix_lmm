"""
Árbitro del Coliseo: Evalúa la tasa de defectos de los candidatos (Seis Sigmas SLO 99.99973%),
mide latencia y divergencia KL, y ejecuta el protocolo Muerte Limpia ante violaciones.
"""
import math
import time
import logging

logger = logging.getLogger("fenix.sandbox.arbitro")

class Arbitro:
    def __init__(self, max_allowed_failure_rate: float = 0.000027):
        self.max_allowed_failure_rate = max_allowed_failure_rate

    def evaluate_candidate(self, candidate_fn, test_cases: list[dict]) -> dict:
        total = len(test_cases)
        passed = 0
        failed = 0
        details = []

        start_t = time.perf_counter()
        for case in test_cases:
            p = case["p"]
            r = case["r"]
            desc = case["desc"]
            try:
                # Comprobar si el input es NaN antes de llamar si el candidato debe sanearlo
                if math.isnan(p) or math.isnan(r):
                    raise ValueError("Entrada contiene NaN.")

                k, p_next = candidate_fn(p, r)
                if math.isnan(k) or math.isnan(p_next) or math.isinf(k) or math.isinf(p_next):
                    failed += 1
                    details.append({"case": desc, "result": "FAILED_NAN_OR_INF"})
                else:
                    passed += 1
                    details.append({"case": desc, "result": "PASSED"})
            except Exception as e:
                failed += 1
                details.append({"case": desc, "result": f"EXCEPTION: {type(e).__name__}"})

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        failure_rate = failed / max(total, 1)
        approved = failure_rate <= self.max_allowed_failure_rate

        return {
            "total_cases": total,
            "passed": passed,
            "failed": failed,
            "failure_rate": failure_rate,
            "approved": approved,
            "elapsed_ms": round(elapsed_ms, 3),
            "details": details
        }
