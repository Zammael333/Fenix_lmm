"""Microagente Sargento Forense (Autopsia de Memoria Volátil y Análisis de Errores)."""
import sys
import traceback
from enjambre.agentes.agente_base import AgenteBase

class AgenteForense(AgenteBase):
    def __init__(self, agent_id: str = "agente_forense"):
        super().__init__(agent_id, "FORENSIC_ANALYSIS")

    def _run_logic(self, task: dict) -> dict:
        exception_data = task.get("raw_exception")
        trace_str = task.get("traceback_str", "")

        root_cause = "UNKNOWN"
        remediation_strategy = "NONE"

        if "ZeroDivisionError" in trace_str or (isinstance(exception_data, ZeroDivisionError)):
            root_cause = "ZERO_DIVISION_IN_METRIC_NORMALIZATION"
            remediation_strategy = "INJECT_EPSILON_DENOMINATOR_GUARD"
        elif "BufferError" in trace_str:
            root_cause = "EXPORTED_POINTER_RETENTION_IN_MMAP"
            remediation_strategy = "UNPIN_AND_COLLECT_GARBAGE_BEFORE_CLOSE"
        elif "FloatingPointError" in trace_str:
            root_cause = "FLOAT32_PRECISION_OVERFLOW"
            remediation_strategy = "LOG_SUM_EXP_TRANSFORMATION"

        return {
            "root_cause": root_cause,
            "remediation_strategy": remediation_strategy,
            "traceback_summary": trace_str[:300] if trace_str else "No trace provided"
        }
