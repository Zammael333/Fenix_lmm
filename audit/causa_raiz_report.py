"""Generador de Reportes de Causa Raíz (RCA) para el Estándar Seis Sigmas."""
import json
import logging

class CausaRaizReport:
    @staticmethod
    def generate_markdown_rca(autopsy_data: dict) -> str:
        agent = autopsy_data.get("agent_id", "Desconocido")
        ts = autopsy_data.get("timestamp", "")
        exc = autopsy_data.get("exception_type", "")
        msg = autopsy_data.get("exception_message", "")
        rem = autopsy_data.get("remediation", {})

        md = f"""# REPORTE FORENSE DE CAUSA RAÍZ (RCA) - FÉNIX AUTOINMUNE
**Fecha**: {ts}  
**Agente Involucrado**: `{agent}`  
**Estado SLO**: INCIDENTE CONTENIDO (< 3.4 DPMO)

## 1. Detección de Falla
- **Tipo de Excepción**: `{exc}`
- **Mensaje de Error**: `{msg}`
- **Vector de Falla**: `{rem.get('vector', 'N/A')}`

## 2. Prescripción de Inmunidad
- **Parche Aplicado**: `{rem.get('fix_applied', 'N/A')}`
- **Patrón de Código**:
```python
{rem.get('code_patch', '')}
```
- **Veredicto del Auditor**: `{rem.get('status', 'PENDING')}`
"""
        return md
