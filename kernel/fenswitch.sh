#!/usr/bin/env bash
# Demonio de Kill Switch en Bash con prioridad máxima (nice -20)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PORT_PRIVATE="${FENIX_KILL_PORT:-9999}"
TOKEN_KILL="${FENIX_KILL_TOKEN:-RED_ALERT_KILL_33}"
LOG_DIR="${FENIX_AUDIT_DIR:-${PROJECT_ROOT}/audit}"
LOG_FILE="${LOG_DIR}/kill_switch.log"

mkdir -p "$LOG_DIR"
echo "$(date -Iseconds) - [IMMUNE] - Demonio Bash nice -20 activo en puerto $PORT_PRIVATE" >> "$LOG_FILE"

# Delegar al ejecutable python nativo con ruta relativa resuelta
exec python3 "${SCRIPT_DIR}/fenswitch.py"
