"""
Módulo de Envoltura Criptográfica y Validación de Prompt Fencing de Fénix LLM.
Previene la inyección de prompts y manipulación de directivas mediante firma HMAC-SHA256.
"""
import re
import hmac
import hashlib
import logging

logger = logging.getLogger("fenix.seguridad.prompt_fence")

class PromptFenceManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode("utf-8")
        self.fence_pattern = re.compile(
            r'^<sec:fence\s+rating="([^"]+)"\s+source="([^"]+)"\s+type="([^"]+)"\s+signature="([a-f0-9]{64})">(.*?)</sec:fence>$',
            re.DOTALL
        )

    def _compute_signature(self, rating: str, source: str, fence_type: str, content: str) -> str:
        """Calcula la firma determinista HMAC-SHA256 del contenido canónico."""
        canonical_str = f"rating={rating}|source={source}|type={fence_type}|content={content}"
        return hmac.new(self.secret_key, canonical_str.encode("utf-8"), hashlib.sha256).hexdigest()

    def create_fenced_prompt(self, content: str, rating: str = "trusted", source: str = "system", fence_type: str = "instructions") -> str:
        """Envuelve una instrucción en una frontera XML firmada criptográficamente."""
        clean_content = content.strip()
        sig = self._compute_signature(rating, source, fence_type, clean_content)
        return f'<sec:fence rating="{rating}" source="{source}" type="{fence_type}" signature="{sig}">{clean_content}</sec:fence>'

    def verify_and_unwrap(self, raw_input: str) -> tuple[bool, str, str]:
        """
        Verifica la validez del Prompt Fencing.
        Retorna (es_valido, contenido_extraido, codigo_estado).
        Estados:
          - "VALID_FENCE": Fence legítimo con firma verificada.
          - "UNFENCED_RAW": Entrada sin envoltura de seguridad (tratada como datos no confiables).
          - "ERR_STRUCTURE": Desbalance de etiquetas o inyección de múltiples fences.
          - "ERR_SIGNATURE_MISMATCH": Manipulación del payload o firma adulterada.
        """
        stripped = raw_input.strip()

        # Comprobar si hay etiquetas de fencing
        has_open = "<sec:fence" in stripped
        has_close = "</sec:fence>" in stripped

        if not has_open and not has_close:
            return True, stripped, "UNFENCED_RAW"

        # Detección de inyección estructural (desbalance o duplicados)
        if stripped.count("<sec:fence") != 1 or stripped.count("</sec:fence>") != 1:
            logger.critical("VIOLACIÓN DE PERÍMETRO: Intento de escape por desbalance o multiplicidad de etiquetas fence.")
            return False, "", "ERR_STRUCTURE"

        match = self.fence_pattern.match(stripped)
        if not match:
            logger.warning("Fallo en parseo estructural del fence XML.")
            return False, "", "ERR_MALFORMED_FENCE"

        rating, source, fence_type, sig_received, content = match.groups()
        sig_expected = self._compute_signature(rating, source, fence_type, content)

        if not hmac.compare_digest(sig_received, sig_expected):
            logger.critical(f"FIRMA INVÁLIDA DETECTADA: Esperado={sig_expected[:16]}... Recibido={sig_received[:16]}...")
            return False, "", "ERR_SIGNATURE_MISMATCH"

        return True, content, "VALID_FENCE"

if __name__ == "__main__":
    pf = PromptFenceManager("test_secret_123")
    fenced = pf.create_fenced_prompt("/status", rating="trusted", source="system", fence_type="instructions")
    print("Fenced:", fenced)
    val, content, code = pf.verify_and_unwrap(fenced)
    print("Verificación:", val, content, code)

    # Intento de sabotaje
    sabotaged = fenced.replace("/status", "/status; rm -rf /")
    val, content, code = pf.verify_and_unwrap(sabotaged)
    print("Sabotaje detectado:", val, code)
