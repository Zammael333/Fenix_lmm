"""
Microagente Sargento Marketing & Neuroeconomía (Optimización de Prompts y Generación Neuronal).
Integra el modelo FenixTransformer para generar completions y optimizaciones autorregresivas.
"""
import os
import re
from enjambre.agentes.agente_base import AgenteBase
from modelo.config import FenixModelConfig
from modelo.tokenizer import FenixTokenizer
from modelo.transformer import FenixTransformer
from modelo.generador import FenixGenerator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class AgenteMarketing(AgenteBase):
    def __init__(self, agent_id: str = "agente_marketing"):
        super().__init__(agent_id, "NEUROECONOMIC_OPTIMIZER")
        self.model = None
        self.tokenizer = None
        self.generator = None
        self._load_neural_model_if_available()

    def _load_neural_model_if_available(self):
        ckpt_path = os.path.join(PROJECT_ROOT, "data", "checkpoints", "nano", "fenix_model.npz")
        tok_path = os.path.join(PROJECT_ROOT, "data", "checkpoints", "nano", "fenix_tokenizer.json")
        if os.path.exists(ckpt_path) and os.path.exists(tok_path):
            try:
                self.tokenizer = FenixTokenizer(vocab_size=512)
                self.tokenizer.load(tok_path)
                cfg = FenixModelConfig(vocab_size=512, d_model=64, n_heads=4, n_layers=2, d_ff=172, max_seq_len=128)
                self.model = FenixTransformer(cfg)
                self.model.load_checkpoint(ckpt_path)
                self.generator = FenixGenerator(self.model, self.tokenizer)
            except Exception:
                self.generator = None

    def _run_logic(self, task: dict) -> dict:
        prompt_text = task.get("prompt", "")
        # Eliminación de relleno decorativo innecesario
        word_count = len(prompt_text.split())
        condensed = re.sub(r'\b(por favor|amablemente|podrías|quizás)\b', '', prompt_text, flags=re.IGNORECASE)
        condensed = re.sub(r'\s+', ' ', condensed).strip()
        token_savings_pct = max(0.0, ((word_count - len(condensed.split())) / max(word_count, 1)) * 100)

        result = {
            "original_prompt": prompt_text,
            "optimized_prompt": condensed,
            "token_savings_pct": round(token_savings_pct, 2),
            "tone_profile": "SOVEREIGN_CONCISE"
        }

        # Si se solicita completado generativo neuronal y el modelo está listo
        if task.get("generate_completion", False) and self.generator is not None:
            max_new = task.get("max_new_tokens", 25)
            temp = task.get("temperature", 0.6)
            generated_text = self.generator.generate(condensed, max_new_tokens=max_new, temperature=temp)
            result["neural_generation"] = generated_text
            result["engine"] = "FenixTransformer-Nano (KV-Cache)"

        return result
