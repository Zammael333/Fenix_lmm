"""
Generador Autorregresivo de Texto con KV-Cache, Muestreo Top-P / Top-K
y Penalización por Repetición para Fénix LLM.
"""
import numpy as np
from modelo.tokenizer import FenixTokenizer
from modelo.transformer import FenixTransformer

class FenixGenerator:
    def __init__(self, model: FenixTransformer, tokenizer: FenixTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def sample_next_token(
        self,
        logits: np.ndarray,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        generated_tokens: list[int] = None
    ) -> int:
        """Aplica temperatura, penalización de repetición y muestreo Top-K / Top-P."""
        # Clonar logits para evitar mutaciones no deseadas
        logits = logits.copy().astype(np.float64)

        # 1. Penalización de repetición
        if repetition_penalty != 1.0 and generated_tokens:
            for token_id in set(generated_tokens):
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

        # Modo Greedy Determinista (temperatura cero)
        if temperature <= 1e-4:
            return int(np.argmax(logits))

        # 2. Escalamiento por temperatura
        logits /= temperature

        # 3. Filtrado Top-K
        if top_k > 0 and top_k < len(logits):
            indices_to_remove = np.argsort(logits)[:-top_k]
            logits[indices_to_remove] = -1e9

        # 4. Softmax
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits)

        # 5. Filtrado Top-P (Nucleus Sampling)
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)

            # Cortar índices que exceden top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # Desplazar uno para asegurar mantener al menos el primer token
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0.0
            sum_p = np.sum(probs)
            if sum_p > 0:
                probs /= sum_p
            else:
                return int(np.argmax(logits))

        # Muestreo estocástico categórico
        return int(np.random.choice(len(probs), p=probs))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_on_eos: bool = True
    ) -> str:
        """
        Genera texto autorregresivamente utilizando KV-Cache para complejidad O(1) por token.
        """
        prompt_tokens = self.tokenizer.encode(prompt, add_bos=True)
        if not prompt_tokens:
            prompt_tokens = [self.tokenizer.bos_id]

        input_ids = np.array(prompt_tokens, dtype=np.int32)[None, :] # (1, T)

        # 1. Fase de Prefill: procesar el prompt y crear el KV-Cache inicial
        logits, kv_caches = self.model.forward(input_ids, use_cache=True)
        last_logits = logits[0, -1, :]

        generated = list(prompt_tokens)
        next_token = self.sample_next_token(
            last_logits, temperature, top_k, top_p, repetition_penalty, generated
        )
        generated.append(next_token)

        # 2. Fase de Decodificación Paso a Paso (O(1) por token con KV-Cache)
        for _ in range(max_new_tokens - 1):
            if stop_on_eos and next_token == self.tokenizer.eos_id:
                break

            curr_input = np.array([[next_token]], dtype=np.int32)
            logits, kv_caches = self.model.forward(curr_input, kv_caches=kv_caches, use_cache=True)
            last_logits = logits[0, -1, :]

            next_token = self.sample_next_token(
                last_logits, temperature, top_k, top_p, repetition_penalty, generated
            )
            generated.append(next_token)

        return self.tokenizer.decode(generated)
