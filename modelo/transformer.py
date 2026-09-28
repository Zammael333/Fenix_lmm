"""
Arquitectura FenixTransformer (Decoder-Only).
Compila los bloques de atención causal, RMSNorm, SwiGLU, KV-Cache y la capa LM-Head
con retropropagación analítica completa para pre-entrenamiento y ajuste autorregresivo.
"""
import os
import numpy as np
from modelo.config import FenixModelConfig
from modelo.capas import RMSNorm, CausalSelfAttention, SwiGLU

class TransformerBlock:
    def __init__(self, config: FenixModelConfig):
        self.norm1 = RMSNorm(config.d_model, eps=config.eps)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads, config.max_seq_len, config.rope_theta)
        self.norm2 = RMSNorm(config.d_model, eps=config.eps)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x: np.ndarray, kv_cache: tuple = None, use_cache: bool = False) -> tuple[np.ndarray, tuple, dict]:
        # Pre-LN 1 + Attn + Residual
        norm1_out, norm1_cache = self.norm1.forward(x)
        attn_out, new_kv_cache, attn_cache = self.attn.forward(norm1_out, kv_cache=kv_cache, use_cache=use_cache)
        x_res1 = x + attn_out

        # Pre-LN 2 + SwiGLU + Residual
        norm2_out, norm2_cache = self.norm2.forward(x_res1)
        ffn_out, ffn_cache = self.ffn.forward(norm2_out)
        out = x_res1 + ffn_out

        cache = {
            "x_input": x, "x_res1": x_res1,
            "norm1_cache": norm1_cache, "attn_cache": attn_cache,
            "norm2_cache": norm2_cache, "ffn_cache": ffn_cache
        }
        return out, new_kv_cache, cache

    def backward(self, d_out: np.ndarray, cache: dict) -> tuple[np.ndarray, dict]:
        # 1. Gradiente del bloque FFN
        d_x_res1 = d_out.copy()
        d_ffn_out = d_out
        d_norm2_out, grads_ffn = self.ffn.backward(d_ffn_out, cache["ffn_cache"])
        d_x_from_norm2, d_norm2_w = self.norm2.backward(d_norm2_out, cache["norm2_cache"])
        d_x_res1 += d_x_from_norm2

        # 2. Gradiente del bloque Attn
        d_x = d_x_res1.copy()
        d_attn_out = d_x_res1
        d_norm1_out, grads_attn = self.attn.backward(d_attn_out, cache["attn_cache"])
        d_x_from_norm1, d_norm1_w = self.norm1.backward(d_norm1_out, cache["norm1_cache"])
        d_x += d_x_from_norm1

        block_grads = {
            "norm1_w": d_norm1_w,
            "norm2_w": d_norm2_w,
            **grads_attn,
            **grads_ffn
        }
        return d_x, block_grads

class FenixTransformer:
    def __init__(self, config: FenixModelConfig):
        self.config = config
        scale = 1.0 / np.sqrt(config.d_model)

        # Matriz de embeddings de tokens
        self.wte = (np.random.randn(config.vocab_size, config.d_model) * scale).astype(np.float32)

        # Pila de bloques Transformer
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layers)]

        # RMSNorm final
        self.ln_f = RMSNorm(config.d_model, eps=config.eps)

        # LM Head
        if not config.tie_weights:
            self.lm_head = (np.random.randn(config.d_model, config.vocab_size) * scale).astype(np.float32)
        else:
            self.lm_head = None # Compartido con wte.T

    def get_lm_head(self) -> np.ndarray:
        if self.config.tie_weights:
            return self.wte.T
        return self.lm_head

    def forward(self, input_ids: np.ndarray, kv_caches: list = None, use_cache: bool = False) -> tuple[np.ndarray, list]:
        """
        input_ids: (B, T)
        Retorna: logits (B, T, vocab_size), new_kv_caches
        """
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        B, T = input_ids.shape

        # Lookup de embeddings
        x = self.wte[input_ids] # (B, T, d_model)

        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_layer_cache, _ = block.forward(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_caches.append(new_layer_cache)

        x_norm, _ = self.ln_f.forward(x)
        logits = x_norm @ self.get_lm_head()
        return logits, (new_caches if use_cache else None)

    def forward_and_loss(self, input_ids: np.ndarray, target_ids: np.ndarray) -> tuple[float, np.ndarray, dict]:
        """
        Calcula el pase forward completo y la pérdida Cross-Entropy causal de forma vectorizada.
        input_ids: (B, T)
        target_ids: (B, T)
        """
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
            target_ids = target_ids[None, :]

        B, T = input_ids.shape
        x = self.wte[input_ids]

        block_caches = []
        for block in self.blocks:
            x, _, cache = block.forward(x, use_cache=False)
            block_caches.append(cache)

        x_norm, ln_f_cache = self.ln_f.forward(x)
        head_matrix = self.get_lm_head()
        logits = x_norm @ head_matrix # (B, T, vocab_size)

        # Cross-Entropy numéricamente estable con log-sum-exp
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Negative Log-Likelihood
        b_idx = np.arange(B)[:, None]
        t_idx = np.arange(T)[None, :]
        log_probs = shifted - np.log(np.sum(exp_logits, axis=-1, keepdims=True))
        loss = -np.mean(log_probs[b_idx, t_idx, target_ids])

        cache_total = {
            "input_ids": input_ids, "target_ids": target_ids,
            "probs": probs, "x_norm": x_norm, "ln_f_cache": ln_f_cache,
            "block_caches": block_caches, "head_matrix": head_matrix
        }
        return float(loss), logits, cache_total

    def backward(self, cache: dict) -> dict:
        """
        Retropropagación analítica completa: calcula gradientes para todos los pesos.
        """
        input_ids = cache["input_ids"]
        target_ids = cache["target_ids"]
        probs = cache["probs"].copy() # (B, T, V)
        x_norm = cache["x_norm"]
        ln_f_cache = cache["ln_f_cache"]
        block_caches = cache["block_caches"]
        head_matrix = cache["head_matrix"]
        B, T = input_ids.shape
        N = B * T

        # 1. Gradiente de la pérdida Cross-Entropy: d_logits = (probs - targets) / N
        b_idx = np.arange(B)[:, None]
        t_idx = np.arange(T)[None, :]
        probs[b_idx, t_idx, target_ids] -= 1.0
        d_logits = (probs / N).astype(np.float32)

        # 2. Gradiente del LM-Head
        d_x_norm = d_logits @ head_matrix.T
        if not self.config.tie_weights:
            d_lm_head = np.sum(x_norm.transpose(0, 2, 1) @ d_logits, axis=0)
        else:
            d_lm_head = None

        # 3. Gradiente del RMSNorm final
        d_x, d_ln_f_w = self.ln_f.backward(d_x_norm, ln_f_cache)

        # 4. Gradientes a través de los bloques Transformer (en reversa)
        block_grads = []
        for i in reversed(range(self.config.n_layers)):
            d_x, b_grads = self.blocks[i].backward(d_x, block_caches[i])
            block_grads.append(b_grads)
        block_grads.reverse()

        # 5. Gradiente de la matriz de embedding (wte)
        d_wte = np.zeros_like(self.wte)
        # Acumular gradiente de las posiciones visitadas por input_ids
        for b in range(B):
            for t in range(T):
                token_id = input_ids[b, t]
                d_wte[token_id] += d_x[b, t]

        if self.config.tie_weights:
            # Añadir la contribución de la proyección de salida
            # head_matrix = wte.T -> d_wte += d_logits.T @ x_norm
            d_head_contrib = np.sum(d_logits.transpose(0, 2, 1) @ x_norm, axis=0) # (V, d)
            d_wte += d_head_contrib

        grads = {
            "wte": d_wte,
            "ln_f_w": d_ln_f_w,
            "blocks": block_grads
        }
        if not self.config.tie_weights:
            grads["lm_head"] = d_lm_head

        return grads

    def save_checkpoint(self, filepath: str):
        """Serializa todos los tensores a formato binario comprimido NPZ."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        weights_dict = {"wte": self.wte, "ln_f_w": self.ln_f.weight}
        if not self.config.tie_weights and self.lm_head is not None:
            weights_dict["lm_head"] = self.lm_head

        for i, block in enumerate(self.blocks):
            prefix = f"block_{i}_"
            weights_dict[prefix + "norm1_w"] = block.norm1.weight
            weights_dict[prefix + "norm2_w"] = block.norm2.weight
            weights_dict[prefix + "W_q"] = block.attn.W_q
            weights_dict[prefix + "W_k"] = block.attn.W_k
            weights_dict[prefix + "W_v"] = block.attn.W_v
            weights_dict[prefix + "W_o"] = block.attn.W_o
            weights_dict[prefix + "W_gate"] = block.ffn.W_gate
            weights_dict[prefix + "W_up"] = block.ffn.W_up
            weights_dict[prefix + "W_down"] = block.ffn.W_down

        np.savez_compressed(filepath, **weights_dict)

    def load_checkpoint(self, filepath: str):
        """Carga los tensores desde formato NPZ."""
        data = np.load(filepath)
        self.wte = data["wte"]
        self.ln_f.weight = data["ln_f_w"]
        if "lm_head" in data and not self.config.tie_weights:
            self.lm_head = data["lm_head"]

        for i, block in enumerate(self.blocks):
            prefix = f"block_{i}_"
            block.norm1.weight = data[prefix + "norm1_w"]
            block.norm2.weight = data[prefix + "norm2_w"]
            block.attn.W_q = data[prefix + "W_q"]
            block.attn.W_k = data[prefix + "W_k"]
            block.attn.W_v = data[prefix + "W_v"]
            block.attn.W_o = data[prefix + "W_o"]
            block.ffn.W_gate = data[prefix + "W_gate"]
            block.ffn.W_up = data[prefix + "W_up"]
            block.ffn.W_down = data[prefix + "W_down"]

    def num_parameters(self) -> int:
        total = self.wte.size + self.ln_f.weight.size
        if not self.config.tie_weights and self.lm_head is not None:
            total += self.lm_head.size
        for block in self.blocks:
            total += (
                block.norm1.weight.size + block.norm2.weight.size +
                block.attn.W_q.size + block.attn.W_k.size + block.attn.W_v.size + block.attn.W_o.size +
                block.ffn.W_gate.size + block.ffn.W_up.size + block.ffn.W_down.size
            )
        return total
