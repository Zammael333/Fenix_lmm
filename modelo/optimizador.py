"""
Optimizador AdamW Vectorizado con Recorte de Gradiente (Gradient Clipping)
y Programador de Tasa de Aprendizaje Cosine Annealing para Fénix LLM.
"""
import numpy as np

class CosineScheduler:
    def __init__(self, base_lr: float, min_lr: float, warmup_steps: int, max_steps: int):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(max(1, self.warmup_steps))
        if step > self.max_steps:
            return self.min_lr
        progress = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
        coeff = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.min_lr + coeff * (self.base_lr - self.min_lr)

class AdamW:
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        self.m = {}
        self.v = {}
        self.t = 0

    def _clip_gradients(self, grads_dict: dict) -> float:
        """Calcula la norma L2 global de todos los gradientes y recorta si supera max_grad_norm."""
        total_norm_sq = 0.0

        def accumulate_norm(g):
            nonlocal total_norm_sq
            if isinstance(g, np.ndarray):
                total_norm_sq += float(np.sum(g ** 2))
            elif isinstance(g, list):
                for item in g:
                    accumulate_norm(item)
            elif isinstance(g, dict):
                for v in g.values():
                    accumulate_norm(v)

        accumulate_norm(grads_dict)
        global_norm = np.sqrt(total_norm_sq)

        if global_norm > self.max_grad_norm and global_norm > 0:
            scale = self.max_grad_norm / (global_norm + 1e-6)

            def apply_scale(g):
                if isinstance(g, np.ndarray):
                    g *= scale
                elif isinstance(g, list):
                    for item in g:
                        apply_scale(item)
                elif isinstance(g, dict):
                    for v in g.values():
                        apply_scale(v)

            apply_scale(grads_dict)

        return float(global_norm)

    def step(self, model, grads: dict, lr: float = None):
        """Aplica la actualización AdamW sobre todos los parámetros del FenixTransformer."""
        current_lr = lr if lr is not None else self.lr
        self.t += 1

        # 1. Recorte de gradientes global
        grad_norm = self._clip_gradients(grads)

        # 2. Función auxiliar de actualización para un tensor individual
        def update_param(key: str, p: np.ndarray, g: np.ndarray):
            if g is None:
                return
            if key not in self.m:
                self.m[key] = np.zeros_like(p)
                self.v[key] = np.zeros_like(p)

            m = self.m[key]
            v = self.v[key]

            # Actualizar momentos
            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * (g ** 2)
            self.m[key] = m
            self.v[key] = v

            # Corrección de sesgo
            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)

            # Actualización con decaimiento de peso desacoplado (Weight Decay)
            p_update = m_hat / (np.sqrt(v_hat) + self.eps)
            p -= current_lr * (p_update + self.weight_decay * p)

        # 3. Actualizar Embeddings y RMSNorm final
        update_param("wte", model.wte, grads["wte"])
        update_param("ln_f_w", model.ln_f.weight, grads["ln_f_w"])
        if not model.config.tie_weights and model.lm_head is not None:
            update_param("lm_head", model.lm_head, grads["lm_head"])

        # 4. Actualizar Bloques Transformer
        for i, block in enumerate(model.blocks):
            b_grads = grads["blocks"][i]
            prefix = f"block_{i}_"
            update_param(prefix + "norm1_w", block.norm1.weight, b_grads["norm1_w"])
            update_param(prefix + "norm2_w", block.norm2.weight, b_grads["norm2_w"])
            update_param(prefix + "W_q", block.attn.W_q, b_grads["W_q"])
            update_param(prefix + "W_k", block.attn.W_k, b_grads["W_k"])
            update_param(prefix + "W_v", block.attn.W_v, b_grads["W_v"])
            update_param(prefix + "W_o", block.attn.W_o, b_grads["W_o"])
            update_param(prefix + "W_gate", block.ffn.W_gate, b_grads["W_gate"])
            update_param(prefix + "W_up", block.ffn.W_up, b_grads["W_up"])
            update_param(prefix + "W_down", block.ffn.W_down, b_grads["W_down"])

        return grad_norm
