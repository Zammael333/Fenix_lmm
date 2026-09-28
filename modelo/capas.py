"""
Capas Neuronales del Transformer Fénix (RMSNorm, RoPE, CausalSelfAttention con KV-Cache, SwiGLU).
Implementadas con precisión analítica en NumPy para garantizar cálculo rápido de inferencia
y gradientes exactos en el entrenamiento local.
"""
import numpy as np

def silu(x: np.ndarray) -> np.ndarray:
    """Función de activación SiLU (Swish-1): x / (1 + exp(-x))."""
    # Estable contra desborde para valores negativos grandes
    return x / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

def d_silu(x: np.ndarray) -> np.ndarray:
    """Derivada analítica de SiLU: s + sig(x)*(1 - s)."""
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
    s = x * sig
    return s + sig * (1.0 - s)

class RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.weight = np.ones(dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        x: (B, T, d) o (T, d)
        y = (x / rms(x)) * weight
        """
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        x_norm = x / rms
        out = x_norm * self.weight
        cache = {"x": x, "rms": rms, "x_norm": x_norm}
        return out.astype(np.float32), cache

    def backward(self, d_out: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula gradiente respecto a x y respecto a weight.
        """
        x = cache["x"]
        rms = cache["rms"]
        x_norm = cache["x_norm"]

        d_weight = np.sum(d_out * x_norm, axis=tuple(range(d_out.ndim - 1)))
        d_xnorm = d_out * self.weight

        d_x = (d_xnorm / rms) - (x / (rms ** 3 * self.dim)) * np.sum(d_xnorm * x, axis=-1, keepdims=True)
        return d_x.astype(np.float32), d_weight.astype(np.float32)

class RoPE:
    def __init__(self, head_dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Calcular frecuencias angulares
        half_dim = head_dim // 2
        freqs = 1.0 / (theta ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))
        t = np.arange(max_seq_len, dtype=np.float32)
        angles = np.outer(t, freqs) # (max_seq_len, half_dim)

        self.cos = np.cos(angles).astype(np.float32) # (max_seq_len, half_dim)
        self.sin = np.sin(angles).astype(np.float32)

    def apply(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        """
        Aplica rotación 2D a pares contiguos.
        x: (B, n_heads, seq_len, head_dim)
        """
        B, H, T, D = x.shape
        half = D // 2
        cos = self.cos[offset:offset+T, :][None, None, :, :] # (1, 1, T, half)
        sin = self.sin[offset:offset+T, :][None, None, :, :]

        x1 = x[..., :half]
        x2 = x[..., half:]

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return np.concatenate([out1, out2], axis=-1)

class CausalSelfAttention:
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 256, rope_theta: float = 10000.0):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.rope = RoPE(self.head_dim, max_seq_len, rope_theta)

        # Inicialización de pesos (Xavier / He normalizada)
        scale_init = 1.0 / np.sqrt(d_model)
        self.W_q = (np.random.randn(d_model, d_model) * scale_init).astype(np.float32)
        self.W_k = (np.random.randn(d_model, d_model) * scale_init).astype(np.float32)
        self.W_v = (np.random.randn(d_model, d_model) * scale_init).astype(np.float32)
        self.W_o = (np.random.randn(d_model, d_model) * scale_init).astype(np.float32)

    def forward(self, x: np.ndarray, kv_cache: tuple = None, use_cache: bool = False) -> tuple[np.ndarray, tuple, dict]:
        """
        x: (B, T, d_model)
        """
        B, T, D = x.shape
        H = self.n_heads
        d_h = self.head_dim

        # Proyecciones lineales Q, K, V
        Q = (x @ self.W_q).reshape(B, T, H, d_h).transpose(0, 2, 1, 3) # (B, H, T, d_h)
        K = (x @ self.W_k).reshape(B, T, H, d_h).transpose(0, 2, 1, 3)
        V = (x @ self.W_v).reshape(B, T, H, d_h).transpose(0, 2, 1, 3)

        offset = 0
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            offset = cached_k.shape[2]

        # Aplicar RoPE a Q y K con el offset correspondiente
        Q_rot = self.rope.apply(Q, offset=offset)
        K_rot = self.rope.apply(K, offset=offset)

        if kv_cache is not None:
            K_rot = np.concatenate([cached_k, K_rot], axis=2)
            V = np.concatenate([cached_v, V], axis=2)

        new_kv_cache = (K_rot, V) if use_cache else None
        T_k = K_rot.shape[2]

        # Scores de atención Q @ K^T / sqrt(d_h)
        scores = (Q_rot @ K_rot.transpose(0, 1, 3, 2)) * self.scale # (B, H, T, T_k)

        # Máscara causal si es entrenamiento o secuencia completa
        if T == T_k:
            mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
            scores = scores + mask[None, None, :, :]

        # Softmax numéricamente estable
        shifted_scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(shifted_scores)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Context output: A @ V
        context = attn_weights @ V # (B, H, T, d_h)
        context_trans = context.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = (context_trans @ self.W_o).astype(np.float32)

        cache = {
            "x": x, "Q_rot": Q_rot, "K_rot": K_rot, "V": V,
            "attn_weights": attn_weights, "context_trans": context_trans
        }
        return out, new_kv_cache, cache

    def backward(self, d_out: np.ndarray, cache: dict) -> tuple[np.ndarray, dict]:
        """Retropropagación analítica para W_q, W_k, W_v, W_o y entrada X."""
        x = cache["x"]
        Q_rot = cache["Q_rot"]
        K_rot = cache["K_rot"]
        V = cache["V"]
        attn = cache["attn_weights"]
        context_trans = cache["context_trans"]
        B, T, D = x.shape
        H = self.n_heads
        d_h = self.head_dim

        # Gradiente para W_o
        d_context_trans = d_out @ self.W_o.T
        d_W_o = np.sum(context_trans.transpose(0, 2, 1) @ d_out, axis=0)

        # Reshape d_context
        d_context = d_context_trans.reshape(B, T, H, d_h).transpose(0, 2, 1, 3)

        # Gradientes para V y Attn
        d_V = attn.transpose(0, 1, 3, 2) @ d_context # (B, H, T, d_h)
        d_attn = d_context @ V.transpose(0, 1, 3, 2) # (B, H, T, T)

        # Backward softmax: d_scores = attn * (d_attn - sum(d_attn * attn))
        d_scores = attn * (d_attn - np.sum(d_attn * attn, axis=-1, keepdims=True)) * self.scale

        # Gradientes para Q y K
        d_Q_rot = d_scores @ K_rot
        d_K_rot = d_scores.transpose(0, 1, 3, 2) @ Q_rot

        # Re-proyección a d_model
        d_Q = d_Q_rot.transpose(0, 2, 1, 3).reshape(B, T, D)
        d_K = d_K_rot.transpose(0, 2, 1, 3).reshape(B, T, D)
        d_V_flat = d_V.transpose(0, 2, 1, 3).reshape(B, T, D)

        d_W_q = np.sum(x.transpose(0, 2, 1) @ d_Q, axis=0)
        d_W_k = np.sum(x.transpose(0, 2, 1) @ d_K, axis=0)
        d_W_v = np.sum(x.transpose(0, 2, 1) @ d_V_flat, axis=0)

        d_x = d_Q @ self.W_q.T + d_K @ self.W_k.T + d_V_flat @ self.W_v.T
        grads = {"W_q": d_W_q, "W_k": d_W_k, "W_v": d_W_v, "W_o": d_W_o}
        return d_x.astype(np.float32), grads

class SwiGLU:
    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        scale = 1.0 / np.sqrt(d_model)
        self.W_gate = (np.random.randn(d_model, d_ff) * scale).astype(np.float32)
        self.W_up = (np.random.randn(d_model, d_ff) * scale).astype(np.float32)
        self.W_down = (np.random.randn(d_ff, d_model) * (1.0 / np.sqrt(d_ff))).astype(np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        y = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
        """
        gate = x @ self.W_gate
        up = x @ self.W_up
        silu_gate = silu(gate)
        hidden = silu_gate * up
        out = (hidden @ self.W_down).astype(np.float32)
        cache = {"x": x, "gate": gate, "up": up, "silu_gate": silu_gate, "hidden": hidden}
        return out, cache

    def backward(self, d_out: np.ndarray, cache: dict) -> tuple[np.ndarray, dict]:
        x = cache["x"]
        gate = cache["gate"]
        up = cache["up"]
        silu_gate = cache["silu_gate"]
        hidden = cache["hidden"]

        d_W_down = np.sum(hidden.transpose(0, 2, 1) @ d_out, axis=0)
        d_hidden = d_out @ self.W_down.T

        d_up = d_hidden * silu_gate
        d_silu_gate = d_hidden * up
        d_gate = d_silu_gate * d_silu(gate)

        d_W_up = np.sum(x.transpose(0, 2, 1) @ d_up, axis=0)
        d_W_gate = np.sum(x.transpose(0, 2, 1) @ d_gate, axis=0)

        d_x = d_gate @ self.W_gate.T + d_up @ self.W_up.T
        grads = {"W_gate": d_W_gate, "W_up": d_W_up, "W_down": d_W_down}
        return d_x.astype(np.float32), grads
