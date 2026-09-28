"""
Configuración de hiperparámetros de la arquitectura FenixTransformer.
Permite instanciar planos escalables (Nano, Micro, Soberano).
"""
from dataclasses import dataclass

@dataclass
class FenixModelConfig:
    vocab_size: int = 512
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 172         # ~ 8/3 * d_model para SwiGLU
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    eps: float = 1e-6
    dropout: float = 0.0
    tie_weights: bool = True

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model debe ser divisible por n_heads"
        return self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, data: dict):
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)
