"""
Cargador de Datos y Generador de Lotes Causal para Entrenamiento de LLM.
Realiza segmentación continua de texto, tokenización y desplazamiento causal (x -> y).
"""
import numpy as np
from modelo.tokenizer import FenixTokenizer

class CausalLanguageModelingDataset:
    def __init__(self, text: str, tokenizer: FenixTokenizer, seq_len: int = 64):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        # Tokenizar todo el corpus
        self.tokens = np.array(tokenizer.encode(text, add_bos=True, add_eos=True), dtype=np.int32)
        # Número de muestras completas
        self.n_samples = max(0, len(self.tokens) - seq_len)

    def get_batch(self, batch_size: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """Extrae un lote aleatorio de secuencias (X, Y) con desplazamiento causal."""
        if self.n_samples <= 0:
            raise ValueError("El corpus es demasiado corto para la longitud de secuencia configurada.")

        indices = np.random.randint(0, self.n_samples, size=batch_size)
        x_batch = np.zeros((batch_size, self.seq_len), dtype=np.int32)
        y_batch = np.zeros((batch_size, self.seq_len), dtype=np.int32)

        for i, idx in enumerate(indices):
            chunk = self.tokens[idx : idx + self.seq_len + 1]
            x_batch[i] = chunk[:-1]
            y_batch[i] = chunk[1:]

        return x_batch, y_batch

    def __len__(self) -> int:
        return self.n_samples
