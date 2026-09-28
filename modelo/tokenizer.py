"""
Tokenizador Autónomo de Fénix (FenixTokenizer).
Implementa codificación a nivel de subpalabras con Byte-Fallback UTF-8 integral,
asegurando que cualquier texto o código se tokenice sin tokens desconocidos (<unk>).
"""
import json
import os

PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"

class FenixTokenizer:
    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2

        # 1. Inicializar vocabulario base
        self.id_to_token = {
            self.pad_id: PAD_TOKEN,
            self.bos_id: BOS_TOKEN,
            self.eos_id: EOS_TOKEN,
        }
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}

        # 2. Agregar los 256 bytes elementales UTF-8 (IDs 3 a 258)
        self.byte_offset = 3
        for b in range(256):
            token_id = self.byte_offset + b
            byte_val = bytes([b])
            self.id_to_token[token_id] = byte_val
            self.token_to_id[byte_val] = token_id

        # 3. Vocabulario de subpalabras frecuentes semilla
        seed_words = [
            " ", "de", "la", "el", "en", " y", " que", " a", " los", " del",
            " se", " las", " por", " un", " para", " con", " no", " una", " su", " al",
            " lo", " como", " más", " pero", " sus", " le", " ya", " o", " fue", " este",
            " ha", " sí", " porque", " esta", " son", " entre", " está", " cuando", " muy",
            "fenix", "llm", "cluster", "kernel", "nodo", "soberano", "agente", "sre",
            "osint", "forense", "marketing", "memoria", "kalman", "mmap", "ssd", "ram",
            "seis", "sigmas", "error", "status", "bypass", "kill", "switch", "coliseo",
            "modelo", "red", "token", "prompt", "seguridad", "cifrado", "mtls", "tls",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "\n", "def ", "class ", "return ", "import ", "if ", "else:", "for ", "in ",
            "(", ")", "[", "]", "{", "}", ":", ",", ".", "=", "+", "-", "*", "/", "_", '"', "'"
        ]
        curr_id = 259
        for word in seed_words:
            if curr_id >= self.vocab_size:
                break
            b_word = word.encode("utf-8")
            if b_word not in self.token_to_id:
                self.id_to_token[curr_id] = b_word
                self.token_to_id[b_word] = curr_id
                curr_id += 1

    def train_bpe(self, text: str, max_vocab_size: int = None):
        """Aprende nuevos pares frecuentes de tokens sobre el corpus para enriquecer el vocabulario."""
        max_size = max_vocab_size or self.vocab_size
        # Inicializar con los tokens elementales de byte correspondientes
        raw_tokens = [self.byte_offset + b for b in text.encode("utf-8")]

        curr_id = len(self.id_to_token)
        while curr_id < max_size:
            # Contar pares contiguos
            pairs = {}
            for i in range(len(raw_tokens) - 1):
                p = (raw_tokens[i], raw_tokens[i+1])
                pairs[p] = pairs.get(p, 0) + 1

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                break

            t1 = self.id_to_token[best_pair[0]]
            t2 = self.id_to_token[best_pair[1]]
            new_token = (t1 if isinstance(t1, bytes) else t1.encode()) + (t2 if isinstance(t2, bytes) else t2.encode())

            if new_token not in self.token_to_id:
                self.id_to_token[curr_id] = new_token
                self.token_to_id[new_token] = curr_id
                merged_id = curr_id
                curr_id += 1
            else:
                merged_id = self.token_to_id[new_token]

            # Reemplazar en la secuencia
            new_tokens = []
            i = 0
            while i < len(raw_tokens):
                if i < len(raw_tokens) - 1 and raw_tokens[i] == best_pair[0] and raw_tokens[i+1] == best_pair[1]:
                    new_tokens.append(merged_id)
                    i += 2
                else:
                    new_tokens.append(raw_tokens[i])
                    i += 1
            raw_tokens = new_tokens

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Codifica un string en una lista de IDs de tokens de forma determinista."""
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)

        raw_bytes = text.encode("utf-8")
        i = 0
        n = len(raw_bytes)

        # Greedy longest-match tokenization sobre el vocabulario
        while i < n:
            matched = False
            for length in range(min(32, n - i), 1, -1):
                chunk = raw_bytes[i:i+length]
                if chunk in self.token_to_id:
                    tokens.append(self.token_to_id[chunk])
                    i += length
                    matched = True
                    break

            if not matched:
                b = raw_bytes[i]
                tokens.append(self.byte_offset + b)
                i += 1

        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decodifica una lista de IDs de tokens de vuelta a string UTF-8."""
        byte_chunks = []
        for tid in token_ids:
            if tid in [self.pad_id, self.bos_id, self.eos_id]:
                continue
            token_val = self.id_to_token.get(tid, b"")
            if isinstance(token_val, bytes):
                byte_chunks.append(token_val)
            elif isinstance(token_val, str):
                byte_chunks.append(token_val.encode("utf-8"))

        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, filepath: str):
        """Serializa el vocabulario a JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "vocab": {
                str(k): (v.decode("latin1") if isinstance(v, bytes) else v)
                for k, v in self.id_to_token.items()
            }
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """Carga el vocabulario desde JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.id_to_token = {}
        self.token_to_id = {}
        for k, v in data["vocab"].items():
            tid = int(k)
            if tid in [0, 1, 2]:
                self.id_to_token[tid] = v
                self.token_to_id[v] = tid
            else:
                b_val = v.encode("latin1")
                self.id_to_token[tid] = b_val
                self.token_to_id[b_val] = tid
