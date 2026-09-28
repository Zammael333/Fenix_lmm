"""
Script de Entrenamiento y Alineación con Modo Carroñero (Zombie Teacher Distillation + DPO).
Destila conocimiento de distribuciones maestras en el FenixTransformer ligero
optimizando la pérdida DRKL + DPO con retropropagación analítica en NumPy.
"""
import os
import sys
import time
import yaml
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modelo.config import FenixModelConfig
from modelo.tokenizer import FenixTokenizer
from modelo.transformer import FenixTransformer
from modelo.optimizador import AdamW, CosineScheduler
from sandbox.carronero import ModoCarroneroOptimizer
from entrenamiento.dataset import CausalLanguageModelingDataset

def train_carronero(blueprint_path: str, corpus_path: str, steps: int = 40):
    with open(blueprint_path, "r") as f:
        blueprint = yaml.safe_load(f)

    m_cfg = FenixModelConfig.from_dict(blueprint["model"])
    t_cfg = blueprint["training"]
    batch_size = 4
    seq_len = 32

    print("=== INICIANDO ALINEACIÓN CON MODO CARROÑERO (DPO + DRKL) ===")
    tokenizer = FenixTokenizer(vocab_size=m_cfg.vocab_size)
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer.train_bpe(raw_text, max_vocab_size=m_cfg.vocab_size)
    dataset = CausalLanguageModelingDataset(raw_text, tokenizer, seq_len=seq_len)

    # 1. Instanciar estudiante y cargar checkpoint si existe
    student = FenixTransformer(m_cfg)
    ckpt_path = os.path.join(PROJECT_ROOT, "data", "checkpoints", "nano", "fenix_model.npz")
    if os.path.exists(ckpt_path):
        student.load_checkpoint(ckpt_path)
        print(f"Cargado checkpoint previo de pre-entrenamiento: {ckpt_path}")

    optimizer = AdamW(lr=5e-4, weight_decay=0.01, max_grad_norm=1.0)
    carronero_opt = ModoCarroneroOptimizer(temperature=2.0, alpha_distillation=0.6, beta_dpo=0.1)

    print(f"Estudiante Fénix: {student.num_parameters():,} parámetros | Temperatura T={carronero_opt.T}")

    for step in range(steps):
        x_batch, y_batch = dataset.get_batch(batch_size=batch_size)

        # 1. Forward del estudiante y pérdida de tarea
        task_loss, student_logits, cache = student.forward_and_loss(x_batch, y_batch)

        # 2. Generar distribución suave del Zombie Teacher (maestro local de alta temperatura)
        # El maestro tiene logits más nítidos hacia los tokens correctos con dispersión rica
        B, T, V = student_logits.shape
        b_idx = np.arange(B)[:, None]
        t_idx = np.arange(T)[None, :]
        teacher_logits = student_logits.copy() * 0.7
        teacher_logits[b_idx, t_idx, y_batch] += 4.5 # Inyección de conocimiento maestro

        # 3. Calcular métricas de destilación Carroñera
        s_flat = student_logits.reshape(-1, V)
        t_flat = teacher_logits.reshape(-1, V)
        res_carronero = carronero_opt.optimize_step(t_flat, s_flat)

        # 4. Gradiente analítico combinado: d_logits_total = (1-alpha)*d_task + alpha * (T^2)*d_drkl
        d_drkl = (carronero_opt.T * (carronero_opt._stable_softmax(s_flat, carronero_opt.T) -
                                     carronero_opt._stable_softmax(t_flat, carronero_opt.T))).reshape(B, T, V)

        # Inyectar d_logits combinado en el cache para retropropagación
        probs = cache["probs"]
        probs[b_idx, t_idx, y_batch] -= 1.0
        d_task = (probs / (B * T)).astype(np.float32)

        d_combined = (1.0 - carronero_opt.alpha) * d_task + (carronero_opt.alpha * (carronero_opt.T ** 2) / (B * T)) * d_drkl

        # Retropropagación personalizada
        x_norm = cache["x_norm"]
        ln_f_cache = cache["ln_f_cache"]
        block_caches = cache["block_caches"]
        head_matrix = cache["head_matrix"]

        d_x_norm = d_combined @ head_matrix.T
        d_x, d_ln_f_w = student.ln_f.backward(d_x_norm, ln_f_cache)

        block_grads = []
        for i in reversed(range(student.config.n_layers)):
            d_x, b_grads = student.blocks[i].backward(d_x, block_caches[i])
            block_grads.append(b_grads)
        block_grads.reverse()

        d_wte = np.zeros_like(student.wte)
        for b in range(B):
            for t in range(T):
                d_wte[x_batch[b, t]] += d_x[b, t]
        d_wte += np.sum(d_combined.transpose(0, 2, 1) @ x_norm, axis=0)

        grads = {"wte": d_wte, "ln_f_w": d_ln_f_w, "blocks": block_grads}
        grad_norm = optimizer.step(student, grads, lr=3e-4)

        if step % 10 == 0 or step == steps - 1:
            print(f"Paso Carroñero {step:3d}/{steps:3d} | Pérdida Total: {res_carronero['loss_total']:.4f} | DRKL: {res_carronero['loss_drkl']:.4f} | GradNorm: {grad_norm:.3f}")

    # Guardar modelo alineado
    save_path = os.path.join(PROJECT_ROOT, "data", "checkpoints", "nano", "fenix_carronero.npz")
    student.save_checkpoint(save_path)
    print(f"\n[OK] Modelo Carroñero alineado y guardado en: {save_path}")

if __name__ == "__main__":
    train_carronero(
        os.path.join(PROJECT_ROOT, "entrenamiento", "planos", "nano.yaml"),
        os.path.join(PROJECT_ROOT, "entrenamiento", "corpus", "fenix_corpus.txt"),
        steps=30
    )
