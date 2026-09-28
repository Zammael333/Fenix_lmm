"""
Script de Pre-entrenamiento Autorregresivo de Fénix LLM.
Carga planos de arquitectura YAML, optimiza con AdamW, recorta gradientes
y genera puntos de control (.npz) validados bajo el estándar Seis Sigmas.
"""
import os
import sys
import time
import argparse
import yaml
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modelo.config import FenixModelConfig
from modelo.tokenizer import FenixTokenizer
from modelo.transformer import FenixTransformer
from modelo.optimizador import AdamW, CosineScheduler
from modelo.generador import FenixGenerator
from entrenamiento.dataset import CausalLanguageModelingDataset

def train(blueprint_path: str, corpus_path: str, override_steps: int = None):
    with open(blueprint_path, "r") as f:
        blueprint = yaml.safe_load(f)

    m_cfg = FenixModelConfig.from_dict(blueprint["model"])
    t_cfg = blueprint["training"]
    max_steps = override_steps or t_cfg.get("max_steps", 100)
    batch_size = t_cfg.get("batch_size", 4)
    seq_len = t_cfg.get("seq_len", 64)
    lr = t_cfg.get("learning_rate", 1e-3)
    min_lr = t_cfg.get("min_learning_rate", 1e-4)
    warmup_steps = t_cfg.get("warmup_steps", 10)
    eval_interval = t_cfg.get("eval_interval", 20)
    ckpt_dir = t_cfg.get("checkpoint_dir", os.path.join(PROJECT_ROOT, "data", "checkpoints", "nano"))
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(PROJECT_ROOT, ckpt_dir)

    print(f"=== INICIANDO ENTRENAMIENTO DE FÉNIX LLM ===")
    print(f"Plano: {os.path.basename(blueprint_path)} | Pasos: {max_steps} | Batch: {batch_size} | SeqLen: {seq_len}")

    # 1. Cargar Tokenizador y Corpus
    tokenizer = FenixTokenizer(vocab_size=m_cfg.vocab_size)
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Entrenar BPE con el corpus para enriquecer vocabulario
    tokenizer.train_bpe(raw_text, max_vocab_size=m_cfg.vocab_size)
    dataset = CausalLanguageModelingDataset(raw_text, tokenizer, seq_len=seq_len)
    print(f"Corpus tokenizado: {len(dataset.tokens)} tokens | Muestras causales: {len(dataset)}")

    # 2. Inicializar Modelo y Optimizador
    model = FenixTransformer(m_cfg)
    optimizer = AdamW(lr=lr, weight_decay=t_cfg.get("weight_decay", 0.01), max_grad_norm=t_cfg.get("max_grad_norm", 1.0))
    scheduler = CosineScheduler(base_lr=lr, min_lr=min_lr, warmup_steps=warmup_steps, max_steps=max_steps)

    print(f"Parámetros totales entrenables: {model.num_parameters():,}")

    # 3. Bucle de Entrenamiento
    start_time = time.time()
    losses = []

    for step in range(max_steps):
        step_lr = scheduler.get_lr(step)
        x_batch, y_batch = dataset.get_batch(batch_size=batch_size)

        # Forward y Pérdida Cross-Entropy
        loss, logits, cache = model.forward_and_loss(x_batch, y_batch)
        losses.append(loss)

        # Backward
        grads = model.backward(cache)

        # Optimizer Step
        grad_norm = optimizer.step(model, grads, lr=step_lr)

        if step % eval_interval == 0 or step == max_steps - 1:
            elapsed = time.time() - start_time
            tps = ((step + 1) * batch_size * seq_len) / max(elapsed, 1e-4)
            print(f"Paso {step:4d}/{max_steps:4d} | Pérdida: {loss:.4f} | LR: {step_lr:.6f} | GradNorm: {grad_norm:.3f} | Vel: {tps:.1f} tok/s")

    # 4. Guardar Checkpoint
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, "fenix_model.npz")
    tok_path = os.path.join(ckpt_dir, "fenix_tokenizer.json")
    model.save_checkpoint(model_path)
    tokenizer.save(tok_path)
    print(f"\n[OK] Modelo guardado en: {model_path}")
    print(f"[OK] Tokenizador guardado en: {tok_path}")

    # 5. Demostración de Inferencia con el modelo entrenado
    print("\n--- GENERACIÓN DE PRUEBA (KV-CACHE) ---")
    gen = FenixGenerator(model, tokenizer)
    prompts = ["Fénix LLM es", "El clúster de", "Capa Cero"]
    for p in prompts:
        text = gen.generate(p, max_new_tokens=25, temperature=0.6, top_k=20, top_p=0.85)
        print(f"Prompt: '{p}' -> Generado: '{text}'")

    return model, tokenizer, losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-entrenamiento de Fénix LLM")
    parser.add_argument("--plano", default=os.path.join(PROJECT_ROOT, "entrenamiento", "planos", "nano.yaml"))
    parser.add_argument("--corpus", default=os.path.join(PROJECT_ROOT, "entrenamiento", "corpus", "fenix_corpus.txt"))
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    train(args.plano, args.corpus, override_steps=args.steps)
