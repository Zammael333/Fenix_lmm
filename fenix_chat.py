#!/usr/bin/env python3
"""
Interfaz de Línea de Comandos (CLI) de Chat Interactivo con Fénix LLM.
Integra enrutamiento de Capa Cero en O(1) con generación neuronal autorregresiva (KV-Cache).
"""
import os
import sys
import time
import argparse

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modelo.config import FenixModelConfig
from modelo.tokenizer import FenixTokenizer
from modelo.transformer import FenixTransformer
from modelo.generador import FenixGenerator
from routing.capa_cero_router import CapaCeroRouter

def start_chat(checkpoint_dir: str = None):
    ckpt_dir = checkpoint_dir or os.path.join(PROJECT_ROOT, "data", "checkpoints", "nano")
    model_path = os.path.join(ckpt_dir, "fenix_model.npz")
    tok_path = os.path.join(ckpt_dir, "fenix_tokenizer.json")
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")

    print("=" * 70)
    print("      🔥 FÉNIX LLM - CONSOLA DE INTERACCIÓN SOBERANA 🔥")
    print("=" * 70)

    # 1. Cargar router de Capa Cero
    router = CapaCeroRouter(cfg_path)

    # 2. Cargar modelo neuronal si existe
    generator = None
    if os.path.exists(model_path) and os.path.exists(tok_path):
        print(f"Cargando modelo neuronal desde: {model_path}...")
        tok = FenixTokenizer(vocab_size=512)
        tok.load(tok_path)
        m_cfg = FenixModelConfig(vocab_size=512, d_model=64, n_heads=4, n_layers=2, d_ff=172, max_seq_len=128)
        model = FenixTransformer(m_cfg)
        model.load_checkpoint(model_path)
        generator = FenixGenerator(model, tok)
        print(f"Modelo en línea: {model.num_parameters():,} parámetros activos con KV-Cache.")
    else:
        print("Aviso: Checkpoint neuronal no encontrado. Ejecuta 'python3 entrenamiento/pretrain.py' para entrenar.")

    print("\nEscribe tu prompt o comandos deterministas (/status, /net, /mem, /version, /exit):")
    print("-" * 70)

    while True:
        try:
            user_input = input("\n[Operador] > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "exit", "quit"]:
                print("Cerrando sesión de Fénix LLM. Perímetro seguro.")
                break

            # 1. Interceptar por Capa Cero
            route_status, route_result = router.route(user_input)

            if route_status == "BYPASS":
                print(f"\n⚡ [CAPA CERO O(1)]:\n{route_result}")
                continue
            elif route_status == "ROUTE_REJECTED":
                print(f"\n🛑 [SEGURIDAD]: {route_result}")
                continue

            # 2. Si es inferencia de LLM y el generador está disponible
            if generator is not None:
                print("\n🧠 [FÉNIX LLM]:")
                t0 = time.perf_counter()
                reply = generator.generate(
                    user_input,
                    max_new_tokens=40,
                    temperature=0.7,
                    top_k=30,
                    top_p=0.9
                )
                t_elapsed = (time.perf_counter() - t0) * 1000.0
                print(reply)
                print(f"\n[Telemetría]: Generado en {t_elapsed:.1f} ms con KV-Cache.")
            else:
                print(f"\n[INFERENCIA SIMULADA]: Recibido '{user_input}'. Entrena el modelo para respuestas completas.")

        except (KeyboardInterrupt, EOFError):
            print("\nInterrupción detectada. Saliendo de forma segura.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consola de Chat Fénix")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()
    start_chat(args.checkpoint_dir)
