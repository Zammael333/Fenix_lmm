.PHONY: all setup build-c benchmark-c certs run-kernel run-kill-switch run-maestro train-nano train-carronero chat test clean

SHELL := /usr/bin/bash
PYTHON := python3
CC := gcc
CFLAGS := -O3 -mavx2 -std=c99 -Wall -Wextra

all: setup build-c test

setup: certs
	@mkdir -p certs data/mmap_store data/checkpoints bin lib audit /dev/shm/fenix_volatile 2>/dev/null || true
	@chmod +x kernel/fenswitch.sh fenix_chat.py
	@echo "[OK] Entorno Fénix configurado correctamente."

build-c:
	@mkdir -p bin lib
	@echo "Compilando acelerador C99 / AVX2 (gcc -O3 -mavx2)..."
	@$(CC) $(CFLAGS) -shared -fPIC c_src/fenix_acelerador.c -o lib/libfenix_accelerator.so -lm
	@$(CC) $(CFLAGS) c_src/fenix_acelerador.c -DFENIX_BENCHMARK -o bin/fenix_acelerador -lm
	@echo "[OK] Binarios compilados exitosamente en lib/ y bin/."

benchmark-c: build-c
	@./bin/fenix_acelerador

certs:
	@$(PYTHON) seguridad/cert_manager.py

run-kernel:
	@$(PYTHON) kernel/kernel_central.py

run-kill-switch:
	@$(PYTHON) kernel/fenswitch.py

run-maestro:
	@$(PYTHON) enjambre/nodo_maestro.py

train-nano:
	@$(PYTHON) entrenamiento/pretrain.py --plano entrenamiento/planos/nano.yaml --steps 100

train-carronero:
	@$(PYTHON) entrenamiento/entrenar_carronero.py

chat:
	@$(PYTHON) fenix_chat.py

test:
	@echo "Ejecutando suite completa de validación Fénix..."
	@$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

clean:
	@rm -rf bin/* lib/* certs/*.crt certs/*.key audit/*.log audit/*.jsonl data/mmap_store/*.bin data/test_* /dev/shm/fenix_volatile/* 2>/dev/null || true
	@echo "[OK] Limpieza de residuos completada."
