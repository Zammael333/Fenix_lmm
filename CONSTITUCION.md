# CONSTITUCIÓN DE FÉNIX LLM

Este documento define la **Constitución del Sistema Fénix LLM**. No es un conjunto de sugerencias o guías opcionales, sino el marco de directivas innegociables, invariantes matemáticos y límites físicos que rigen el comportamiento del clúster de 33 nodos heterogéneos y su enjambre de microagentes. Su propósito absoluto es actuar como el **ancla de seguridad del sistema** y el cordón umbilical de alineación en tiempo real.

---

## 1. Bucle de Alineación Estocástica y Destilación (RLAIF / Modo Carroñero)

### Formulación Matemática
La optimización del enjambre se rige por la función de recompensa combinada con la penalización por divergencia Kullback-Leibler inversa suavizada con temperatura $T$:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ R(x, y) - \beta D_{KL}(\pi_\theta(\cdot|x) \parallel \pi_{ref}(\cdot|x)) \right]$$

Complementado por la pérdida combinada de destilación del estudiante frente al modelo maestro (*Zombie Teacher*):

$$\mathcal{L}_{total} = (1 - \alpha) \mathcal{L}_{DPO}(y_w, y_l) + \alpha \cdot T^2 \cdot D_{KL}\left( \sigma\left(\frac{z_{teacher}}{T}\right) \parallel \sigma\left(\frac{z_{student}}{T}\right) \right)$$

### Invariante de Control
Para forzar la transferencia de conocimiento hacia los microagentes exploradores ($\pi_\theta$) en la sandbox:
- Si un agente muta de forma caótica en el patio de juegos adversarial y se desvía de los pesos originales del clon de anclaje ($\pi_{ref}$), la penalización $\beta D_{KL}$ se dispara.
- El **Árbitro** liquida de inmediato el proceso para proteger la coherencia lógica del sistema antes de que contamine el clúster.

---

## 2. Doctrina de la Sombra y Rotación de Espejos

### Vigilancia Ortogonal
- Cada anillo de ejecución funcional (ej. Anillo 2 de Neuroeconomía y Marketing) está emparejado de forma cíclica y asíncrona con una **Sombra de Seguridad Defensiva o Análisis Forense**.
- La sombra no interviene en el flujo de inferencia principal en estado normal, pero audita en tiempo real las salidas y transiciones de memoria.
- Rompe la visión de túnel del agente observado, impidiendo que un microagente defectuoso oculte o falsee un fallo en su propio registro.
- Las sombras rotan de forma pseudoaleatoria para evitar patrones de colusión entre procesos.

---

## 3. Ciclo Biológico y de Seguridad ("Muerte Limpia")

### Error Budget Seis Sigmas
- El presupuesto de error tolerable se fija estrictamente en **3.4 defectos por millón de operaciones** (SLO 99.99973%).
- El consumo de recursos (CPU, ciclos de cómputo, excepciones flotantes) se audita en cada ciclo de despacho.

### Protocolo de Muerte Limpia
- Si un nodo entra en bucle infinito, propaga valores `NaN` o cruza el umbral de presupuesto de error, el orquestador aplica una terminación aislada (`Clean Death`).
- **Autopsia Forense**: Antes de purgar el proceso, un auditor aislado extrae el vector de error de los residuos volátiles en memoria (`/dev/shm`), diagnostica la causa raíz y compila automáticamente una réplica inmune con parches preventivos.

---

## 4. Vectores de Seguridad Activa y Confinamiento Termodinámico

### Perímetro Físico Innegociable
- **Memoria RAM**: 16 GB estrictos en el nodo maestro (confinamiento mediante paginación predictiva `mmap` y evicción por Filtro de Kalman).
- **Almacenamiento**: 512 GB SSD.
- **Estrangulamiento Térmico**: Umbral máximo de $78.0^\circ\text{C}$. Ante sobrecalentamiento, el despachador reduce la concurrencia de microagentes y serializa estados a disco.

### Biometría y Detección de Coerción
- **Dinámica de Pulsaciones Criptográficas**: Verificación de la cadencia de escritura (tiempo de vuelo entre pulsaciones con tolerancia de $25.0\,\text{ms}$).
- **Modo de Falsa Complacencia (Honeypot)**: Si se detectan anomalías de tecleo, intrusiones automatizadas o incoherencias acústicas, el enrutador simula obediencia pero desvía todas las acciones a un entorno sandbox aislado, aislando las claves maestras.

---

## 5. Especificación del Kill Switch Out-of-Band (Air-Gapped)

### Protocolo de Emergencia
El sistema inmunitario de corte opera con independencia del intérprete principal de Python:
- Demonio en segundo plano ejecutándose en el puerto privado `9999` con bajo consumo de memoria ($<2\,\text{MB}$).
- Tres niveles estrictos de interrupción mediante token criptográfico `RED_ALERT_KILL_33`:
  1. **Nivel 1 (`STOP_SANDBOX`)**: Detención exclusiva de los procesos del polígono adversarial (*El Coliseo*).
  2. **Nivel 2 (`KILL_ENJAMBRE`)**: Emisión de `SIGTERM` / `SIGKILL` a todo el árbol de procesos del enjambre (`PGID`), liberando el 100% de la memoria volátil.
  3. **Nivel 3 (`HARD_SHUTDOWN`)**: Sincronización forzada de disco (`sync`) y apagado de emergencia de hardware.
- **Dead Man's Switch**: Monitoreo de latido continuo contra un token en `/dev/shm`. Si el latido se extingue, se ejecuta una purga inmediata de memoria volátil.
