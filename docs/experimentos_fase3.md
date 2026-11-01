# Registro de experimentos: Fase 3

Cada corrida deja una entrada aquí, de la más reciente a la más antigua. La entrada registra qué
se quería responder, qué pasó, qué se concluye y qué pregunta abre la siguiente
corrida. Así el historial de decisiones queda legible para cualquiera del equipo,
incluido el día del Q&A.

Los datos crudos de cada corrida están en `logs/runs/<run>/`:
`config.json`, `train_log.jsonl`, `dev_select_metrics.jsonl` y `tau_sweep_*.json`.

## Plan de rangos

La ruta acordada, de menor a mayor costo:

1. **LoRA r = 8** sobre las cuatro matrices de atención (`q_lin`, `k_lin`, `v_lin`, `out_lin`).
2. Si no alcanza, **LoRA r = 16**, después **r = 32**.
3. Si con r = 32 no hay resultados buenos, **fine-tuning completo** con la misma configuración.

Cada salto de rango se justifica con la entrada anterior. No saltamos de r = 8 a fine-tuning completo sin antes subir el rango.

Métricas que reportamos en cada entrada: **EM y F1 globales**, **HasAns F1** (calidad del modelo cuando hay respuesta) y **NoAns accuracy** (depende de τ).

---

## Corrida 5: full_data_r16_ep2

**Pregunta.** Con dos épocas, ¿r = 16 mejora sobre r = 8 (`full_data_r8_ep2`) a los mismos pasos (8 236)?

**Configuración.** LoRA r = 16, α = 32. Igual que `full_data_r8_ep2` salvo el rango. Dos épocas, 8 236 pasos. Pérdida promedio de la corrida: ver `config.json`.

**Resultado en `dev_select`** (`logs/runs/full_data_r16_ep2/`). En la monitorización de entrenamiento (τ = 0):

| Época | EM | F1 | HasAns F1 | NoAns acc |
|---|---|---|---|---|
| 1 | 58.05 | 61.18 | 61.19 | 61.16 |
| 2 | 60.54 | 63.25 | 60.51 | 65.81 |

**Barrido de τ** (`logs/runs/full_data_r16_ep2/tau_sweep_epoch_2.json`), sobre la época 2:

| τ | EM | F1 | HasAns F1 | NoAns acc | % null |
|---|---|---|---|---|---|
| −∞ | 34.17 | 38.65 | 80.01 | 0.0 | 0.0 |
| −4 | 45.06 | 49.32 | 78.36 | 22.20 | 13.15 |
| −2 | 53.79 | 57.44 | 73.16 | 42.76 | 28.19 |
| −1 | 58.05 | 61.31 | 68.18 | 54.89 | 37.84 |
| −0.5 | 59.22 | 62.23 | 64.45 | 60.15 | 42.98 |
| 0 | 60.42 | 63.18 | 60.58 | 65.61 | 48.27 |
| +0.5 | 61.37 | 63.74 | 55.90 | 71.07 | 54.15 |
| +1 | 62.14 | 64.11 | 51.18 | 76.20 | 59.79 |
| +2 | 61.35 | 62.63 | 39.74 | 84.01 | 70.44 |
| +4 | 57.09 | 57.49 | 17.77 | 94.61 | 88.03 |
| +∞ | 51.73 | 51.73 | 0.07 | 100.0 | 100.0 |

**Aproximación reponderada a 25 % sin respuesta.** Mejor τ = −2 con puntaje ≈ 65.6 (τ = −1: ≈ 64.9; τ = 0: ≈ 61.8).

**Comparación con `full_data_r8_ep2` (mismos pasos, τ = 0):** EM +3.8 (56.67 → 60.42), HasAns F1 ≈ igual (60.83 → 60.58), NoAns acc +6.5 (59.07 → 65.61). Puntaje reponderado con τ = −2: 62.9 → 65.6.

**Lectura.**
- r = 16 mejora frente a r = 8 con el mismo número de pasos. La ganancia es de unos 2.6 puntos en el puntaje reponderado.
- Casi toda la ganancia viene de NoAns, no de la extracción. Con τ = −∞ (siempre span), HasAns F1 es 80.0 contra 79.6 de r = 8: la capacidad de extraer apenas cambió. Lo que mejora es la decisión de cuándo responder null.
- La mejora en EM con τ = 0 (+3.8) viene del mismo efecto: el umbral de r = 16 queda mejor ubicado, y por eso su mejor τ aproximado (−2) está a la misma altura que en r = 8.
- Un punto a vigilar: el puntaje reponderado depende de la proporción de null (25 %). Es una aproximación; la decisión final de τ es de la Fase 4.

**Pregunta que abre la siguiente corrida.** ¿Subir a r = 32 con dos épocas mejora más el puntaje reponderado que r = 16? Si la ganancia es menor a unos 2 puntos, el plan indica pasar a fine-tuning completo como control.

## Corrida 4: full_data_r16

**Pregunta.** Con el mismo número de pasos que `full_data_r8` (1 época, 4 118 pasos), ¿subir el rango de 8 a 16 mejora HasAns F1 y EM?

**Configuración.** LoRA r = 16, α = 32. Igual que `full_data_r8` salvo el rango. Una época, 4 118 pasos. Tiempo: 1 218 s (≈ 20 min). Pérdida promedio de la corrida: 1.97.

**Resultado en `dev_select`** (`logs/runs/full_data_r16/`):

| Época | τ | EM | F1 | HasAns F1 | NoAns acc |
|---|---|---|---|---|---|
| 1 | 0 | 56.08 | 58.65 | 55.26 | 61.82 |

**Barrido de τ** (`logs/runs/full_data_r16/tau_sweep_epoch_1.json`):

| τ | EM | F1 | HasAns F1 | NoAns acc | % null |
|---|---|---|---|---|---|
| −∞ | 32.36 | 37.20 | 77.02 | 0.0 | 0.0 |
| −4 | 37.94 | 42.59 | 75.96 | 11.41 | 6.90 |
| −2 | 46.39 | 50.41 | 70.75 | 31.42 | 21.92 |
| −1 | 52.12 | 55.50 | 64.72 | 46.88 | 34.68 |
| −0.5 | 54.05 | 57.02 | 60.04 | 54.20 | 41.79 |
| 0 | 56.04 | 58.65 | 55.40 | 61.69 | 48.84 |
| +0.5 | 57.19 | 59.22 | 49.30 | 68.49 | 56.26 |
| +1 | 58.17 | 59.69 | 42.16 | 76.07 | 64.63 |
| +2 | 57.78 | 58.48 | 27.40 | 87.51 | 79.03 |
| +∞ | 51.73 | 51.73 | 0.07 | 100.0 | 100.0 |

**Aproximación reponderada a 25 % sin respuesta.** Mejor τ = −2 con puntaje ≈ 60.9 (τ = 0: ≈ 57.0).

**Comparación con `full_data_r8` a los mismos pasos (τ = 0):** EM +3.1 (52.97 → 56.04), HasAns F1 +2.4 (53.05 → 55.40 en el barrido), NoAns acc +4.1. Puntaje reponderado: 58.6 → 60.9.

**Lectura.** Subir el rango con los mismos pasos sí ayuda, aunque poco. Eso responde la pregunta: r = 16 mejora frente a r = 8 a igual número de pasos.

## Corrida 3: full_data_r8_ep2

**Pregunta.** ¿Más épocas mejoran HasAns F1 y EM con r = 8? ¿Cuánto más?

**Configuración.** LoRA r = 8, α = 16. Igual que `full_data_r8` salvo el número de épocas. Dos épocas, 8 236 pasos. El scheduler lineal decae hasta cero al final de la época 2, así que la época 1 de esta corrida no es equivalente a la corrida de una época. Tiempo: 2 381 s (≈ 40 min). Pérdida promedio de la corrida: 1.81.

**Resultado en `dev_select`** (`logs/runs/full_data_r8_ep2/`). En la monitorización de entrenamiento (τ = 0):

| Época | EM | F1 | HasAns F1 | NoAns acc |
|---|---|---|---|---|
| 1 | 53.63 | 56.96 | 61.91 | 52.34 |
| 2 | 56.72 | 59.96 | 60.71 | 59.27 |

**Barrido de τ** (`logs/runs/full_data_r8_ep2/tau_sweep_epoch_2.json`), sobre la época 2:

| τ | EM | F1 | HasAns F1 | NoAns acc | % null |
|---|---|---|---|---|---|
| −∞ | 33.41 | 38.43 | 79.57 | 0.0 | 0.0 |
| −4 | 40.88 | 45.67 | 77.58 | 15.85 | 9.92 |
| −2 | 48.17 | 52.47 | 72.74 | 33.54 | 22.68 |
| −1 | 52.75 | 56.47 | 67.81 | 45.86 | 32.70 |
| −0.5 | 54.91 | 58.46 | 65.02 | 52.34 | 38.08 |
| 0 | 56.67 | 59.92 | 60.83 | 59.07 | 44.31 |
| +0.5 | 58.14 | 60.85 | 56.11 | 65.28 | 50.85 |
| +1 | 59.73 | 62.02 | 51.59 | 71.76 | 57.19 |
| +2 | 60.62 | 62.05 | 39.38 | 83.23 | 70.29 |
| +4 | 56.08 | 56.38 | 13.90 | 96.08 | 90.79 |
| +∞ | 51.73 | 51.73 | 0.07 | 100.0 | 100.0 |

**Aproximación reponderada a 25 % sin respuesta.** Mejor τ = −2 con puntaje ≈ 62.9 (τ = 0: ≈ 60.4).

**Comparación con `full_data_r8` (1 época, τ = 0):** EM +3.7 (52.97 → 56.67), HasAns F1 +7.8 (53.05 → 60.83), puntaje reponderado 58.6 → 62.9.

**Lectura.**
- Más épocas mejoran EM y HasAns F1. La mejora en HasAns F1 es la más grande de los experimentos hasta ahora.
- HasAns F1 no sube de forma monótona: en la época 1 (con learning rate todavía alto) vale 61.9 con τ = 0, y en la época 2 baja a 60.7. NoAns sí sube. La medida que decide es el puntaje reponderado, que sigue subiendo entre épocas (≈ 59.5 → 60.4 con τ = 0).
- Los valores del barrido y los de la monitorización difieren en ≈ 0.05 EM (56.67 vs 56.72). No lo investigué; es una diferencia pequeña.

**Pregunta que abre la siguiente corrida.** Con más pasos ya obtenemos la mejor combinación (r = 8, 2 épocas). ¿Ese beneficio sigue creciendo con r = 16 a 2 épocas, o ya estamos cerca de lo que LoRA puede dar con este presupuesto?

## Corrida 2: full_data_r8

**Pregunta.** Con más datos y pasos (train completo, 131 754 ventanas), ¿HasAns F1 supera 10.7 de la piloto?

**Configuración.** LoRA r = 8, α = 16. Mismos hiperparámetros que la piloto. Una época, 4 118 pasos. Único cambio: train completo en lugar de 5 000 ejemplos. Tiempo: 1 189 s (≈ 20 min).

**Pérdida.** Empieza en 5.7 y baja a ≈ 1.65. Pérdida promedio de la corrida: 2.05.

**Resultado en `dev_select` con τ = 0:**

| Métrica | Valor |
|---|---|
| EM | 52.97 |
| F1 | 55.52 |
| HasAns F1 | 53.05 |
| NoAns accuracy | 57.83 |

**Barrido de τ** (`logs/runs/full_data_r8/tau_sweep_epoch_1.json`):

| τ | EM | F1 | HasAns F1 | NoAns acc | % null |
|---|---|---|---|---|---|
| −∞ (siempre span) | 30.67 | 36.09 | 74.71 | 0.0 | 0.0 |
| −2 | 42.00 | 46.60 | 69.81 | 24.91 | 18.35 |
| −1 | 48.17 | 51.79 | 62.65 | 41.65 | 32.63 |
| 0 | 52.97 | 55.54 | 53.30 | 57.63 | 48.15 |
| +1 | 56.40 | 57.83 | 40.02 | 74.47 | 65.19 |
| +∞ (siempre null) | 51.73 | 51.73 | 0.07 | 100.0 | 100.0 |

**Lectura.**
- El modelo aprendió a extraer: HasAns F1 llega a 74.7 cuando se fuerza el span. La piloto estaba en 10.7.
- τ controla un intercambio: bajar τ mejora las preguntas con respuesta y empeora las sin respuesta.
- `dev_select` tiene 48 % de preguntas sin respuesta. El set oculto tiene ≈ 25 %, así que el τ que maximiza F1 aquí no es necesariamente el que conviene allá.

**Aproximación reponderada a 25 % sin respuesta.** Pondera HasAns F1 por 0.75 y NoAns accuracy por 0.25:

| τ | Puntaje reponderado |
|---|---|
| −∞ | 56.0 |
| −4 | 56.6 |
| −2 | **58.6** |
| −1 | 57.4 |
| −0.5 | 56.3 |
| 0 | 54.4 |
| +2 | 41.6 |

Es una estimación: usa las medias por tipo de pregunta, no una re-evaluación ponderada. Sugiere que τ = −2 rendiría mejor que τ = 0 en el set oculto. La decisión final de τ es de la Fase 4 y se toma con el dev reponderado, nunca con `dev_report`.

**Pregunta que abre la siguiente corrida.** ¿Más épocas mejoran HasAns F1 y EM? ¿Con r = 16 mejora más que con más épocas?

## Corrida 1: piloto_r8

**Pregunta.** ¿El pipeline completo funciona de punta a punta (datos, entrenamiento, evaluación, merge)? ¿La pérdida baja?

**Configuración.** LoRA r = 8, α = 16, dropout 0.1. Adaptadores en `q_lin`, `k_lin`, `v_lin`, `out_lin`. Cabeza `qa_outputs` entrenada completa. 5 000 ejemplos de train (5 058 ventanas), 1 época, 159 pasos. Learning rate 2e-4, batch 32, warmup 6 %, scheduler lineal, bf16. Semilla 42. Parámetros entrenables: 296 450 de 66 660 868 (0.445 %).

**Resultado de entrenamiento.** La pérdida baja de 5.31 a 3.29. Pérdida promedio de la corrida: 4.06. Tiempo: 91 s (55.6 ejemplos/s).

**Resultado en `dev_select`** (τ = 0):

| Métrica | Valor |
|---|---|
| EM | 51.73 |
| F1 | 51.73 |
| HasAns F1 | 0.07 |
| NoAns accuracy | 100.0 |

**Lectura.** El modelo responde `null` a todo. EM y F1 coinciden con el baseline de siempre `null` (≈ 50 % en este conjunto). Esto no dice todavía si el problema es el modelo o el umbral. El barrido de τ lo separa (ver abajo).

**Hipótesis.**
- H1: 159 pasos son pocos. El modelo aprende primero el sesgo hacia `[CLS]`, porque es la etiqueta más frecuente en el train (las ventanas que no contienen la respuesta, más las preguntas sin respuesta).
- H2: el umbral τ = 0 es demasiado conservador. Si los spans tienen scores bajos, el `[CLS]` gana siempre.

**Diagnóstico: barrido de τ** (`scripts/sweep_tau.py --checkpoint epoch_1`, `logs/runs/piloto_r8/tau_sweep_epoch_1.json`):

| τ | EM | HasAns F1 | NoAns acc | % null |
|---|---|---|---|---|
| −∞ (siempre span) | 2.48 | 10.73 | 0.0 | 0.0 |
| −4 | 47.12 | 1.00 | 90.42 | 90.45 |
| −2 a +∞ | 51.73 | 0.07 | 100.0 | 100.0 |

**Conclusión.** El umbral no es la causa. Aun obligando al modelo a responder siempre con un span, HasAns F1 es 10.7: los spans que propone son casi todos incorrectos. El modelo no aprendió a extraer respuestas todavía. Se descarta H2 y H1 queda como la explicación más probable: 159 pasos no alcanzan para aprender la extracción.

**Pregunta que abre la siguiente corrida.** ¿Con más pasos de entrenamiento, HasAns F1 sube por encima de 10.7 con τ razonable? La siguiente corrida usa el train completo (131 754 ventanas), mismo r = 8 y misma tasa de aprendizaje, para cambiar una sola variable: la cantidad de datos y pasos.
