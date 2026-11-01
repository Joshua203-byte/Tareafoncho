# Registro de experimentos: Fase 3

Cada corrida deja una entrada aquí, en orden cronológico. La entrada registra qué
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
