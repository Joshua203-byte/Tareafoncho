# Fase 3: fine-tuning

Este documento explica qué entrenamos, cómo, por qué y con qué fuentes. Las decisiones que dependen de un experimento se marcan como **pendientes** hasta que tengamos el piloto.

## 1. Datos y splits

Hay cuatro conjuntos de datos. Solo uno de ellos lo vemos completo; los otros tres tienen funciones distintas.

| Conjunto | De dónde sale | Tamaño | Qué hace | Cuándo se toca |
|---|---|---|---|---|
| **Train** | SQuAD 2.0 `train` | 130 319 preguntas, 442 artículos | Actualiza los pesos (gradientes) | Cada paso de entrenamiento |
| **`dev_select`** | Mitad de SQuAD 2.0 `validation` | 5 917 preguntas, 17 artículos | Monitorea sobreajuste, elige checkpoint, calibra τ | Al final de cada época |
| **`dev_report`** | Otra mitad de `validation` | 5 956 preguntas, 18 artículos | Números finales del reporte | Una sola vez, al final |
| **Test oculto** | Instructor | 50 preguntas, textos no de SQuAD | Evaluación en vivo | Día de la presentación |

**Por qué existen tres conjuntos además del train.** El train no sirve para medir: el modelo ya vio esas respuestas, así que siempre se ve bien ahí. Un conjunto de validación que no entrena es lo único que permite saber si el modelo generaliza.

**Por qué se dividen los artículos de `validation` en dos.** Si reservamos un conjunto para elegir (`dev_select`) y otro para reportar (`dev_report`), los números del reporte no quedan contaminados por las decisiones de entrenamiento. Pero hay un detalle: las preguntas de un mismo artículo comparten el mismo contexto. Si repartimos por pregunta, el modelo vería en `dev_select` un contexto que también aparece en `dev_report`, y los números del reporte quedarían optimistas. Por eso la división es **por artículo**, con semilla fija (42).

**Por qué `dev_select` se usa para varias cosas.** Elegir el checkpoint, el learning rate y el umbral τ son decisiones que se toman mirando datos. Si las tomamos sobre el mismo conjunto que reportamos, el número final queda inflado. Por eso `dev_report` se toca una sola vez.

**Nota de nombres.** En la literatura, `dev_select` es el conjunto de **validación** y `dev_report` el de **test**. El test oculto del instructor es el test final. No confundir nuestro `dev_report` con el test oficial de la rúbrica.

**Piloto.** Antes de la corrida completa, usamos 5 000 ejemplos del train y `dev_select` completo, para validar el pipeline. El resultado del piloto no se reporta.

## 2. Método: LoRA con rango bajo en varios módulos

### Qué es LoRA

LoRA congela los pesos originales `W` y entrena una actualización de bajo rango `ΔW = (α/r) · B · A`, donde `A` y `B` son matrices pequeñas con rango `r`. Solo se entrenan `A` y `B`.

Al terminar, `ΔW` se suma a `W` (*merge*). El modelo final tiene la misma arquitectura que el original, sin módulos extra.

### Qué dice el paper de LoRA

Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*, sección 7.1, "Which Weight Matrices in Transformer Should We Apply LoRA to?":

> "it is preferable to adapt more weight matrices than adapting a single type of weights with a larger rank."

Esa afirmación aparece en la sección 7.1 y se apoya en la Tabla 5 del paper, que compara configuraciones con el mismo presupuesto de **18 M de parámetros entrenables** sobre GPT-3 175B, en WikiSQL y MultiNLI.

Matices que hay que reportar:

- **La tabla apoya la idea, pero con poca diferencia.** En MultiNLI, la configuración con las cuatro matrices de atención a rango 2 obtiene 91.7, y la de Wq y Wv a rango 4 obtiene 91.3. En WikiSQL, ambas empatan en 73.7. La conclusión del paper es un juicio global sobre las dos tareas, no una victoria uniforme.
- **Solo se probó en matrices de atención.** En sus experimentos principales, LoRA se aplica solo a los pesos de atención (`Wq`, `Wv`), y los módulos MLP (feed-forward) están congelados. El paper no aplica LoRA a embeddings ni a LayerNorms, y deja esos casos para trabajo futuro.

### Qué implicaciones tiene para nosotros

Nuestra propuesta inicial incluía embeddings y feed-forward. Eso **no está respaldado por el paper de LoRA**: lo que sí respalda es que, dentro del mismo presupuesto de parámetros, repartir un rango bajo entre varias matrices de atención puede rendir igual o mejor que concentrar un rango alto en una sola.

Por eso la configuración inicial es:

| Parámetro | Valor | Justificación |
|---|---|---|
| Módulos objetivo | `q_lin`, `k_lin`, `v_lin`, `out_lin` (las cuatro matrices de atención de DistilBERT) | Cuatro matrices de atención, según la conclusión de la sección 7.1 |
| Rango `r` | 8 para empezar; luego 16 y 64 | Sweep de rango, como acordamos |
| `α` | `2 · r` | Heurística común en la práctica; con `α = r` el escalado es 1 |
| Dropout de LoRA | 0.1 | Heurística; mismo valor que el dropout de BERT |
| Cabeza QA | Entrenada completa (`modules_to_save=["qa_outputs"]`) | No existe en el checkpoint preentrenado |
| Embeddings, FFN, LayerNorm | Congelados | El paper de LoRA no los adapta |

**Pendiente:** si con `r = 64` en atención sigue por debajo del fine-tuning completo, se prueba adaptar también las capas feed-forward. Ese paso es una extensión mía, no algo que el paper valide.

### Cómo se verá en el reporte

El paper de LoRA justifica la elección de módulos y el rango bajo. Lo que no justifica es que LoRA iguale al fine-tuning completo en DistilBERT. Eso lo medimos con el control de fine-tuning completo.

## 3. Precisión mixta: AMP, fp16 y bf16

### Qué es AMP

**AMP** (*Automatic Mixed Precision*) es el mecanismo que ejecuta cada operación en la precisión adecuada. No es un formato de números; es la forma de usar dos formatos a la vez:

- Las operaciones pesadas, como las multiplicaciones de matrices, se hacen en 16 bits. Son más rápidas y ocupan la mitad de memoria.
- Las operaciones sensibles a la precisión, como `softmax`, LayerNorm o las sumas de la pérdida, se mantienen en 32 bits.
- Los pesos maestros y el estado del optimizador (Adam) siguen en 32 bits, para que las actualizaciones pequeñas no se pierdan.

En PyTorch, AMP se usa con `torch.autocast`, que decide la precisión por operación.

### fp16 vs bf16: los dos formatos de 16 bits

Un número de coma flotante tiene tres partes: signo, exponente (rango) y mantisa (precisión).

| Formato | Exponente | Mantisa | Problema típico | ¿Necesita `GradScaler`? |
|---|---|---|---|---|
| fp16 | 5 bits | 10 bits | Gradientes pequeños se redondean a cero (*underflow*); valores grandes pueden desbordar | Sí |
| bf16 | 8 bits (igual que fp32) | 7 bits | Menos precisión en la mantisa | No |

Con fp16, el `GradScaler` multiplica la pérdida por un factor antes del `backward` para que los gradientes no se pierdan, y lo divide después. Con bf16, el rango es el de fp32, así que el `GradScaler` no hace falta.

### Medición en esta GPU (RTX 4050, DistilBERT, batch 32, ventana de 384)

| Configuración | Tiempo por paso | VRAM pico |
|---|---|---|
| fp32 puro | 1 086 ms | 4.90 GB |
| AMP con fp16 | 413 ms | 3.72 GB |
| AMP con bf16 | 417 ms | 3.72 GB |

- AMP reduce el tiempo por paso a **menos de la mitad** (2.6×) y la VRAM en 1.2 GB.
- fp16 y bf16 tienen la misma velocidad y la misma memoria en esta GPU.

**Decisión:** AMP con bf16. Misma velocidad que fp16, sin `GradScaler`, y sin el riesgo de underflow.

## 4. Hiperparámetros

| Parámetro | Valor inicial | Fuente | Tipo de evidencia |
|---|---|---|---|
| Learning rate (completo) | 2e-5, 3e-5, 5e-5 (sweep) | [BERT, A.3](https://arxiv.org/abs/1810.04805): rango {5e-5, 3e-5, 2e-5}. [Curso HF, cap. 7](https://huggingface.co/learn/llm-course/chapter7/7): 2e-5 | Verificado |
| Learning rate (LoRA) | 2e-4 y 5e-4 (sweep) | [LoRA, Tabla 9](https://arxiv.org/abs/2106.09685): 4e-4 a 5e-4, pero en GLUE con RoBERTa-base | Extrapolado a QA |
| Batch size | 32 | [BERT, A.3](https://arxiv.org/abs/1810.04805): {16, 32}. BERT en SQuAD v1.1 (§4.2): 32 | Verificado |
| Épocas | 2 o 3 | BERT, A.3: {2, 3, 4}. BERT en SQuAD v1.1: 3. Curso HF: 3 | Verificado |
| Dropout del modelo | 0.1 | BERT, A.3: "always kept at 0.1" | Verificado |
| Weight decay | 0.01 | Curso HF: 0.01. BERT lo usa en preentrenamiento (A.2) | Verificado en el curso |
| Warmup | 6 % a 10 % de los pasos | BERT no lo especifica para fine-tuning. LoRA usa 0.06 en GLUE | Heurística |
| Gradient clipping | 1.0 | Práctica común | Heurística |
| Semilla | 42 | Convención | Convención |
| Precisión | AMP con bf16 | Medición de la sección 3 | Medido |

Lo que no encontré en las fuentes: un valor de warmup para fine-tuning de BERT en QA, y un valor de learning rate de LoRA para SQuAD. Esos dos se validan en el piloto.

## 5. Flujo de la corrida

1. **Piloto**: 5 000 ejemplos de train, 1 época, LoRA `r = 8`. Verifica que la loss baja y que la evaluación en `dev_select` funciona. Se compara contra el baseline de siempre `null` (que en `dev_select` da ≈ 0.50).
2. **Sweep corto de learning rate** en el piloto: 2e-4 y 5e-4.
3. **Corrida real con LoRA**: train completo, 2 o 3 épocas, con el mejor learning rate. Evaluación en `dev_select` al final de cada época.
4. **Sweep de rango**: `r = 16` y `r = 64`, con la misma configuración.
5. **Control**: fine-tuning completo con la misma configuración, para medir cuánto se pierde con LoRA.
6. **Calibración de τ** en `dev_select`, reponderado a 25 % sin respuesta (Fase 4).
7. **Evaluación final** en `dev_report`, una sola vez.

## 6. Métricas

- **EM y F1 globales** en `dev_select` y `dev_report`.
- **HasAns F1**: solo sobre preguntas con respuesta. Mide la calidad del modelo cuando sí hay respuesta.
- **NoAns accuracy**: solo sobre preguntas sin respuesta. Depende del umbral τ.
- **Loss de entrenamiento y de validación por época**. Si la validación sube mientras la de entrenamiento baja, hay sobreajuste.

Un modelo que siempre responde `null` obtiene 0.50 en el dev completo, pero 0 en HasAns F1. Por eso reportamos las dos métricas por separado.

## 7. Reproducibilidad

Cada corrida guarda:

- `config.json`: hiperparámetros, semilla, módulos LoRA y precisión.
- Hash del commit de git.
- `logs/train_loss.jsonl`: loss por paso.
- `logs/dev_metrics.json`: métricas por época.
- `logs/loss.png`: curva de loss, generada desde los logs.

## 8. Implementación: Trainer

Usamos `Trainer`, con documentación en el código: una sección por bloque (configuración, datos, métricas, entrenamiento, guardado), un docstring por función y una tabla que relaciona cada argumento de `TrainingArguments` con lo que hace por debajo.

## 9. Decisiones tomadas

- **Learning rate**: 2e-4 con LoRA y 3e-5 con fine-tuning completo. El sweep del piloto no se hizo: el tiempo lo dejó fuera, y se usó la referencia de la sección 4.
- **Rango de LoRA**: r = 8 y r = 16 a los mismos pasos. r = 16 mejora sobre r = 8 (≈ 2.6 puntos del proxy). No se corrió r = 64: ya no había tiempo, y la comparación entre r = 8 y r = 16 mostró que la extracción casi no cambia con el rango.
- **Control con fine-tuning completo**: supera a LoRA (0.721 contra 0.697, en el proxy sobre `dev_select`). Por eso el modelo final es de fine-tuning completo.
- **Épocas**: 3. La curva se aplana alrededor de 0.72 entre la época 3 y la 4. Se elige la época 3 por ser la mejor con el mismo τ que las demás.
- **Dependencia nueva**: `peft`, fijado en `requirements.txt`.

## 10. Resultados finales de la fase 3

Modelo elegido: `full_ft_4ep`, época 3, fine-tuning completo, τ = −2. Detalle de cada corrida en [`docs/experimentos_fase3.md`](experimentos_fase3.md).

| Conjunto | EM | F1 | HasAns F1 | NoAns acc | Proxy de la guía (reponderado) |
|---|---|---|---|---|---|
| `dev_select` (selección) | 61.53 (época 2) | 65.10 (época 2) | 79.17 (época 2) | 51.95 (época 2) | 0.721 |
| `dev_report` (una sola vez) | 62.56 | 66.27 | 74.20 | 57.83 | 0.694 |

El proxy de la guía usa una regla para el crédito parcial, no el juez LLM, así que no es la nota real. Ver [`docs/fase4_umbral_y_evaluacion.md`](fase4_umbral_y_evaluacion.md) para la calibración de τ y la limitación.

**Lo que no se hizo**: el sweep de learning rate del piloto, la corrida con r = 64 y la comparación con un encoder más grande. Quedan como trabajo futuro, con el tiempo como razón.
