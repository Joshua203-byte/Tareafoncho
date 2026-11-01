# Fase 1: datos y elección del modelo

Este documento explica qué es SQuAD 2.0, qué encontramos en los datos y por qué elegimos el modelo base. Los números salen de `scripts/explore_data.py` y están guardados en `logs/data_stats.json`.

## 1. Qué es SQuAD 2.0

**SQuAD** (Stanford Question Answering Dataset) es un benchmark de *lectura comprensiva*. Cada ejemplo tiene tres piezas:

- **Contexto**: un párrafo de Wikipedia.
- **Pregunta**: escrita por un crowdworker mirando el párrafo.
- **Respuesta**: un *span*, es decir, un fragmento del contexto copiado tal cual, con su posición de inicio en caracteres (`answer_start`).

La tarea es *extractiva*: el modelo no genera texto, solo señala dónde empieza y dónde termina la respuesta dentro del contexto.

**Versión 1.0** (Rajpurkar et al., 2016): todas las preguntas tienen respuesta.

**Versión 2.0** (Rajpurkar et al., 2018): agrega preguntas *sin respuesta* (`is_impossible`). Son preguntas escritas para parecer respondibles, pero el párrafo no contiene la respuesta. El modelo debe decidir cuándo no contestar. Ese es el sentido de "no answer" en la guía.

Estructura de un ejemplo en el formato de Hugging Face (`rajpurkar/squad_v2`):

```python
{
  "id": "56ddde6b9a695914005b9628",
  "context": "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. ...",
  "question": "In what country is Normandy located?",
  "answers": {"text": ["France", "France", "France", "France"], "answer_start": [159, 159, 159, 159]}
}
```

Observa dos cosas:

- **Varias referencias.** La validación trae varias anotaciones por pregunta (crowdworkers distintos). La evaluación EM/F1 toma la mejor coincidencia entre todas ellas.
- **Sin respuesta.** Si `answers["text"]` es una lista vacía, la pregunta no tiene respuesta. En nuestro proyecto esto se traduce a `answer: null`, `start: null`.

## 2. Qué encontramos en los datos

| Métrica | Train (SQuAD 2.0) | Validation (SQuAD 2.0) | Public sample (5 ítems) |
|---|---|---|---|
| Ejemplos | 130 319 | 11 873 | 5 |
| Sin respuesta | **33.4 %** | **50.1 %** | 2 de 5 (40 %) |
| Tokens de entrada (mediana) | 158 | 163 | — |
| Tokens de entrada (p95) | 292 | 304 | — |
| Ejemplos > 384 tokens | 1 373 (1.1 %) | 210 (1.8 %) | 1 de 5 (`sample-05`, 665 tokens) |
| Offsets inconsistentes | 0 de 86 821 | 0 de 20 302 | 0 de 3 respuestas |

### Hallazgos que cambian decisiones

**1. La distribución cambia entre splits.** El train tiene 33 % sin respuesta, la validación 50 % y el set oculto de la guía unas 25 %. Un umbral de "sin respuesta" calibrado en validación sesgaría al modelo a responder menos de lo que debería en el set oculto. Por eso el umbral debe calibrarse con una muestra que tenga proporciones parecidas a 25 %, o al menos corregir esa diferencia (lo veremos en la Fase 4).

**2. El baseline "siempre null" vale la proporción de preguntas sin respuesta.** Si siempre respondemos `null`, acertamos en todas las preguntas sin respuesta y fallamos en el resto. Con 25 % sin respuesta, el puntaje esperado es 0.25, que es justo el baseline provisional de la rúbrica. En validación (50 %) el mismo baseline daría 0.50, así que no es comparable con el set oculto. Los números del reporte deben explicarlo.

**3. Los contextos largos son raros, pero existen.** El 95 % de los ejemplos cabe en 292 tokens. Solo el 1 % supera 384, pero esos son justo los casos que la guía usa para probar el manejo de ventanas (`sample-05`: 665 tokens). Si no implementamos ventanas deslizantes, esos ítems fallan.

**4. Los offsets de los datos son consistentes.** Ninguna respuesta de train, validación ni del public sample tiene un texto distinto al que marca `answer_start`. Esto importa porque el servicio debe devolver `context[start:start+len(answer)] == answer`. Si los datos ya tuvieran errores, nuestro servicio heredaría esos errores.

## 3. Los candidatos a encoder

Un **encoder** lee el texto completo a la vez (atención bidireccional) y produce un vector por token. Para QA extractivo añadimos dos clasificadores lineales: uno predice la probabilidad de que cada token sea el inicio de la respuesta, y otro la probabilidad de que sea el final. El `[CLS]` (el primer token) sirve para representar "no hay respuesta".

Un **decoder** (estilo GPT) genera texto token a token. La guía lo prohíbe para la implementación: es más lento en CPU y su salida generada no encaja con la evaluación exacta. Queda para la investigación del reporte (sección 8).

Hay tres aspectos que pesan en la decisión: tamaño (latencia en laptop), calidad y disponibilidad del checkpoint base.

Medidos con `scripts/benchmark_models.py` (con el modelo QA completo, cabeza start/end incluida; latencia en CPU, batch 1, 384 tokens, mediana de 10 corridas; los datos completos están en `logs/model_benchmark.json`):

| Modelo | Hugging Face id | Arquitectura | Capas / hidden | Parámetros | Latencia CPU | Paper original |
|---|---|---|---|---|---|---|
| DistilBERT | `distilbert-base-uncased` | Encoder destilado de BERT | 6 / 768 | **66.4 M** | **110 ms** | [Sanh et al., 2019](https://arxiv.org/abs/1910.01108) |
| BERT-base | `bert-base-uncased` | Encoder | 12 / 768 | 108.9 M | 235 ms | [Devlin et al., 2019](https://arxiv.org/abs/1810.04805) |
| RoBERTa-base | `roberta-base` | Encoder (mismo esquema que BERT) | 12 / 768 | 124.1 M | 229 ms | [Liu et al., 2019](https://arxiv.org/abs/1907.11692) |
| DeBERTa-v3-base | `microsoft/deberta-v3-base` | Encoder con atención desentrelazada | 12 / 768 | 183.8 M | 387 ms | [He et al., 2021](https://arxiv.org/abs/2111.09543) (y [DeBERTa, 2020](https://arxiv.org/abs/2006.03654)) |
| ModernBERT-base | `answerdotai/ModernBERT-base` | Encoder moderno (RoPE, atención local/global) | 22 / 768 | 149.6 M | 317 ms | [Warner et al., 2024](https://arxiv.org/abs/2412.13663) |

Notas sobre la tabla:

- El conteo de DistilBERT (66.4 M) coincide con el ejemplo de `/info` de la guía (66 362 880).
- Las latencias son de forward puro en el laptop de desarrollo, sin tokenización ni red. En el servicio real se suma el tiempo de tokenizar y el túnel. Aun así, **los cinco caben holgadamente en el límite de 5 s**: la latencia no es lo que separa a los candidatos.

Los cinco son checkpoints *base* sin fine-tuning en SQuAD. Verificamos en el Hub que ninguno tenga la etiqueta `squad`. El checkpoint `distilbert/distilbert-base-cased-distilled-squad` **sí** está entrenado en SQuAD y queda prohibido por la regla 1.

## 4. Decisión: `distilbert-base-uncased`

**Qué elegimos:** `distilbert-base-uncased`.

**Por qué:**

- **Es la referencia de la rúbrica.** La guía dice que el 0.70 es "un DistilBERT fine-tuned por el instructor". Si usamos el mismo modelo, nuestra meta es clara: llegar a 0.70 da los 60 puntos completos.
- **Latencia.** El límite es 5 s por request y 150 s para 50 ítems. Con 6 capas, DistilBERT cabe holgado en CPU, incluso con el tiempo del túnel. BERT-base o DeBERTa-v3 arriesgan el límite.
- **Entrenamiento viable.** En la RTX 4050 (6 GB) entrena con batch 16–32 y fp16 sin problemas, en tiempo razonable.

**Trade-off:** pierde calidad frente a BERT-base o RoBERTa-base. Tomamos esa pérdida a cambio de latencia y tiempo. Si la Fase 3 muestra que nos sobra margen, podemos probar BERT-base y comparar con el mismo harness.

**Alternativa descartada:** `distilbert-base-cased` (sin minúsculas). Conserva mayúsculas, lo que podría ayudar con nombres propios. La dejamos como experimento opcional, porque el resultado de EM depende de cómo se normaliza el texto y necesita validación.

**Qué verificar en la Fase 2:** el tokenizer `uncased` normaliza a minúsculas, pero los `offset_mapping` deben seguir apuntando al texto original. Si no lo hacen, los `start` del servicio estarían mal. Lo probamos con tests.

## 5. Cómo reproducir

```bash
.venv\Scripts\python.exe scripts\explore_data.py
```

La primera ejecución descarga SQuAD 2.0 y el tokenizer de Hugging Face (unos cientos de MB). Las siguientes usan la caché.

## 6. Notas de entorno (Windows)

- Hugging Face avisa que la caché usa symlinks, que Windows no siempre permite sin el modo desarrollador. La caché funciona igual, solo ocupa más espacio. Para silenciar el aviso: `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
- Sin `HF_TOKEN` las descargas son más lentas y con límite de tasa. No es necesario para el proyecto.

## 7. Lecturas

1. [Rajpurkar et al., 2016. SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250). Versión 1.0 y la definición de la tarea.
2. [Rajpurkar et al., 2018. Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822). Versión 2.0, el origen de las preguntas sin respuesta.
3. [Devlin et al., 2019. BERT](https://arxiv.org/abs/1810.04805). Secciones 4.2–4.3 para el fine-tuning en SQuAD.
4. [Sanh et al., 2019. DistilBERT](https://arxiv.org/abs/1910.01108). El modelo que elegimos y cómo se destila.
5. [Vaswani et al., 2017. Attention Is All You Need](https://arxiv.org/abs/1706.03762). La arquitectura base de todos los candidatos.
