# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `team-XX` (pendiente: registrar con el instructor) |
| Members | Pendiente: nombres del equipo |
| Repository URL | Pendiente |
| `submission` commit hash | Pendiente: se llena al crear el tag |
| Public service URL used at the dry run | Pendiente: túnel de la Fase 6, aún sin elegir |
| AI tools used (which, and for what) | Claude Code (Claude Haiku 5.5) para escribir el servicio, las pruebas, los scripts de medición y este reporte. Todas las cifras salen de corridas nuestras, citadas en el repo. |

## 1. Model choice and justification

- **Base checkpoint:** `distilbert-base-uncased` (66.4 M parámetros, 6 capas, dimensión 768). Es el checkpoint base, sin fine-tuning en SQuAD. Verificamos en el Hub que no tenga la etiqueta `squad`.
- **Alternatives considered:** `bert-base-uncased` (108.9 M, 235 ms por pregunta en CPU), `roberta-base` (124.1 M, 229 ms), `deberta-v3-base` (183.8 M, 387 ms) y `ModernBERT-base` (149.6 M, 317 ms), medidos en la Fase 1 (`docs/fase1_datos_y_modelo.md`). Los descartamos: con 5 s por petición, el túnel y un tope global de 150 s para 50 ítems, un encoder de 12 capas deja poco margen. También descartamos `distilbert-base-uncased-distilled-squad` porque ya está entrenado en SQuAD y la guía lo prohíbe.
- **Trade-off:** perdemos calidad frente a los modelos de 12 capas. A cambio, cada pregunta corta responde en 55–70 ms en CPU y el modelo ocupa 265 MB.
- **Adaptación:** LoRA de rango 16 sobre las cuatro matrices de atención de cada bloque, con la cabeza de QA entrenada completa. Son 591 362 parámetros entrenables (0.88 %). Al terminar se fusionan los adaptadores, así que la API carga un modelo normal sin `peft`.

## 2. Training

| Item | Value |
|---|---|
| Training data | SQuAD 2.0 `train` completo: 130 319 preguntas (131 754 ventanas) |
| Validation data | `dev_select`: 5 917 preguntas de SQuAD 2.0 `validation`, separadas por artículo. `dev_report` (5 956) no se usó para ninguna cifra de este reporte. |
| Epochs | Tope de 8 con parada temprana (paciencia 2). Paró después de la época 5. El servicio sirve la época 5. |
| Learning rate and schedule | 2e-4, calendario lineal con 6 % de calentamiento |
| Batch size | 32 |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | RTX 4050 Laptop (6 GB). 6 529 s (≈ 1 h 49 min) hasta la época 5. |

- **Loss curve:** `logs/runs/r16_es/train_log.jsonl` (pérdida por paso). Todavía no hay imagen `logs/loss.png`; la generaremos antes de la entrega.
- **Validation results (`dev_select`, época 5, τ = 0):** EM 66.17, F1 68.78, HasAns F1 64.72, NoAns accuracy 72.57. Con el τ candidato de la sección 3 (−2), el puntaje con la escala de la guía, reponderado a 25 % sin respuesta, es 0.697. Es un proxy calculado sobre el mismo conjunto que usamos para elegir, así que está inflado.

## 3. No-answer threshold

- **Scores compared:** el puntaje nulo es `start[CLS] + end[CLS]`, tomado como el mínimo entre ventanas. El puntaje del span es el mejor par de inicio y fin válido (top-20 por lado, longitud máxima 30 tokens) entre todas las ventanas. Respondemos el span si `mejor_span − nulo > τ` (regla de BERT §4.3). Si no, devolvemos `null`.
- **How we chose the value:** barrimos τ sobre `dev_select` época por época (`scripts/sweep_tau.py`, `scripts/item_score.py`) y reponderamos a 25 % sin respuesta, la proporción esperada en el set oculto. Para la época 5 el mejor τ es −2 (puntaje 0.697), y con τ = −2 el puntaje se mantiene igual en las épocas finales, así que el valor es estable.
- **Final value:** **pendiente**. El código usa `QA_TAU=0.0` por defecto hasta que el equipo fije el valor. Hay que fijarlo antes de servir.
- **Too low or too high:** con τ muy bajo el modelo responde casi todo y la precisión en preguntas sin respuesta cae. Con τ muy alto devuelve `null` con demasiada frecuencia y pierde respuestas válidas (HasAns F1 baja).

## 4. Serving architecture

- **Framework and server:** FastAPI con uvicorn, **1 worker**. El modelo vive en un solo proceso.
- **Model loading:** en un hilo al arrancar (*lifespan*). Mientras carga, `/health` responde 503 `loading`. Si `QA_API_TOKEN` falta, el arranque falla.
- **Request flow:** primero se valida el token. Después se parsea el JSON y se valida contra el esquema del contrato (límites de longitud y campos extra). Luego `ValidationFeatures` parte la pregunta y el contexto en ventanas de 384 tokens con solapamiento de 128. Un solo pase del modelo sobre el lote de ventanas, en CPU y bajo un lock. `predict_example` aplica τ y recorta el texto del contexto con los offsets. Un `assert` garantiza `context[start:start+len] == answer`.
- **Access control:** `Authorization: Bearer` se compara con `hmac.compare_digest`. El esquema y el token distinguen mayúsculas. Sin token válido siempre hay 401, aunque el cuerpo esté roto. El valor viene de `QA_API_TOKEN`, nunca del código; `.env` está en `.gitignore`.
- **Tunnel:** pendiente de elegir y de instalar (Fase 6).
- **How to run** from a clean machine:

```bash
pip install -r requirements.txt
# Copia la carpeta models/ (no está en git) o apunta QA_MODEL_DIR a ella.
export QA_API_TOKEN=...            # PowerShell: $env:QA_API_TOKEN="..."
export QA_MODEL_DIR=models/r16_es/merged_epoch5
export QA_TAU=...                  # pendiente de fijar con dev_select
uvicorn service.app:app --host 0.0.0.0 --port 8000 --workers 1
```

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | `16` | `16` | 383 | yes |
| sample-02 | `3.5 days` | `3.5 days` | 437 | yes |
| sample-03 | `null` | `null` | null | yes |
| sample-04 | `null` | `3` | 310 | no |
| sample-05 | `50,257` | `50,257` | 3157 | yes |

sample-04 falla en un caso cercano: el contexto dice "3 times" sobre GPT-3, pero la pregunta pregunta por BERT-base. El modelo empareja el tipo de cantidad y no ve que el sujeto cambió. Cinco ítems no dan significancia estadística. Esta cifra solo sirve como prueba de sanidad.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66 364 418 (modelo fusionado; `/info` lo reporta) |
| Size on disk | 265 MB (`model.safetensors`, fp32) |
| Mean latency per `/predict` request on our laptop (CPU) | 55–70 ms para contextos de ≈ 450 caracteres; 466 ms para el contexto de 3 171 caracteres (varias ventanas). Medido con `scripts/bench_predict.py`, con otra sesión de entrenamiento corriendo en el mismo equipo. |
| HTTP smoke test (localhost, sin túnel) | 50 peticiones secuenciales en 7.3 s, la más lenta 488 ms, 0 fallas (`scripts/smoke_test.py`) |
| Laptop CPU and RAM | Intel Core i7-12650H (10 núcleos, 16 hilos), 16 GB de RAM |

## 7. Problems encountered and how you fixed them

1. **Problem:** la parada temprana con τ = 0 no era confiable. El HasAns F1 de la corrida r16 oscilaba entre 53 y 70 entre épocas. **Cause:** el umbral τ = 0 se desplaza en cada época, así que el puntaje cambia por el umbral y no por el modelo. **Fix:** elegir con el puntaje de la escala de la guía reponderado a 25 % sin respuesta, y barrer τ por época. **Commit:** `9fb7163` (parada temprana y selección) y `909035a` (puntaje por ítem).
2. **Problem:** `predict_example` devuelve un margen para los spans y el logit nulo crudo para `null`, pero el contrato exige `score` entre 0 y 1. **Cause:** las dos ramas usan escalas distintas. **Fix:** mapear con sigmoide en `service/qa.py`. Sigue sin ser un valor calibrado. **Commit:** `498992e`.
3. **Problem:** en un worktree sin `models/`, las pruebas con el modelo real se saltaban en silencio. **Cause:** los pesos no están en git. **Fix:** `QA_MODEL_DIR` y un motivo visible en el `skip`. **Commit:** `0a391e2`.

## 8. Decoder investigation

Dos problemas nuestros, medidos en el set público, motivan la pregunta:

- **sample-04:** el modelo extractivo devuelve un span para una pregunta sin respuesta. Un decoder entrenado para responder `null` con texto podría decir "no answer" directamente, pero la evaluación exige una coincidencia exacta del span. Un decoder que parafrasea rompe la regla `context[start:start+len] == answer`, así que tendríamos que validar cada salida contra el contexto.
- **Preguntas largas:** nuestro contexto de 3 171 caracteres usa varias ventanas. Un decoder con contexto largo tendría costo de atención mayor en CPU.

El costo de un decoder en CPU y su latencia real no están medidos todavía. Eso queda para la Fase 8.

## 9. Known limitations

- **Threshold:** τ no está fijado. El valor por defecto `0.0` no es el valor calibrado.
- **Score:** no está calibrado. Para `null` es el logit nulo pasado por una sigmoide, así que no es comparable con el de un span.
- **Concurrency:** un solo worker y un lock. Las peticiones se encolan; con un contexto de 5 000 caracteres la latencia sigue muy por debajo de 5 s, pero una cola larga sí puede pasar el límite.
- **Input:** solo inglés. No validamos idioma, y el texto fuera de la distribución de SQuAD no lo medimos.
- **Auth:** un único token compartido, sin rotación ni límite de peticiones.
- **Evaluation:** `dev_report` no se evaluó en este reporte. Esa evaluación única la decide el equipo.
