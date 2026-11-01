# Plan de implementación — Extractive QA as a Service

> Documento de trabajo (no es el `REPORT.md`). Cada fase termina con un **entregable verificable** y una nota de **qué punto de la rúbrica cubre**. En cada fase primero se explica el concepto, luego se implementa juntos.

## 1. El problema en una página

**Qué piden:** hacer fine-tuning de un encoder Transformer *base* (no decoders, no checkpoints ya entrenados en SQuAD) para QA extractivo estilo SQuAD 2.0, y servirlo como API web con token, expuesta por un túnel público desde un laptop con CPU.

**Cómo se evalúa (100 pts, luego × 0.5 si falta REPORT.md):**

| Componente | Pts | Alcance | Qué lo mueve |
|---|---|---|---|
| Evaluación en vivo | 60 | Equipo | 50 preguntas ocultas, ≈25 % sin respuesta. EM/F1 + juez LLM. Mapeo lineal entre baseline (0.25, siempre `null`) y referencia (0.70, DistilBERT) |
| Pitch | 20 | Equipo | 3 min: modelo, entrenamiento, serving, investigación de decoders (4 criterios × 0–5) |
| Q&A | 20 | **Individual** | Una pregunta al azar sobre el código, sin IA, sin ayuda |

**Lo que hace fallar un ítem (score 0):** timeout > 5 s, respuesta no-200, JSON mal formado, `id` distinto, `context[start:start+len(answer)] != answer`, `null` incorrecto. Presupuesto global: 50 ítems en 150 s (≈3 s/ítem en promedio, incluyendo latencia del túnel).

**Reglas duras (revisadas automáticamente):** base checkpoint; sin APIs comerciales; token solo vía `QA_API_TOKEN`; historial git sin squash; pesos fuera de git; tag `submission`; REPORT.md en ese commit.

**Los 3 retos técnicos reales (el resto es plomería):**
1. **Preprocesamiento de QA** — mapear respuesta en caracteres → posiciones de tokens (y viceversa), con ventanas deslizantes para contextos largos.
2. **Decidir "no hay respuesta"** — comparar score del `[CLS]` vs. mejor span y calibrar un umbral con validación.
3. **Contrato de API exacto** — offsets de caracteres correctos, errores con formato específico, 401 vs 400/422.

**Estado del entorno detectado:** Windows 11, Python 3.12.7, GPU RTX 4050 Laptop (6 GB VRAM), `torch` y `fastapi`/`uvicorn` ya instalados; **falta** `transformers`, `datasets`, `accelerate`. Ramas `main` y `emi` existen; ya hay 2 commits (historial se conserva).

---

## 2. Decisión de arquitectura (se discute en Fase 1)

Recomendación preliminar: **`distilbert-base-uncased`** como primera línea (66M params, rápido en CPU, es la referencia del instructor → llegar a 0.70 da los 60 pts completos), y **`roberta-base`/`deberta-v3-base`** solo si sobra tiempo y la latencia en CPU se mantiene < ~2 s. Con 6 GB de VRAM, DistilBERT con fp16 y batch 16–32 entrena cómodo; DeBERTa-v3-base entra pero es lento y más pesado en CPU.

Stack: `transformers` + `datasets` + PyTorch para entrenar; **FastAPI + uvicorn (1 worker)** para servir; `cloudflared`/`ngrok` para el túnel.

---

## 3. Fases

### Fase 0 — Entorno, repo y reglas (≈ medio día)
**Aprenderás:** qué es un entorno virtual y por qué `requirements.txt`; por qué el token nunca va en el código; qué hace `.gitignore`; por qué importa el historial de git aquí.

- Crear venv, instalar `transformers`, `datasets`, `accelerate`, `evaluate`, `pytest`, `httpx`; verificar `torch.cuda.is_available()`.
- Estructura del repo: `src/` (training), `service/` (API), `tests/`, `logs/`, `models/` (ignorado).
- `.gitignore`: `models/`, `*.safetensors`, `*.bin`, `.env`, `.venv/`, `__pycache__/`.
- `.env.example` (sin token real) y convención de commits pequeños y frecuentes (el historial se audita y alimenta la sección "Problemas" del reporte, que pide hash de commit).
- Decidir con el equipo: nombre `team-XX`, quién corre el servidor el día de la presentación (el laptop con túnel).

**Hecho cuando:** `python -c "import torch, transformers"` funciona, GPU detectada, primer commit de estructura. 
**Rúbrica:** reglas V.4 y V.5 (token fuera de git, historial preservado).

### Fase 1 — Entender los datos y elegir modelo (≈ 1 día)
**Aprenderás:** qué es QA extractivo; anatomía de SQuAD 2.0 (`context`, `question`, `answers.text`, `answer_start`, `is_impossible`); qué es un encoder, un tokenizer, subwords/WordPiece; por qué un encoder predice "inicio" y "fin" (dos clasificadores sobre cada token); por qué decoders no encajan (EM exacto, latencia CPU).

- Explorar SQuAD 2.0 (proporción sin respuesta, largos de contexto, cuántos exceden 384 tokens).
- Leer juntos `public_sample.json` (formato, `answers[].start`, sample-05 > 512 tokens).
- Elegir modelo con justificación **específica** (tradeoff tamaño/CPU/calidad) → alimenta pitch criterio 1 (hasta 5 pts) y REPORT §1 (se piden ≥ 1 alternativa descartada).
- Medir un baseline trivial (siempre `null` ⇒ ~0.25 esperado) para entender el ancla de la rúbrica.

**Hecho cuando:** notebook/script de exploración + decisión de modelo escrita en 5–10 líneas.
**Rúbrica:** Pitch (modelo), REPORT §1.

### Fase 2 — Preprocesamiento: tokenización y ventanas (≈ 1 día) ⭐ concepto clave
**Aprenderás:** `offset_mapping` (tokens ↔ caracteres); sliding window con `max_seq_length` y `doc_stride`; por qué una respuesta puede quedar fuera de una ventana y cómo se etiqueta como "no answer" (apunta a `[CLS]`, posición 0); padding vs truncation (`only_second`); por qué `overflow_to_sample_mapping` existe.

- Implementar `prepare_train_features` y `prepare_validation_features` (con tests unitarios sobre ejemplos pequeños: verificar que decodificar los tokens de start/end devuelve el texto de la respuesta).
- Elegir `max_seq_length=384`, `doc_stride=128` (valores estándar; se justifican y se exponen en `/info`).
- Decidir subset de entrenamiento (completo ≈ 130k vs. subsample si el tiempo apremia).

**Hecho cuando:** tests pasan; muestra visual de un ejemplo tokenizado con sus etiquetas.
**Rúbrica:** Pitch (training approach: "key decisions"), REPORT §2 (`max_seq_length`/`doc_stride`).

### Fase 3 — Fine-tuning y baseline (≈ 1.5 días)
**Aprenderás:** qué ajusta realmente el fine-tuning (cabeza QA + pesos del encoder); loss = cross-entropy sobre start y end; learning rate, warmup, scheduler lineal, epochs, batch size, fp16; por qué se guardan logs y qué dice una loss curve (sobreajuste, divergencia).

- Entrenamiento con `Trainer` (o loop propio — decidimos juntos cuál enseña más; `Trainer` es más rápido de escribir, loop propio más fácil de explicar en el Q&A).
- Primero una corrida corta (subset 5–10k, 1 epoch) para validar el pipeline; luego la corrida real (2–3 epochs).
- Guardar `logs/` (JSON/CSV de loss por paso) y `logs/loss.png`; checkpoints en `models/` (fuera de git).
- Reservar un **conjunto de validación propio** (dev de SQuAD 2.0, ≈ 12k) — nunca entrenar con él.

**Hecho cuando:** modelo guardado, curva de loss generada, EM/F1 de dev calculados (objetivo orientativo DistilBERT: ≈ 65–70 EM/F1 con umbral bien calibrado).
**Rúbrica:** Entregable 2 (logs y curvas), REPORT §2, regla V.1.

### Fase 4 — Post-procesamiento y umbral de "sin respuesta" (≈ 1 día) ⭐ concepto clave
**Aprenderás:** logits start/end → mejores candidatos (top-k × top-k, `max_answer_len`); score de "null" = `start_logit[CLS] + end_logit[CLS]`; regla `null_score − best_span_score > τ`; cómo combinar ventanas de un contexto largo; cómo elegir τ barriendo valores sobre validación (maximizar F1); qué pasa con τ muy bajo/alto (responde siempre ↔ nunca; ver baseline 0.25).

- Implementar `postprocess` que devuelve `(answer, start_char, score)` garantizando `context[start:start+len]==answer` (aserción defensiva).
- Barrido de τ, gráfica F1 vs. τ, valor final guardado en config (REPORT §3, pide el razonamiento).
- Ojo con la distribución: el set oculto tiene ≈ 25 % sin respuesta (SQuAD train tiene ≈ 33 %) y textos **no** de SQuAD. Consideramos calibrar τ ponderando para ~25 % y ser algo conservadores.

**Hecho cuando:** función `predict(context, question)` pura y probada sobre `public_sample.json` (≥ 4/5 correctos, incluyendo ambos `null` y el contexto largo).
**Rúbrica:** Live evaluation (el 60 % de la nota se decide aquí), REPORT §3 y §5.

### Fase 5 — API según el contrato (≈ 1 día)
**Aprenderás:** qué es un contrato OpenAPI y por qué se valida automáticamente; HTTP status codes; bearer tokens y comparación en tiempo constante (`hmac.compare_digest`); carga del modelo en el *startup* (lifespan) y `/health` 503→200; validación con Pydantic y por qué FastAPI devuelve 422 y cómo reformatear al `{"error": ...}`; por qué **1 solo worker**; inferencia síncrona vs. async (el modelo bloquea CPU).

- `/health` público (503 `loading` hasta cargar el modelo), `/info` y `/predict` protegidos.
- Handlers de excepción para que **todo** error tenga forma `{"error": "<code>", "detail": ...}` (401 con `WWW-Authenticate: Bearer`; 400/422 `bad_request`; 500 `internal_error`).
- `QA_API_TOKEN` leído del entorno; si falta, el servicio falla al arrancar (mejor que dejarlo abierto).
- `/info`: `team`, `model` (id HF **base**, p. ej. `distilbert-base-uncased`), `params` real (`sum(p.numel())`), `max_seq_length`, `doc_stride`.
- Tests con `pytest` + `TestClient`: los 5 tests de la guía + id echo, contexto 5000 chars, campos faltantes, JSON roto, token con distinto case.
- Medir latencia por request en CPU (objetivo ≪ 3 s para entrar en el cap de 150 s con túnel).

**Hecho cuando:** los 5 `curl.exe` de la guía dan lo esperado en `localhost:8000` y los tests pasan.
**Rúbrica:** Live evaluation (fallos = 0 pts por ítem), control de acceso, Pitch (serving), REPORT §4 y §6.

### Fase 6 — Empaquetado, túnel y simulacro (≈ 1 día)
**Aprenderás:** qué es un túnel inverso (ngrok/Cloudflare/Tailscale) y por qué la URL es pública (y por eso el token); cómo probar "desde otra red"; configuración de energía en Windows; Docker vs. venv (la guía acepta cualquiera).

- `requirements.txt` fijado, `README` con "cómo correr desde cero", script de arranque (`run.ps1`) que setea `QA_API_TOKEN` en la sesión.
- Túnel principal + uno de respaldo instalado y probado (la guía lo pide). Probar los 5 curl contra la URL pública **desde datos móviles**.
- Script `scripts/smoke_test.py` que simula el harness: 50 requests secuenciales con timeout 5 s y cronómetro total (< 150 s).
- Pesos: publicar como release o carpeta compartida; documentar dónde bajarlos.

**Hecho cuando:** smoke test completo pasa por la URL del túnel desde otra red.
**Rúbrica:** Dry run (día 8), regla IX, Entregable 3.

### Fase 7 — Reporte, entrega y dry run (≈ medio día)
**Aprenderás:** cómo documentar decisiones técnicas con evidencia; qué significa el tag de git como "snapshot" de entrega.

- Llenar `REPORT.md` con **todas** las secciones de `REPORT_TEMPLATE.md` (cualquier sección vacía ⇒ nota × 0.5; es la trampa más cara del proyecto). Mantener ≲ 1000 palabras, específico del equipo.
- Sección 7 requiere ≥ 2 problemas reales **con hash de commit** → anotarlos durante el proyecto, no al final.
- Verificar que no hay token en el historial (`git log -S` / grep) y que los pesos no están versionados.
- `git tag submission`, push del tag, compartir URL (o invitar al instructor si es privado). Enviar URL del túnel antes del slot del dry run.

**Hecho cuando:** `git show submission:REPORT.md` muestra el reporte completo; dry run pasado.
**Rúbrica:** Condición del reporte (×1 vs ×0.5), regla V.4/V.5, Entregables 1–4.

### Fase 8 — Investigación de decoders (≈ medio día; puede ir en paralelo desde la Fase 4)
**Aprenderás:** qué cambia con un decoder (GPT) o encoder-decoder (T5/BART): generación vs. extracción, respuestas no contiguas, sí/no, contextos largos, costo en latencia/tamaño/evaluación (EM exacto), LoRA para fine-tuning barato.

- Basarlo en **errores reales** de nuestro modelo (sample set y dev set): ¿qué preguntas falló y por qué? Esto es lo que da 5/5 en el criterio ("atado a problemas específicos del equipo", no comparación genérica).
- Lecturas: GPT-1 (Radford 2018), LoRA (Hu 2021), BERT §4.2–4.3, Rajpurkar 2018.

**Hecho cuando:** REPORT §8 escrito con ≥ 2 problemas propios → alternativa → costo.
**Rúbrica:** Pitch (criterio 4), REPORT §8.

### Fase 9 — Pitch y preparación para el Q&A (≈ 1 día antes de presentar)
**Aprenderás:** a explicar el sistema completo de punta a punta sin mirar notas; qué preguntas son probables sobre *tu* código.

- Guion de 3 min cubriendo los 4 criterios (modelo + tradeoff, training con decisiones y evidencia, serving: framework/carga/flujo/token, decoders ligado a problemas propios).
- **Q&A es individual (20 pts) y sin IA:** cada integrante debe poder explicar con su propio código: ¿cómo se obtiene `start` en caracteres?, ¿cómo se decide `null`?, ¿qué hace `doc_stride`?, ¿dónde se valida el token?, ¿qué pasa si el contexto > 384 tokens?, ¿qué pasa si falta un campo? Haremos un banco de preguntas y simulacros.
- Día de presentación: setear el token nuevo en `QA_API_TOKEN`, reiniciar servicio, verificar `/health`, laptop enchufada y sin suspensión, túnel arriba, URL enviada al instructor.

**Rúbrica:** Pitch (20) y Q&A (20).

---

## 4. Matriz de cobertura (rúbrica → fase)

| Requisito de la guía | Fase |
|---|---|
| Base checkpoint, sin SQuAD-finetuned, sin APIs comerciales | 1, 3 |
| Logs y loss curves | 3 |
| Contrato exacto + errores + access control | 5 |
| Token solo por env var, nunca en git | 0, 5, 7 |
| Historial git preservado, tag `submission`, pesos fuera | 0, 7 |
| Túnel público + dry run por túnel | 6, 7 |
| `/info` con team/model/params | 5 |
| Contextos > 512 tokens (sliding window) | 2, 4 |
| Umbral no-answer justificado | 4 |
| REPORT.md completo (evita × 0.5) | 7 (+ notas desde Fase 1) |
| Decoder investigation | 8 |
| Live eval 60 pts | 3, 4, 5, 6 |
| Pitch 20 pts | 1, 3, 5, 8, 9 |
| Q&A individual 20 pts | 9 (y entender cada fase) |

## 5. Riesgos y mitigaciones

- **Latencia por túnel + 150 s global:** modelo pequeño, 1 worker, modelo precargado, smoke test cronometrado.
- **Offsets incorrectos ⇒ ítem = 0:** aserción `context[start:start+len]==answer` en el servicio; si falla, devolver `null` en lugar de un span inválido.
- **Distribución distinta (textos fuera de SQuAD):** no sobreajustar τ a dev; probar con preguntas propias.
- **Laptop/túnel caído el día D:** túnel de respaldo + "last resort" (harness local con `localhost`).
- **Reporte olvidado:** empezar `REPORT.md` en Fase 3 e ir llenándolo.

## 6. Orden sugerido (1.5 semanas)

| Días | Fases |
|---|---|
| 1–2 | 0, 1, 2 |
| 3–5 | 3, 4 (+ 8 en paralelo) |
| 6–7 | 5, 6 |
| 8 | 7 (dry run + tag `submission`) |
| 9–10 | 9 (presentación) |
