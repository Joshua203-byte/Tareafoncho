# Fase 2: preprocesamiento

Este documento explica las decisiones de preprocesamiento de `src/features.py` y las respalda con mediciones de `scripts/measure_preprocessing.py` (`logs/preprocessing_stats.json`).

## 0. Glosario: ejemplo, ventana, fila, padding

- **Ejemplo**: una pregunta con su contexto y sus respuestas, tal como viene en SQuAD. Una pregunta, un ejemplo.
- **Ventana**: un trozo del contexto, tokenizado junto con la pregunta, con un tamaño máximo (`max_length`). Si el contexto no cabe, se parte en varias ventanas.
- **Fila**: una ventana en la salida del tokenizer. Es lo que el modelo procesa en una pasada.
- **Padding**: tokens `[PAD]` (ID 0) que se agregan al final de una fila para igualar la longitud de las demás del lote. La máscara de atención los marca con 0.

**Una pregunta puede generar varias filas.** Por eso `overflow_to_sample_mapping` existe: dice a qué pregunta pertenece cada fila.

### Ejemplo paso a paso

Pregunta: `Where is the Lempa River?`
Contexto: `The Lempa River is the longest river in El Salvador. It flows from Guatemala to the Pacific Ocean.`
Con `max_length = 24` y `stride = 8` (en el proyecto son 384 y 128, para que el corte se vea).

El tokenizer produce 3 filas:

```
fila 0: [CLS] where is the lempa river ? [SEP] the lempa river is the longest river in el salvador . [SEP]
fila 1: [CLS] where is the lempa river ? [SEP] is the longest river in el salvador . it flows from guatemala to [SEP]
fila 2: [CLS] where is the lempa river ? [SEP] el salvador . it flows from guatemala to the pacific ocean . [SEP]
```

- La fila 1 **retrocede** para empezar antes del final de la fila 0. Ese retroceso es el *stride*.
- `is the longest river in el salvador .` aparece en las filas 0 y 1. Si la respuesta cae en esa zona compartida, aparece completa en alguna ventana.
- `overflow_to_sample_mapping = [0, 0, 0]`: las tres filas son de la pregunta 0.

Con dos preguntas en el mismo lote (`sample-01` y `sample-05`):

```
filas: 4
overflow_to_sample_mapping: [0, 1, 1, 1]
```

- Pregunta 0 (`sample-01`): 1 fila.
- Pregunta 1 (`sample-05`): 3 filas, que son tres ventanas de la **misma** pregunta.

### Ejemplo de padding

Dos filas en un lote: una de 16 tokens y otra de 24. El lote necesita una tabla rectangular, así que la corta se rellena hasta 24:

```
tokens:  [CLS] where is the river ? [SEP] the le ##mp ##a river is long . [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
mascara:   1     1     1  1   1    1  1    1   1   1   1  1    1  1    1  1   1   0     0     0     0     0     0     0     0
```

La máscara oculta el relleno en la atención, así que no cambia la respuesta. Pero el cómputo se hace sobre toda la tabla: las columnas `[PAD]` se procesan igual.

**Por qué esto importa.** Con padding fijo a 384, una fila de 158 tokens procesa 226 columnas de relleno. Medido sobre el train, solo el **44.4 %** de las columnas con padding fijo son tokens reales. Con padding dinámico, cada lote se rellena solo hasta su fila más larga.

## 1. Estructura y diseño

Entrenamiento y validación comparten la tokenización. Solo cambia lo que se hace con cada ventana:

```
BaseQAFeatures          tokenize(): pregunta + contexto -> ventanas (sin padding)
│                       __call__(): llama a tokenize() y luego a build()
├── TrainFeatures       build(): añade start_positions / end_positions
└── ValidationFeatures  build(): añade example_id y offset_mapping (None fuera del contexto)
```

**Por qué esta forma.** Es un patrón de *template method*: la base define el esqueleto del proceso y cada subclase implementa el paso que cambia.

- **Un solo lugar para la tokenización.** Si entrenamiento y validación tokenizaran por separado, un cambio en `max_length` o en el stride podría aplicarse a uno y no al otro. El error sería silencioso: las etiquetas no coincidirían con las ventanas de validación.
- **Herencia y no un flag.** Con dos variantes, una base con `build` es más legible que una función con `if train:`. Cada nueva variante agregaría otra rama a la función. Si algún día hubiera muchas variantes que cambian pasos distintos, convendría componer objetos pequeños; con dos, sería sobre-ingeniería.
- **Lo que la base no delega.** La subclase no puede cambiar cómo se tokeniza, solo qué se hace con el resultado. Así el contrato entre preprocesamiento y post-procesamiento (Fase 4) queda en un solo lugar.

## 2. Parámetros: `max_length = 384`, `stride = 128`

**Qué mide cada uno.** `max_length` es el número máximo de tokens por ventana, pregunta y especiales incluidos. `stride` es cuántos tokens se repiten entre ventanas consecutivas.

**Por qué 384.** Es el valor que usa el curso oficial de Hugging Face en su capítulo de *Question answering* (`max_length = 384`, `stride = 128`, [fuente](https://huggingface.co/learn/llm-course/chapter7/7)). Es también lo que la guía pide reportar en `/info`. El paper de BERT **no** menciona ventanas ni estos valores: en SQuAD 2.0 solo describe el `[CLS]` para preguntas sin respuesta (sección 4.3). Con este valor, según nuestros datos:

| Longitud de entrada (pregunta + contexto) | Valor |
|---|---|
| Mediana | 158 tokens |
| Percentil 95 | 292 tokens |
| Ejemplos que superan 384 | 1 373 (1.05 %) |

Es decir, 384 cubre el 98.95 % de los ejemplos sin ventanas extra. Subir a 512 (el máximo de DistilBERT) cubriría casi todo lo restante, pero el cómputo sube y la ganancia es de 0.9 %. Ver la decisión abierta más abajo.

**Por qué 128.** Es un tercio de la ventana. Una respuesta de hasta 128 tokens que caiga en el corte aparece entera en al menos una ventana. Las respuestas de SQuAD son cortas (mediana de 14 caracteres), así que este margen es amplio.

**Aviso sobre la comparación con papers.** El valor 384 es una convención, no un resultado que medimos para elegirlo. Usarlo nos permite comparar nuestra configuración con implementaciones de referencia. No garantiza que coincida con la configuración del modelo de referencia de la rúbrica, que no conocemos.

## 3. Padding dinámico

No rellenamos a 384 durante el preprocesamiento. El padding se hace por lote, en el collator (`make_collator`).

**Qué es el padding.** El modelo recibe tensores rectangulares: todas las secuencias de un lote deben tener la misma longitud. Si una ventana mide 158 tokens y otra 384, la corta se rellena con tokens `[PAD]` (id 0 en DistilBERT) hasta 384. La máscara de atención (`attention_mask`) marca esos tokens con 0 para que no influyan en el resultado, pero el modelo **igual los procesa**: las multiplicaciones de matrices se hacen sobre toda la longitud, relleno incluido. Por eso el relleno cuesta cómputo aunque no cambie la respuesta.

**Padding fijo vs dinámico.** Con padding fijo, cada ventana se rellena a 384. Con padding dinámico, cada lote se rellena solo hasta su ventana más larga.

**Fuente que dice lo contrario.** El mismo capítulo del curso de Hugging Face escribe que "there is no real benefit to applying dynamic padding here" porque "most of the contexts will be long". Nuestra medición contradice esa premisa para SQuAD 2.0 de entrenamiento: la mediana de la entrada es de 158 tokens, no de 384. Confiamos en la medición, pero conviene repetirla si cambia el dataset.

**Medición sobre el train completo:**

| Métrica | Valor |
|---|---|
| Ventanas | 131 754 |
| Tokens reales / tokens con padding fijo a 384 | **44.4 %** |
| Reducción de cómputo aproximada | **~2.25×** |

Con padding fijo, más del 55 % del cómputo iría a tokens de relleno. El padding dinámico evita eso y solo cuesta un collator, que es código estándar.

**Por qué no es 2.4×.** Antes estimé 2.4× comparando la mediana (158) contra 384. Esa es una comparación de la mediana, y la medición correcta suma todas las ventanas. El número medido es el que usamos.

## 4. Tiempo de preprocesamiento

**17.3 s** para las 130 319 preguntas del train, con caché desactivada (`load_from_cache_file=False`). La medición con caché activa da 0.1 s, porque no recalcula nada; no es el número que sirve para el reporte.

Esto hace innecesario el subconjunto del entrenamiento desde el punto de vista del preprocesamiento. La decisión de entrenar con todo o con un subconjunto se toma en la Fase 3, por tiempo de GPU, no por el preprocesamiento.

## 5. Respuestas a mitad de token: opción B

Una respuesta puede empezar o terminar dentro de una subpalabra. Ejemplo: `Tegucig` termina dentro del token `##gal` de `Tegucigalpa`.

**Medición** (respuestas answerable que caben completas en una ventana, 86 821):

| Caso | Cantidad | Porcentaje |
|---|---|---|
| Termina a mitad de token | 146 | 0.17 % |
| Empieza a mitad de token | 76 | 0.09 % |

**Opciones consideradas:**

- **A. Etiquetar el token completo.** La respuesta recuperada sería más larga que la referencia, y EM fallaría.
- **B. Marcar la ventana como `[CLS]`.** Pierde un ejemplo de entrenamiento, pero nunca enseña una etiqueta incorrecta.

**Decisión: B.** Son menos del 0.3 % de las respuestas, así que el costo de perderlas es mínimo. Con A, en cambio, cada caso enseñaría un span que no coincide con la referencia.

**Respaldo en el paper de BERT.** La sección 4.3 usa `[CLS]` como respuesta nula: "treat questions that do not have an answer as having an answer span with start and end at the [CLS] token". Usamos la misma convención para ventanas que no contienen la respuesta completa.

**Limitación que aceptamos.** Con B, una ventana que sí tiene la respuesta, pero partida, se enseña como "sin respuesta". Eso empuja al modelo hacia `null` en un caso que no lo es. Afecta a 222 casos del train; la alternativa de descartar el ejemplo entero evita ese sesgo, pero cambia la distribución de entrenamiento. Decidimos B por ser lo que ya implementamos y probamos, y lo medimos en la Fase 4 con el dev set, que no se modifica.

**Validación nunca se filtra.** Los casos a mitad de token se conservan en validación, para que el dev set mida el costo real de esa limitación. Filtrar casos difíciles de la evaluación inflaría las métricas que reportamos.

## 6. Cobertura: ninguna respuesta se pierde

**0 de 86 821** respuestas answerable quedan sin una ventana que las contenga completa. Con stride 128 y respuestas cortas, esto era esperable, pero lo medimos para no asumirlo.

## 7. Validación: offsets

Para cada ventana de validación, `offset_mapping` conserva el offset solo de los tokens del contexto (`sequence_id == 1`). Los tokens de la pregunta y los especiales quedan en `None`. La post-procesamiento de la Fase 4 saltará los `None`, así que nunca puede cortar texto de la pregunta.

## 8. Pruebas

`tests/test_features.py`, 8 pruebas con el tokenizer real:

- La etiqueta decodifica exactamente a la respuesta.
- Pregunta sin respuesta: todas las ventanas en `[CLS]`.
- Contexto largo (`sample-05`): solo la ventana 2 tiene etiqueta.
- Respuesta a mitad de token: `[CLS]` (opción B), con control de que el tokenizer efectivamente parte la palabra.
- Respuesta sobre palabra completa: se etiqueta (control).
- Validación: offsets en `None` fuera del contexto; `example_id` por ventana.
- Padding dinámico: el lote queda a la longitud de la ventana más larga, no a 384.

## 9. Decisiones tomadas

1. **`max_length = 384`.** Decidido por el equipo. Cubre el 98.95 % de los ejemplos sin ventanas extra, y es la configuración de referencia del curso de Hugging Face.
2. **Opción B: respuestas a mitad de token como `[CLS]`.** Decidido por el equipo. Ver la sección 5 para la limitación que aceptamos y cómo se mide.
3. **`num_workers = 0` con `pin_memory = True`.** Se configurará así en la Fase 3. Con el preprocesamiento ya precalculado, el collator es barato, y en Windows los workers añaden costo de arranque. Si la GPU queda ociosa, medimos antes de cambiarlo.
