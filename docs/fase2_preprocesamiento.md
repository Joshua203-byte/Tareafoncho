# Fase 2: preprocesamiento

Este documento explica las decisiones de preprocesamiento de `src/features.py` y las respalda con mediciones de `scripts/measure_preprocessing.py` (`logs/preprocessing_stats.json`).

## 1. Estructura

Entrenamiento y validación comparten la tokenización. Solo cambia lo que se hace con cada ventana:

```
BaseQAFeatures          tokenize(): pregunta + contexto -> ventanas (sin padding)
├── TrainFeatures       build(): añade start_positions / end_positions
└── ValidationFeatures  build(): añade example_id y offset_mapping (None fuera del contexto)
```

Así la lógica de tokenización no puede divergir entre los dos flujos. Esto es un patrón de *template method*: la base define el esqueleto y las subclases solo implementan el paso que cambia.

## 2. Parámetros: `max_length = 384`, `stride = 128`

**Qué mide cada uno.** `max_length` es el número máximo de tokens por ventana, pregunta y especiales incluidos. `stride` es cuántos tokens se repiten entre ventanas consecutivas.

**Por qué 384.** Es el valor estándar de los ejemplos de QA de Hugging Face y lo que la guía pide reportar en `/info`. Con este valor, según nuestros datos:

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

Una alternativa que no evaluamos a fondo es descartar esos ejemplos del entrenamiento en lugar de etiquetarlos como `[CLS]`. Técnicamente es más limpio, porque `[CLS]` significa "no está en esta ventana" y en estos casos la respuesta sí está. Con 222 casos el efecto probablemente es pequeño, y dejamos B por ser lo que ya implementamos y probamos.

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

## 9. Decisiones abiertas para el equipo

1. **`max_length` 384 o 512.** Con 512 cubrimos casi todo sin ventanas extra, a costa de más cómputo. Mi recomendación es 384 por ser el estándar y tener la medición.
2. **Opción B o descartar los 222 casos.** Probablemente no cambia el resultado final. Lo dejamos como B a menos que la Fase 3 muestre lo contrario.
