# Fase 4: umbral de no respuesta τ y evaluación final

## 1. Qué decide τ

El modelo predice un span dentro del contexto y un puntaje para `[CLS]`, que representa "no hay respuesta". Con `null_score = start[CLS] + end[CLS]` y `best_span_score`, la decisión es:

- responder el span si `best_span_score > null_score + τ`;
- responder `null` en otro caso.

τ = 0 significa que el modelo responde sólo cuando el span gana al `[CLS]`. Un τ más negativo responde más seguido; uno más positivo, menos. Ver `src/postprocess.py`.

## 2. Cómo se elige τ

No se elige con F1 sobre `dev_select` a secas. La muestra de `dev_select` tiene 48 % de preguntas sin respuesta, y el conjunto oculto tiene cerca de 25 %. Si optimizamos sobre `dev_select` tal cual, el τ resultante favorece las preguntas sin respuesta más de lo que conviene en el oculto.

Por eso calibramos sobre la **escala de la nota en vivo** (guía, sección XIV.1), reponderada a 25 % sin respuesta:

- Cada pregunta vale 1 si la respuesta es exacta, o si la pregunta no tiene respuesta y el modelo responde `null`.
- Vale 0.5 si hay solapamiento parcial. Como no tenemos el juez LLM, usamos una regla: cualquier predicción con F1 de tokens > 0 que no sea exacta cuenta como 0.5. Es generosa, así que también reportamos la versión sin crédito parcial, que es una cota inferior.
- Puntaje reponderado = 0.75 × promedio en preguntas con respuesta + 0.25 × promedio en preguntas sin respuesta.
- Control: siempre `null` debe dar 0.25, que es el baseline de la guía. Todas las corridas lo cumplen.

Código: `src/metrics.py` (`item_score`, `guide_scores`) y `scripts/item_score.py`, que barre τ en `[−∞, −4, −2, −1, −0.5, 0, 0.5, 1, 2, 4, +∞]`.

## 3. Qué encontramos al calibrar

El mejor τ aparece en la misma zona en casi todos los checkpoints:

| Checkpoint | Mejor τ | Proxy | Con τ = −2 |
|---|---|---|---|
| LoRA r = 16, época 5 (`r16_es`) | −2 | 0.697 | 0.697 |
| Fine-tuning completo, época 2 | −2 | 0.716 | 0.716 |
| Fine-tuning completo, época 3 | −2 | 0.721 | 0.721 |
| Fine-tuning completo, época 4 | −4 | 0.721 | 0.709 |

Para el fine-tuning completo de la época 2, el puntaje queda por encima de 0.704 para cualquier τ entre −4 y −0.5. Es una meseta, no un pico aislado.

Con τ = 0 el puntaje era más bajo y más ruidoso entre épocas (la parada temprana de `r16_es` se fue por ahí). Por eso dejamos de usar τ = 0 para elegir época.

## 4. Decisión

- **Modelo:** `models/full_ft_4ep/epoch_3`, fine-tuning completo, época 3.
- **τ:** −2.

Razones: es el mejor τ común entre checkpoints, la región alrededor es estable, y la época 3 da el mejor puntaje en el proxy con ese mismo τ.

## 5. Evaluación final sobre `dev_report`

`dev_report` se usó una sola vez, con el modelo y τ ya fijados. El script `scripts/eval_report.py` no sobrescribe el resultado: si el archivo existe, se niega a correr de nuevo.

| Métrica | `dev_report` (5 956 preguntas) |
|---|---|
| EM | 62.56 |
| F1 | 66.27 |
| HasAns F1 | 74.20 |
| NoAns acc | 57.83 |
| Proxy de la guía, reponderado | 0.694 |
| Proxy, sólo exactas, reponderado | 0.647 |

Resultado: `logs/report/full_ft_4ep_epoch_3_tau-2.json`.

## 6. Cómo leer este número

- El proxy de 0.694 corresponde a unos 59 puntos según la fórmula de la guía (`(0.694 − 0.25) / (0.70 − 0.25) × 60`). Con sólo exactas, unos 53.
- La caída frente a `dev_select` (0.721 a 0.694) es esperable: la época y τ se eligieron mirando `dev_select`, así que ese número tiene un sesgo optimista.
- La nota real depende del juez LLM y de 50 preguntas ocultas. El error de muestreo de 50 preguntas ronda ±0.05, así que estos números orientan, no garantizan.

## 7. Qué no cubre esta fase

- El juez LLM de la guía: nuestro crédito parcial es una regla, no el juicio real.
- El conjunto oculto: su distribución de preguntas es desconocida; sólo asumimos que ≈ 25 % no tiene respuesta.
- Cambios de modelo o de τ después de esta evaluación: dejarían `dev_report` de ser un número limpio. Si se hacen, hay que reportarlo como tal.
