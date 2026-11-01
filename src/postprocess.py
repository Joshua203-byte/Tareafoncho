"""Post-procesamiento: de logits de ventanas a una respuesta por pregunta.

Para cada ventana, el modelo devuelve dos vectores de logits (inicio y fin), uno
por token. Este módulo los convierte en una sola respuesta por pregunta:

1. **Candidatos de span.** Se toman los ``n_best`` mejores inicios y fines de la
   ventana. Se descartan los pares que no son válidos: fin antes que inicio,
   longitud mayor que ``max_answer_len`` o tokens fuera del contexto (offset
   ``None``).
2. **Score de no respuesta.** Es ``start[CLS] + end[CLS]`` en cada ventana. Como
   una pregunta puede tener varias ventanas, se toma el **mínimo** entre ellas:
   la ventana que más evidencia tiene de que no hay respuesta.
3. **Decisión.** Se devuelve el mejor span de todas las ventanas si supera el
   score nulo por más de τ (regla del paper de BERT, §4.3):

       span  si  mejor_span  >  score_nulo + τ
       null  si no

   Con τ = 0 se responde cuando el span es más probable que el [CLS]. τ es el
   umbral que calibramos en la Fase 4.

Por qué el texto sale del contexto y no de los tokens
-----------------------------------------------------
Los tokens están en minúsculas y partidos en subpalabras. Cortamos la respuesta
del texto original con los offsets: ``context[inicio:fin]``. Así el texto
devuelto es exactamente un fragmento del contexto, que es lo que exige la API.
"""

import numpy as np

CLS_INDEX = 0


def best_span_in_window(start_logits, end_logits, offsets, n_best=20, max_answer_len=30):
    """Mejor span válido dentro de una ventana.

    Devuelve ``(score, (char_inicio, char_fin))`` o ``(None, None)`` si ningún
    par de candidatos es válido.
    """
    start_idx = np.argsort(start_logits)[-n_best:][::-1]
    end_idx = np.argsort(end_logits)[-n_best:][::-1]
    best_score, best_chars = None, None
    for s in start_idx:
        for e in end_idx:
            if offsets[s] is None or offsets[e] is None:
                continue  # token de la pregunta o especial: no es parte del contexto
            if e < s or (e - s + 1) > max_answer_len:
                continue
            score = float(start_logits[s] + end_logits[e])
            if best_score is None or score > best_score:
                best_score = score
                best_chars = (offsets[s][0], offsets[e][1])
    return best_score, best_chars


def predict_example(windows, context, tau=0.0, n_best=20, max_answer_len=30):
    """Respuesta final de una pregunta a partir de todas sus ventanas.

    ``windows`` es una lista de dicts con ``start_logits``, ``end_logits`` y
    ``offsets`` (uno por ventana). Devuelve ``{"text": str, "start": int|None,
    "score": float}``. ``text == ""`` significa "sin respuesta".
    """
    best_score, best_chars, null_score = None, None, None
    for w in windows:
        window_null = float(w["start_logits"][CLS_INDEX] + w["end_logits"][CLS_INDEX])
        null_score = window_null if null_score is None else min(null_score, window_null)
        span_score, chars = best_span_in_window(
            w["start_logits"], w["end_logits"], w["offsets"], n_best, max_answer_len
        )
        if span_score is not None and (best_score is None or span_score > best_score):
            best_score, best_chars = span_score, chars

    if best_chars is None or best_score <= null_score + tau:
        return {"text": "", "start": None, "score": null_score}

    start, end = best_chars
    text = context[start:end]
    # Guarda para cambios futuros: hoy es trivialmente cierta, porque text sale de
    # context[start:end]. Si alguien cambia cómo se calcula text, esto lo detecta.
    assert context[start : start + len(text)] == text
    return {"text": text, "start": start, "score": best_score - null_score}
