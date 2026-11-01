"""Métricas de SQuAD 2.0: Exact Match (EM) y F1 a nivel de token.

Dos tipos de pregunta
---------------------
- **Con respuesta** (HasAns): la referencia tiene al menos un texto. Se mide
  EM y F1 contra la mejor de las referencias.
- **Sin respuesta** (NoAns): la referencia está vacía. La predicción es correcta
  solo si devolvemos ``null`` (cadena vacía aquí).

La normalización (minúsculas, sin puntuación, sin artículos, espacios
colapsados) es la del script oficial de SQuAD. Sin ella, "The Lempa." y "lempa"
contarían como respuestas distintas.
"""

import collections
import re
import string


def normalize_answer(text):
    """Normalización oficial de SQuAD: minúsculas, sin puntuación ni artículos."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred, gold):
    """1.0 si las dos cadenas son iguales después de normalizar, 0.0 si no."""
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred, gold):
    """Solapamiento de tokens entre predicción y referencia.

    Es la media armónica de precisión (cuántas palabras predichas están en la
    referencia) y recall (cuántas de la referencia se predijeron).
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        # Si una de las dos está vacía, solo puede haber acierto si ambas lo están.
        return float(pred_tokens == gold_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_example(pred_text, gold_texts):
    """EM, F1 y tipo (con o sin respuesta) de una pregunta.

    ``gold_texts`` puede tener varias referencias: se toma la mejor, como en la
    evaluación oficial. Una lista vacía significa "sin respuesta".
    """
    if not gold_texts:
        correct = float(normalize_answer(pred_text) == "")
        return {"em": correct, "f1": correct, "has_answer": False}
    em = max(exact_match(pred_text, g) for g in gold_texts)
    score = max(f1(pred_text, g) for g in gold_texts)
    return {"em": em, "f1": score, "has_answer": True}


def aggregate(scores):
    """Promedia las puntuaciones por ejemplo y separa HasAns y NoAns.

    ``scores`` es una lista de diccionarios devueltos por ``score_example``.
    """
    def mean(values):
        return round(100 * sum(values) / len(values), 2) if values else None

    has = [s for s in scores if s["has_answer"]]
    no = [s for s in scores if not s["has_answer"]]
    return {
        "n": len(scores),
        "EM": mean([s["em"] for s in scores]),
        "F1": mean([s["f1"] for s in scores]),
        "HasAns_n": len(has),
        "HasAns_EM": mean([s["em"] for s in has]),
        "HasAns_F1": mean([s["f1"] for s in has]),
        "NoAns_n": len(no),
        "NoAns_acc": mean([s["em"] for s in no]),
    }


# ---------------------------------------------------------------------------
# Escala de la nota en vivo (guía, sección XIV.1)
# ---------------------------------------------------------------------------
# La nota no es F1: cada pregunta vale 1 (exacta, o null correcto), 0.5 (solapamiento
# parcial juzgado válido) o 0. Aquí usamos una regla en lugar del juez LLM, así que
# el crédito parcial es un proxy. Por eso también se reporta sin crédito parcial.

SHARE_NULL = 0.25  # proporción de preguntas sin respuesta en el conjunto oculto


def item_score(pred, golds, partial_credit=True):
    """Puntaje de una pregunta según la escala de la guía.

    - Sin respuesta en la referencia: 1 si la predicción es ``null``, 0 si no.
    - Con respuesta: 0 si la predicción es ``null``. 1 si coincide exacto con alguna
      referencia. 0.5 si solapa parcialmente (solo con ``partial_credit``). 0 si no.
    """
    if not golds:
        return float(normalize_answer(pred) == "")
    if normalize_answer(pred) == "":
        return 0.0
    if max(exact_match(pred, g) for g in golds):
        return 1.0
    if partial_credit and max(f1(pred, g) for g in golds) > 0:
        return 0.5
    return 0.0


def guide_scores(pairs, share_null=SHARE_NULL):
    """Puntaje de la escala de la guía sobre una lista de ``(pred, golds)``.

    Devuelve tres valores en [0, 1]:

    - ``guide_raw``: promedio simple sobre las preguntas recibidas.
    - ``guide_reweighted``: promedia por tipo y pondera a ``share_null`` sin
      respuesta. Es la aproximación al conjunto oculto. La parada temprana usa este.
    - ``guide_exact_reweighted``: igual, pero sin crédito parcial. Es una cota inferior.

    Con ``share_null`` igual a 0.25, siempre ``null`` da 0.25: es el baseline de la guía.
    """
    has_full, has_exact, no_answer, every = [], [], [], []
    for pred, golds in pairs:
        full = item_score(pred, golds, partial_credit=True)
        every.append(full)
        if golds:
            has_full.append(full)
            has_exact.append(item_score(pred, golds, partial_credit=False))
        else:
            no_answer.append(full)
    if not has_full or not no_answer:
        raise ValueError("guide_scores necesita preguntas con y sin respuesta")

    def mean(values):
        return sum(values) / len(values)

    no_mean = mean(no_answer)
    return {
        "guide_raw": round(mean(every), 4),
        "guide_reweighted": round((1 - share_null) * mean(has_full) + share_null * no_mean, 4),
        "guide_exact_reweighted": round((1 - share_null) * mean(has_exact) + share_null * no_mean, 4),
    }
