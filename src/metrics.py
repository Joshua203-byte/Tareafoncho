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
