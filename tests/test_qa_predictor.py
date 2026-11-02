"""Pruebas del predictor con el modelo real. Se saltan si no hay pesos en disco.

Los pesos no van a git: en una copia limpia hay que apuntar ``QA_MODEL_DIR``
a la carpeta del modelo fusionado. Estas pruebas son las que dicen si el
modelo que sirve la API responde algo sensato.
"""

import json
import os
from pathlib import Path

import pytest

from service.config import DEFAULT_MODEL_DIR

MODEL_DIR = os.environ.get("QA_MODEL_DIR", DEFAULT_MODEL_DIR)
SAMPLE = Path(__file__).resolve().parent.parent / "public_sample.json"

requires_model = pytest.mark.skipif(
    not (Path(MODEL_DIR) / "model.safetensors").exists(),
    reason=f"no hay pesos en {MODEL_DIR}",
)


@pytest.fixture(scope="module")
def predictor():
    """Un solo predictor para todo el módulo: cargar el modelo cuesta unos segundos."""
    from service.qa import QAPredictor

    return QAPredictor(MODEL_DIR, tau=0.0)


@requires_model
def test_answer_is_a_verbatim_span(predictor):
    """La respuesta es un trozo literal del contexto y ``start`` apunta a él (o ambos son None)."""
    context = "The Lempa River is the longest river in El Salvador, flowing about 422 km from Guatemala to the Pacific Ocean."
    pred = predictor.predict("t-1", context, "How long is the Lempa River?")
    if pred["answer"] is not None:
        assert context[pred["start"] : pred["start"] + len(pred["answer"])] == pred["answer"]
    else:
        assert pred["start"] is None
    assert 0.0 <= pred["score"] <= 1.0


@requires_model
def test_long_context_is_handled_in_windows(predictor):
    """El contexto más largo del set público (varias ventanas) también devuelve un span válido."""
    items = json.loads(SAMPLE.read_text(encoding="utf-8"))["items"]
    long_item = max(items, key=lambda it: len(it["context"]))
    pred = predictor.predict(long_item["id"], long_item["context"], long_item["question"])
    if pred["answer"] is not None:
        start = pred["start"]
        assert long_item["context"][start : start + len(pred["answer"])] == pred["answer"]


@requires_model
def test_sample_set_has_at_least_four_of_five_right(predictor):
    """Criterio de la Fase 4: ≥ 4 de 5 en el set público, con ambos null incluidos."""
    items = json.loads(SAMPLE.read_text(encoding="utf-8"))["items"]
    correct = 0
    for it in items:
        pred = predictor.predict(it["id"], it["context"], it["question"])
        if it.get("is_impossible"):
            correct += pred["answer"] is None
        else:
            correct += pred["answer"] in {a["text"] for a in it["answers"]}
    assert correct >= min(4, len(items))
