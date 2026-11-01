"""Pruebas de la escala de la guía y de la selección de época.

La parada temprana y la elección de la mejor época deciden qué modelo se sirve.
Por eso se prueban por separado, sin entrenar nada.
"""

from src.metrics import guide_scores, item_score
from src.train import EpochSelector


def test_item_score_exact_partial_and_null():
    assert item_score("Tegucigalpa", ["Tegucigalpa"]) == 1.0
    assert item_score("the Tegucigalpa city", ["Tegucigalpa"]) == 0.5
    assert item_score("the Tegucigalpa city", ["Tegucigalpa"], partial_credit=False) == 0.0
    assert item_score("Paris", ["Tegucigalpa"]) == 0.0
    assert item_score("", ["Tegucigalpa"]) == 0.0
    assert item_score("", []) == 1.0
    assert item_score("Paris", []) == 0.0


def test_guide_scores_always_null_is_baseline():
    # Siempre null: las preguntas sin respuesta aciertan y las demás fallan.
    pairs = [("", ["lempa"]), ("", []), ("", ["x"]), ("", [])]
    scores = guide_scores(pairs)
    assert scores["guide_reweighted"] == 0.25


def test_selector_stops_after_patience_without_gain():
    selector = EpochSelector(patience=2)
    assert selector.update(1, 0.60) is False
    assert selector.update(2, 0.65) is False
    assert selector.update(3, 0.64) is False
    assert selector.update(4, 0.63) is True
    assert selector.best_epoch == 2
    assert selector.best_value == 0.65


def test_selector_never_stops_without_patience():
    selector = EpochSelector(patience=0)
    results = [selector.update(e, v) for e, v in [(1, 0.7), (2, 0.6), (3, 0.5)]]
    assert not any(results)
    assert selector.best_epoch == 1
