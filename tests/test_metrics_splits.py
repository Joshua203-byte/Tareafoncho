"""Pruebas de métricas, de la división de dev y del post-procesamiento.

Son pruebas de reglas que no deben romperse: si cambia la normalización, el
puntaje de una respuesta sin respuesta o la regla de decisión, estas pruebas fallan.
"""

import numpy as np
import pytest

from src.metrics import aggregate, exact_match, f1, normalize_answer, score_example
from src.postprocess import predict_example
from src.splits import split_indices, split_titles


# --- Métricas ----------------------------------------------------------------

def test_normalization_removes_case_punctuation_and_articles():
    assert normalize_answer("The Lempa River.") == "lempa river"
    assert normalize_answer("  An   apple!  ") == "apple"


def test_exact_match_ignores_case_and_punctuation():
    assert exact_match("Lempa", "lempa.") == 1.0
    assert exact_match("about 422 km", "422 km") == 0.0


def test_f1_gives_partial_credit():
    # pred = {about, 422, km}; gold = {422, km}: precisión 2/3, recall 1, F1 0.8
    assert f1("about 422 km", "422 km") == pytest.approx(0.8)
    assert f1("Guatemala", "422 km") == 0.0


def test_no_answer_scores_only_null_predictions():
    assert score_example("", [])["em"] == 1.0
    assert score_example("France", [])["em"] == 0.0
    assert score_example("", [])["has_answer"] is False


def test_answerable_takes_best_of_several_references():
    result = score_example("422 km", ["422 kilometers", "422 km"])
    assert result["em"] == 1.0 and result["has_answer"] is True


def test_aggregate_separates_has_and_no_answer():
    scores = [
        score_example("", []),  # acierto de null
        score_example("France", []),  # error: respondió sin respuesta
        score_example("France", ["France"]),  # acierto
    ]
    summary = aggregate(scores)
    assert summary["NoAns_acc"] == pytest.approx(50.0)
    assert summary["HasAns_EM"] == pytest.approx(100.0)
    assert summary["n"] == 3


# --- División de dev ---------------------------------------------------------

def test_split_is_disjoint_and_covers_all_articles():
    titles = ["A", "B", "C", "D", "E", "F", "A", "B"]
    select, report = split_titles(titles)
    assert select.isdisjoint(report)
    assert select | report == set(titles)


def test_split_keeps_each_article_in_one_side():
    titles = ["A", "A", "B", "B", "C", "D"]
    select_idx, report_idx = split_indices(titles)
    select_titles = {titles[i] for i in select_idx}
    report_titles = {titles[i] for i in report_idx}
    assert select_titles.isdisjoint(report_titles)
    assert len(select_idx) + len(report_idx) == len(titles)


def test_split_is_reproducible():
    titles = [f"t{i % 7}" for i in range(50)]
    assert split_indices(titles) == split_indices(titles)


# --- Post-procesamiento ------------------------------------------------------

def _window(start, end, offsets):
    return {"start_logits": np.array(start, dtype=float),
            "end_logits": np.array(end, dtype=float),
            "offsets": offsets}


def test_predicts_span_when_it_beats_null():
    context = "The Lempa River is long."
    # Tokens: 0=[CLS] (sin offset), 1="The"(0,3), 2="Lempa"(4,9), 3="River"(10,15)
    offsets = [None, (0, 3), (4, 9), (10, 15)]
    # El span más fuerte es el token 2 ("Lempa") como inicio y fin.
    start = [0.0, 0.1, 5.0, 0.2]
    end = [0.0, 0.1, 5.0, 0.2]
    pred = predict_example([_window(start, end, offsets)], context, tau=0.0)
    assert pred["text"] == "Lempa"
    assert pred["start"] == 4


def test_predicts_null_when_cls_wins():
    context = "The Lempa River is long."
    offsets = [None, (0, 3), (4, 9), (10, 15)]
    start = [9.0, 0.1, 1.0, 0.2]
    end = [9.0, 0.1, 1.0, 0.2]
    pred = predict_example([_window(start, end, offsets)], context, tau=0.0)
    assert pred["text"] == ""
    assert pred["start"] is None


def test_threshold_tau_moves_the_decision():
    context = "The Lempa River is long."
    offsets = [None, (0, 3), (4, 9), (10, 15)]
    start = [1.0, 0.1, 1.5, 0.2]
    end = [1.0, 0.1, 1.5, 0.2]
    window = [_window(start, end, offsets)]
    # Span = 3.0, null = 2.0: diferencia 1.0
    assert predict_example(window, context, tau=0.5)["text"] == "Lempa"
    assert predict_example(window, context, tau=1.5)["text"] == ""


def test_multiple_windows_use_minimum_null_score():
    context = "The Lempa River is long."
    offsets = [None, (0, 3), (4, 9), (10, 15)]
    # Ventana A: null alto (CLS 5+5=10). Ventana B: null bajo (0+0=0) y span fuerte.
    a = _window([5.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0], offsets)
    b = _window([0.0, 0.0, 6.0, 0.0], [0.0, 0.0, 6.0, 0.0], offsets)
    pred = predict_example([a, b], context, tau=0.0)
    assert pred["text"] == "Lempa"
    assert pred["score"] == pytest.approx(12.0 - 0.0)
