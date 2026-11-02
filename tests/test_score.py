"""Pruebas del ``score`` de la API (opción B: margen de decisión con sigmoide).

No cargan el modelo: ``_confidence`` solo usa τ, así que se crea el predictor
vacío con ``__new__`` y se le pone τ a mano.
"""

import pytest

from service.qa import QAPredictor
from src.postprocess import predict_example


def predictor_with_tau(tau):
    """QAPredictor sin modelo, solo con τ, para probar el cálculo del score."""
    p = QAPredictor.__new__(QAPredictor)
    p.tau = tau
    return p


def test_score_is_half_at_the_threshold_and_grows_with_distance():
    """Un margen justo en τ da 0.5; más lejos de τ, más cerca de 1, en span y en null."""
    p = predictor_with_tau(-2.0)
    assert p._confidence(margin=-2.0, answered=False) == 0.5
    assert p._confidence(margin=3.0, answered=True) > p._confidence(margin=-1.0, answered=True) > 0.5
    assert p._confidence(margin=-9.0, answered=False) > p._confidence(margin=-3.0, answered=False) > 0.5


def test_score_means_the_same_for_span_and_null():
    """La misma distancia al umbral da el mismo score, respondas span o null."""
    p = predictor_with_tau(0.0)
    assert p._confidence(margin=2.0, answered=True) == p._confidence(margin=-2.0, answered=False)


def test_score_is_one_when_there_is_no_valid_span():
    """Si ninguna ventana tuvo un span válido, el null es seguro."""
    assert predictor_with_tau(0.0)._confidence(margin=None, answered=False) == 1.0


@pytest.mark.parametrize("margin", [-50.0, -1.0, 0.0, 1.0, 50.0])
@pytest.mark.parametrize("answered", [True, False])
def test_score_stays_in_range(margin, answered):
    """Sea cual sea el margen, el score cumple el contrato: está en [0, 1]."""
    assert 0.0 <= predictor_with_tau(0.0)._confidence(margin, answered) <= 1.0


def test_predict_example_exposes_margin_without_changing_the_decision():
    """``margin`` es mejor_span − nulo y no depende de τ; la decisión sí."""
    import numpy as np

    start = np.array([0.0, 0.0, 3.0, 0.0])  # CLS=0, mejor inicio en el token 2
    end = np.array([0.0, 0.0, 3.0, 0.0])
    window = [{"start_logits": start, "end_logits": end, "offsets": [None, None, (0, 5), None]}]
    context = "Lempa river"
    answered = predict_example(window, context, tau=0.0)
    refused = predict_example(window, context, tau=10.0)
    assert answered["margin"] == refused["margin"] == 6.0
    assert answered["text"] == "Lempa" and refused["text"] == ""
