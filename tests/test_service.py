"""Pruebas del contrato HTTP. Usan un predictor falso: no cargan el modelo.

Lo que se prueba aquí es la forma de las respuestas, la autenticación y los
errores. Que el modelo responda bien se prueba en test_qa_predictor.py.
"""

import threading

import pytest
from fastapi.testclient import TestClient

from service.app import create_app
from service.config import ConfigError, Settings, load_settings

TOKEN = "test-token-123"
CONTEXT = "The Lempa River is the longest river in El Salvador, flowing about 422 km from Guatemala to the Pacific Ocean."


class FakePredictor:
    """Sustituye al modelo: responde según la pregunta, sin cargar nada.

    "unanswerable" simula un null, "boom" simula un fallo interno y cualquier otra
    pregunta devuelve el span de ejemplo del contrato.
    """

    params = 66362880

    def __init__(self, model_dir, tau, tokenizer_dir=None):
        self.tau = tau

    def predict(self, example_id, context, question):
        if question == "boom":
            raise RuntimeError("fallo simulado")
        if question == "unanswerable":
            return {"answer": None, "start": None, "score": 0.84}
        return {"answer": "about 422 km", "start": 61, "score": 0.91}


def make_client(predictor_factory=FakePredictor):
    """Cliente de pruebas con token ``TOKEN`` y un predictor falso. ``raise_server_exceptions=False`` deja ver el 500 real."""
    settings = Settings(api_token=TOKEN, model_dir="unused", tau=0.0, team="team-03")
    app = create_app(settings=settings, predictor_factory=predictor_factory)
    client = TestClient(app, raise_server_exceptions=False)
    return client


@pytest.fixture
def client():
    """Cliente con el servicio ya arrancado (``with`` ejecuta el lifespan)."""
    with make_client() as c:
        yield c


def auth(token=TOKEN):
    """Cabecera de autorización Bearer."""
    return {"Authorization": f"Bearer {token}"}


def ask(question="How long is the Lempa River?", **overrides):
    """Cuerpo válido de /predict; ``overrides`` cambia o añade campos para probar errores."""
    body = {"id": "q-017", "context": CONTEXT, "question": question}
    body.update(overrides)
    return body


def test_health_is_public_and_ok_after_loading(client):
    """/health no pide token y responde 200 ok cuando el modelo está cargado."""
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/health").status_code == 200


def test_health_is_503_while_model_loads():
    """Mientras el modelo carga, /health responde 503 loading (la carga se retiene con un Event)."""
    release = threading.Event()

    def slow_factory(model_dir, tau, tokenizer_dir=None):
        release.wait(5)
        return FakePredictor(model_dir, tau)

    with make_client(slow_factory) as c:
        response = c.get("/health")
        assert response.status_code == 503
        assert response.json() == {"status": "loading"}
        release.set()


def test_info_returns_team_model_and_params(client):
    """/info con token válido devuelve los cinco campos del contrato."""
    response = client.get("/info", headers=auth())
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "team": "team-03",
        "model": "distilbert-base-uncased",
        "params": 66362880,
        "max_seq_length": 384,
        "doc_stride": 128,
    }


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": f"Bearer wrong-token"},
        {"Authorization": f"bearer {TOKEN}"},  # el esquema es sensible a mayúsculas
        {"Authorization": f"Token {TOKEN}"},
        {"Authorization": TOKEN},
        {"Authorization": f"Bearer {TOKEN.upper()}"},  # el token también distingue mayúsculas
    ],
)
def test_info_rejects_missing_or_wrong_token(client, headers):
    """Sin token, con token malo, o con esquema/mayúsculas distintas: 401 con WWW-Authenticate."""
    response = client.get("/info", headers=headers)
    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_predict_answer_echoes_id_and_offsets_are_valid(client):
    """El id vuelve igual y ``context[start:start+len(answer)] == answer``, la regla que decide si un ítem cuenta."""
    response = client.post("/predict", json=ask(), headers=auth())
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "q-017"
    assert body["start"] == 61
    assert CONTEXT[body["start"] : body["start"] + len(body["answer"])] == body["answer"]
    assert 0.0 <= body["score"] <= 1.0


def test_predict_null_has_null_start(client):
    """Si answer es null, start también (y solo entonces)."""
    body = client.post("/predict", json=ask("unanswerable"), headers=auth()).json()
    assert body["answer"] is None
    assert body["start"] is None


def test_predict_accepts_context_of_5000_chars(client):
    """5000 caracteres es el máximo permitido y debe aceptarse."""
    response = client.post("/predict", json=ask(context="a" * 5000), headers=auth())
    assert response.status_code == 200


@pytest.mark.parametrize(
    "body, detail",
    [
        ({"id": "q-1", "context": CONTEXT}, "field 'question' is required"),
        ({"context": CONTEXT, "question": "q"}, "field 'id' is required"),
        (ask(context="a" * 5001), "field 'context': String should have at most 5000 characters"),
        (ask(question="q" * 501), "field 'question': String should have at most 500 characters"),
        (ask(id="x" * 65), "field 'id': String should have at most 64 characters"),
        (ask(id=""), "field 'id': String should have at least 1 character"),
        (ask(id=17), "field 'id': Input should be a valid string"),
        (ask(extra="no"), "field 'extra': Extra inputs are not permitted"),
    ],
)
def test_predict_invalid_body_is_400_bad_request(client, body, detail):
    """Campos ausentes, demasiado largos, vacíos, de tipo equivocado o extra: 400 bad_request con detail."""
    response = client.post("/predict", json=body, headers=auth())
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"
    assert response.json()["detail"] == detail


def test_predict_broken_json_is_400(client):
    """Un JSON truncado es 400, no 500."""
    response = client.post("/predict", content=b'{"id": "q-1", "context": ', headers={**auth(), "Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json() == {"error": "bad_request", "detail": "body is not valid JSON"}


def test_predict_non_object_body_is_400(client):
    """Un JSON válido pero que no es un objeto (aquí una lista) también es 400."""
    response = client.post("/predict", json=["q-1"], headers=auth())
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"


def test_predict_without_token_is_401_even_with_broken_json(client):
    """La autenticación se revisa antes que el cuerpo: sin token siempre es 401."""
    response = client.post("/predict", content=b"not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}


def test_predict_internal_failure_is_500_internal_error(client):
    """Una excepción dentro del predictor se convierte en 500 internal_error, sin filtrar el mensaje."""
    response = client.post("/predict", json=ask("boom"), headers=auth())
    assert response.status_code == 500
    assert response.json() == {"error": "internal_error"}


def test_unknown_path_keeps_error_shape(client):
    """Hasta un 404 de ruta inexistente usa la forma {"error": ...}."""
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json()["error"] == "bad_request"


def test_app_refuses_to_start_without_token(monkeypatch):
    """Sin QA_API_TOKEN el arranque (lifespan) falla: mejor caerse que quedar abierto."""
    monkeypatch.delenv("QA_API_TOKEN", raising=False)
    app = create_app(predictor_factory=FakePredictor)
    with pytest.raises(ConfigError):
        with TestClient(app):
            pass


def test_load_settings_requires_token():
    """Un token ausente o solo con espacios es un error de configuración."""
    with pytest.raises(ConfigError):
        load_settings({})
    with pytest.raises(ConfigError):
        load_settings({"QA_API_TOKEN": "   "})


def test_load_settings_reads_env_and_defaults():
    """Lee las cuatro variables; sin QA_TAU, τ vale -2.0 y sin QA_TEAM el nombre es Fonchos."""
    cfg = load_settings({"QA_API_TOKEN": "abc", "QA_TAU": "-1.5", "QA_MODEL_DIR": "models/x", "QA_TEAM": "team-09"})
    assert cfg == Settings(api_token="abc", model_dir="models/x", tau=-1.5, team="team-09")
    defaults = load_settings({"QA_API_TOKEN": "abc"})
    assert defaults.tau == -2.0
    assert defaults.tokenizer_dir is None
    assert defaults.team == "Fonchos"


def test_load_settings_rejects_non_numeric_tau():
    """QA_TAU="alto" no es un número: error claro al arrancar, no al primer /predict."""
    with pytest.raises(ConfigError):
        load_settings({"QA_API_TOKEN": "abc", "QA_TAU": "alto"})
