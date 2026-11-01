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
    params = 66362880

    def __init__(self, model_dir, tau):
        self.tau = tau

    def predict(self, example_id, context, question):
        if question == "boom":
            raise RuntimeError("fallo simulado")
        if question == "unanswerable":
            return {"answer": None, "start": None, "score": 0.84}
        return {"answer": "about 422 km", "start": 61, "score": 0.91}


def make_client(predictor_factory=FakePredictor):
    settings = Settings(api_token=TOKEN, model_dir="unused", tau=0.0, team="team-03")
    app = create_app(settings=settings, predictor_factory=predictor_factory)
    client = TestClient(app, raise_server_exceptions=False)
    return client


@pytest.fixture
def client():
    with make_client() as c:
        yield c


def auth(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


def ask(question="How long is the Lempa River?", **overrides):
    body = {"id": "q-017", "context": CONTEXT, "question": question}
    body.update(overrides)
    return body


def test_health_is_public_and_ok_after_loading(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/health").status_code == 200


def test_health_is_503_while_model_loads():
    release = threading.Event()

    def slow_factory(model_dir, tau):
        release.wait(5)
        return FakePredictor(model_dir, tau)

    with make_client(slow_factory) as c:
        response = c.get("/health")
        assert response.status_code == 503
        assert response.json() == {"status": "loading"}
        release.set()


def test_info_returns_team_model_and_params(client):
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
    response = client.get("/info", headers=headers)
    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_predict_answer_echoes_id_and_offsets_are_valid(client):
    response = client.post("/predict", json=ask(), headers=auth())
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "q-017"
    assert body["start"] == 61
    assert CONTEXT[body["start"] : body["start"] + len(body["answer"])] == body["answer"]
    assert 0.0 <= body["score"] <= 1.0


def test_predict_null_has_null_start(client):
    body = client.post("/predict", json=ask("unanswerable"), headers=auth()).json()
    assert body["answer"] is None
    assert body["start"] is None


def test_predict_accepts_context_of_5000_chars(client):
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
    response = client.post("/predict", json=body, headers=auth())
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"
    assert response.json()["detail"] == detail


def test_predict_broken_json_is_400(client):
    response = client.post("/predict", content=b'{"id": "q-1", "context": ', headers={**auth(), "Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json() == {"error": "bad_request", "detail": "body is not valid JSON"}


def test_predict_non_object_body_is_400(client):
    response = client.post("/predict", json=["q-1"], headers=auth())
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"


def test_predict_without_token_is_401_even_with_broken_json(client):
    response = client.post("/predict", content=b"not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}


def test_predict_internal_failure_is_500_internal_error(client):
    response = client.post("/predict", json=ask("boom"), headers=auth())
    assert response.status_code == 500
    assert response.json() == {"error": "internal_error"}


def test_unknown_path_keeps_error_shape(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json()["error"] == "bad_request"


def test_app_refuses_to_start_without_token(monkeypatch):
    monkeypatch.delenv("QA_API_TOKEN", raising=False)
    app = create_app(predictor_factory=FakePredictor)
    with pytest.raises(ConfigError):
        with TestClient(app):
            pass


def test_load_settings_requires_token():
    with pytest.raises(ConfigError):
        load_settings({})
    with pytest.raises(ConfigError):
        load_settings({"QA_API_TOKEN": "   "})


def test_load_settings_reads_env_and_defaults():
    cfg = load_settings({"QA_API_TOKEN": "abc", "QA_TAU": "-1.5", "QA_MODEL_DIR": "models/x", "QA_TEAM": "team-09"})
    assert cfg == Settings(api_token="abc", model_dir="models/x", tau=-1.5, team="team-09")
    assert load_settings({"QA_API_TOKEN": "abc"}).tau == 0.0


def test_load_settings_rejects_non_numeric_tau():
    with pytest.raises(ConfigError):
        load_settings({"QA_API_TOKEN": "abc", "QA_TAU": "alto"})
