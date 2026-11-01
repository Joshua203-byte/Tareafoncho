"""API FastAPI según ``qa_api_contract.yaml``.

Decisiones que siguen el contrato al pie de la letra:

- ``/health`` es público. Responde 503 ``loading`` hasta que el modelo está en memoria.
- ``/info`` y ``/predict`` exigen ``Authorization: Bearer <QA_API_TOKEN>``. La
  autenticación va antes de leer el cuerpo: sin token válido siempre es 401, aunque
  el JSON esté roto.
- Todo error tiene la forma ``{"error": <código>, "detail": <texto opcional>}``.
- El modelo se carga en un hilo al arrancar (*lifespan*), así que ``/health`` ya
  responde 503 mientras tanto. Uvicorn arranca con 1 worker: el modelo vive en un
  solo proceso.
"""

import hmac
import json
import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from service.config import BASE_MODEL_ID, DOC_STRIDE, MAX_SEQ_LENGTH, load_settings
from service.qa import QAPredictor

logger = logging.getLogger("qa_service")


class PredictRequest(BaseModel):
    """Cuerpo de ``/predict``. ``extra="forbid"`` sigue el esquema del contrato."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    context: str = Field(min_length=1, max_length=5000)
    question: str = Field(min_length=1, max_length=500)


def error_response(status, code, detail=None, headers=None):
    body = {"error": code}
    if detail is not None:
        body["detail"] = detail
    return JSONResponse(body, status_code=status, headers=headers)


def unauthorized():
    return error_response(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})


def is_authorized(request, expected_token):
    """Compara el token en tiempo constante y el esquema ``Bearer`` exacto (sensible a mayúsculas)."""
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme != "Bearer" or not token:
        return False
    return hmac.compare_digest(token.encode("utf-8"), expected_token.encode("utf-8"))


def first_validation_detail(exc):
    error = exc.errors()[0]
    field = ".".join(str(part) for part in error["loc"] if part != "body")
    if error["type"] == "missing":
        return f"field '{field}' is required"
    return f"field '{field}': {error['msg']}"


def create_app(settings=None, predictor_factory=None):
    """Crea la aplicación. Los argumentos existen para las pruebas.

    ``settings=None`` lee el entorno al arrancar: si falta ``QA_API_TOKEN``, el
    arranque falla. ``predictor_factory(model_dir, tau)`` devuelve el predictor.
    """
    factory = predictor_factory or QAPredictor

    def load_predictor(app, cfg):
        try:
            app.state.predictor = factory(cfg.model_dir, cfg.tau)
            logger.info("modelo cargado desde %s (tau=%s)", cfg.model_dir, cfg.tau)
        except Exception:
            # Sin modelo el servicio se queda en 503 "loading": mejor que responder mal.
            logger.exception("no se pudo cargar el modelo desde %s", cfg.model_dir)

    @asynccontextmanager
    async def lifespan(app):
        cfg = load_settings() if settings is None else settings
        app.state.settings = cfg
        app.state.predictor = None
        threading.Thread(target=load_predictor, args=(app, cfg), daemon=True).start()
        yield

    app = FastAPI(title="Extractive QA Service", version="1.0", lifespan=lifespan)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        if exc.status_code == 401:
            return unauthorized()
        return error_response(exc.status_code, "bad_request", str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled_error(request, exc):
        logger.exception("error no controlado en %s", request.url.path)
        return error_response(500, "internal_error")

    @app.get("/health")
    def health():
        if app.state.predictor is None:
            return JSONResponse({"status": "loading"}, status_code=503)
        return {"status": "ok"}

    @app.get("/info")
    def info(request: Request):
        if not is_authorized(request, app.state.settings.api_token):
            return unauthorized()
        predictor = app.state.predictor
        if predictor is None:
            return error_response(503, "internal_error", "model is loading")
        return {
            "team": app.state.settings.team,
            "model": BASE_MODEL_ID,
            "params": predictor.params,
            "max_seq_length": MAX_SEQ_LENGTH,
            "doc_stride": DOC_STRIDE,
        }

    @app.post("/predict")
    async def predict(request: Request):
        if not is_authorized(request, app.state.settings.api_token):
            return unauthorized()
        try:
            data = json.loads(await request.body())
        except ValueError:
            return error_response(400, "bad_request", "body is not valid JSON")
        if not isinstance(data, dict):
            return error_response(400, "bad_request", "body must be a JSON object")
        try:
            item = PredictRequest.model_validate(data)
        except ValidationError as exc:
            return error_response(400, "bad_request", first_validation_detail(exc))

        predictor = app.state.predictor
        if predictor is None:
            return error_response(500, "internal_error", "model is loading")

        started = time.perf_counter()
        result = await run_in_threadpool(predictor.predict, item.id, item.context, item.question)
        logger.info("predict ms=%.0f", (time.perf_counter() - started) * 1000)
        return {"id": item.id, **result}

    return app


app = create_app()
