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

Control de acceso
-----------------
**Dónde se valida el token:** en ``is_authorized`` (más abajo), que llaman ``/info`` y
``/predict`` como primera comprobación, antes de leer el cuerpo. El valor esperado sale
de ``QA_API_TOKEN`` vía ``service/config.py`` y vive en ``app.state.settings``; nunca
está en el código ni en git.

**Qué NO hace la autenticación** (límites conocidos, también en REPORT.md §9):

- Es un único token compartido por todos los clientes: no distingue quién llama.
- No hay rotación automática: para cambiarlo se cambia ``QA_API_TOKEN`` y se reinicia.
- No limita peticiones por segundo ni bloquea tras intentos fallidos, así que alguien
  podría probar tokens sin freno (``compare_digest`` solo evita filtrar el token por tiempo).
- No cifra nada por sí misma: el cifrado lo pone el túnel HTTPS.
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
    """Cuerpo de ``/predict``. ``extra="forbid"`` sigue el esquema del contrato.

    Los límites de longitud son los del contrato: id 1–64, context 1–5000,
    question 1–500 caracteres. Un tipo equivocado (p. ej. un número en ``id``)
    también falla la validación.
    """

    model_config = ConfigDict(extra="forbid")  # un campo desconocido es un 400, no se ignora

    id: str = Field(min_length=1, max_length=64)
    context: str = Field(min_length=1, max_length=5000)
    question: str = Field(min_length=1, max_length=500)


def error_response(status, code, detail=None, headers=None):
    """Respuesta de error con la forma del contrato: ``{"error": code, "detail": ...}``.

    ``detail`` es opcional y solo se incluye si se da.
    """
    body = {"error": code}
    if detail is not None:
        body["detail"] = detail
    return JSONResponse(body, status_code=status, headers=headers)


def unauthorized():
    """401 con ``WWW-Authenticate: Bearer``, como pide el contrato."""
    return error_response(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})


def is_authorized(request, expected_token):
    """Compara el token en tiempo constante y el esquema ``Bearer`` exacto (sensible a mayúsculas).

    ``hmac.compare_digest`` tarda lo mismo aunque el token acierte más o menos
    caracteres, así que el tiempo de respuesta no filtra el token.

    Este es el único lugar donde se valida el token. Es un único token compartido,
    sin rotación ni límite de intentos (ver "Control de acceso" al inicio del módulo).
    """
    # "Bearer abc" -> scheme="Bearer", token="abc". Si no hay cabecera, ambos quedan vacíos.
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme != "Bearer" or not token:
        return False
    return hmac.compare_digest(token.encode("utf-8"), expected_token.encode("utf-8"))


def first_validation_detail(exc):
    """Convierte el primer error de Pydantic en un texto corto para el campo ``detail``.

    Un campo ausente da ``field 'question' is required``, igual que el ejemplo del contrato.
    """
    error = exc.errors()[0]
    field = ".".join(str(part) for part in error["loc"] if part != "body")
    if error["type"] == "missing":
        return f"field '{field}' is required"
    return f"field '{field}': {error['msg']}"


def create_app(settings=None, predictor_factory=None):
    """Crea la aplicación. Los argumentos existen para las pruebas.

    ``settings=None`` lee el entorno al arrancar: si falta ``QA_API_TOKEN``, el
    arranque falla. ``predictor_factory(model_dir, tau)`` devuelve el predictor;
    por defecto es ``QAPredictor``, y las pruebas pasan uno falso para no cargar el modelo.
    """
    factory = predictor_factory or QAPredictor

    def load_predictor(app, cfg):
        """Carga el modelo (lento) y lo deja en ``app.state.predictor``. Corre en un hilo."""
        try:
            app.state.predictor = factory(cfg.model_dir, cfg.tau, tokenizer_dir=cfg.tokenizer_dir)
            logger.info("modelo cargado desde %s (tau=%s)", cfg.model_dir, cfg.tau)
        except Exception:
            # Sin modelo el servicio se queda en 503 "loading": mejor que responder mal.
            logger.exception("no se pudo cargar el modelo desde %s", cfg.model_dir)

    @asynccontextmanager
    async def lifespan(app):
        """Arranque: lee la configuración y lanza la carga del modelo sin bloquear el servidor."""
        cfg = load_settings() if settings is None else settings  # sin token, esto lanza y no arranca
        app.state.settings = cfg
        app.state.predictor = None  # None = todavía cargando
        # Hilo aparte: si la carga bloqueara el arranque, /health ni siquiera respondería 503.
        threading.Thread(target=load_predictor, args=(app, cfg), daemon=True).start()
        yield

    app = FastAPI(title="Extractive QA Service", version="1.0", lifespan=lifespan)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        """Errores HTTP de FastAPI (404, 405...) con la forma de error del contrato."""
        if exc.status_code == 401:
            return unauthorized()
        return error_response(exc.status_code, "bad_request", str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled_error(request, exc):
        """Cualquier fallo inesperado es un 500 ``internal_error``; el detalle va al log, no al cliente."""
        logger.exception("error no controlado en %s", request.url.path)
        return error_response(500, "internal_error")

    @app.get("/health")
    def health():
        """Público. 200 ``ok`` cuando el modelo está cargado; 503 ``loading`` mientras tanto."""
        if app.state.predictor is None:
            return JSONResponse({"status": "loading"}, status_code=503)
        return {"status": "ok"}

    @app.get("/info")
    def info(request: Request):
        """Protegido. Equipo, checkpoint base, parámetros reales y parámetros de ventana."""
        if not is_authorized(request, app.state.settings.api_token):
            return unauthorized()
        predictor = app.state.predictor
        if predictor is None:
            return error_response(503, "internal_error", "model is loading")
        return {
            "team": app.state.settings.team,
            "model": BASE_MODEL_ID,  # el checkpoint base, no la carpeta que servimos
            "params": predictor.params,
            "max_seq_length": MAX_SEQ_LENGTH,
            "doc_stride": DOC_STRIDE,
        }

    @app.post("/predict")
    async def predict(request: Request):
        """Protegido. Orden de las comprobaciones: token, JSON, esquema, modelo, inferencia.

        Se lee el cuerpo a mano (no con un parámetro tipado) para poder revisar el
        token antes de validar el cuerpo: 401 gana sobre 400.
        """
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
        # La inferencia bloquea la CPU: va a un hilo para que el servidor siga atendiendo /health.
        result = await run_in_threadpool(predictor.predict, item.id, item.context, item.question)
        # Solo se registra el tiempo: nunca el contexto ni la pregunta.
        logger.info("predict ms=%.0f", (time.perf_counter() - started) * 1000)
        return {"id": item.id, **result}  # el id se devuelve tal cual llegó

    return app


app = create_app()  # instancia que carga uvicorn: ``uvicorn service.app:app``
