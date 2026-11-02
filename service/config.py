"""Configuración del servicio, leída del entorno al arrancar.

Variables de entorno
--------------------
``QA_API_TOKEN``  token compartido (obligatorio). Si falta o está vacío, el
                  servicio no arranca: es mejor caerse que quedar abierto.
``QA_MODEL_DIR``  carpeta con el modelo fusionado y su tokenizer.
``QA_TAU``        umbral τ de "sin respuesta". Pendiente: se fija al final con
                  el modelo definitivo. Mientras tanto vale 0.0, que NO es un
                  valor calibrado.
``QA_TEAM``       nombre del equipo, tal como lo registró el instructor.

El token nunca se guarda en el código ni en un archivo versionado. Se lee solo
aquí (``load_settings``) y se compara en ``service/app.py``.

Las constantes de abajo (ventana, solapamiento, n_best, longitud máxima) no son
variables de entorno a propósito: deben coincidir con las del entrenamiento y
con lo que ``/info`` informa.
"""

import os
from dataclasses import dataclass

DEFAULT_MODEL_DIR = "models/r16_es/merged_epoch5"
DEFAULT_TAU = 0.0  # placeholder: el valor final se fija con los resultados de dev_select
DEFAULT_TEAM = "Fonchos"
BASE_MODEL_ID = "distilbert-base-uncased"  # checkpoint base del que parte nuestro fine-tuning

MAX_SEQ_LENGTH = 384  # tokens por ventana, igual que en el entrenamiento
DOC_STRIDE = 128  # solapamiento entre ventanas consecutivas
N_BEST = 20  # cuántos inicios y fines considerar por ventana
MAX_ANSWER_LEN = 30  # longitud máxima del span, en tokens


class ConfigError(RuntimeError):
    """El entorno no tiene lo mínimo para arrancar el servicio."""


@dataclass(frozen=True)
class Settings:
    """Configuración ya validada. Es inmutable: no cambia mientras el servicio corre."""

    api_token: str
    model_dir: str
    tau: float
    team: str


def load_settings(env=None):
    """Lee y valida la configuración.

    ``env`` es un dict opcional que sustituye a ``os.environ``; las pruebas lo
    usan para no tocar el entorno real. Lanza ``ConfigError`` si falta el token
    o si ``QA_TAU`` no es un número.
    """
    env = os.environ if env is None else env
    token = env.get("QA_API_TOKEN", "").strip()
    if not token:
        raise ConfigError("QA_API_TOKEN no está definido: el servicio no arranca sin token.")
    try:
        tau = float(env.get("QA_TAU", DEFAULT_TAU))
    except ValueError as exc:
        raise ConfigError(f"QA_TAU debe ser un número, no {env['QA_TAU']!r}.") from exc
    return Settings(
        api_token=token,
        model_dir=env.get("QA_MODEL_DIR", DEFAULT_MODEL_DIR),
        tau=tau,
        team=env.get("QA_TEAM", DEFAULT_TEAM),
    )
