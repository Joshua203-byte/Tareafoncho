"""Predictor: carga el modelo fusionado y responde una pregunta sobre un contexto.

Es la misma cadena que en la evaluación, en orden:

1. ``ValidationFeatures`` parte la pregunta y el contexto en ventanas de 384 tokens
   con solapamiento de 128, y guarda el offset en caracteres de cada token del contexto.
2. El modelo devuelve logits de inicio y fin para cada ventana.
3. ``predict_example`` (src/postprocess.py) decide el span o ``null`` con τ.

Relleno (padding) dinámico
--------------------------
Las ventanas de una misma pregunta se pasan en un solo lote, y un tensor necesita
que todas midan lo mismo. Por eso se rellenan con ceros, pero **no hasta 384**:
solo hasta la ventana más larga de esa petición (el mismo principio que el
``DataCollatorWithPadding`` del entrenamiento).

- Contexto corto: hay una sola ventana y no se rellena nada.
- Contexto largo: las ventanas llenas miden 384 y solo la última, más corta, se
  completa con ceros.

El relleno va al final y ``attention_mask`` lo marca con 0 para que el modelo lo
ignore. Los tokens rellenados no tienen offset (``None``), así que el
post-procesamiento los descarta aunque sus logits fueran altos.
"""

import math
import threading

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from service.config import DOC_STRIDE, MAX_ANSWER_LEN, MAX_SEQ_LENGTH, N_BEST
from src.features import ValidationFeatures
from src.postprocess import predict_example


def _sigmoid(x):
    """Comprime un número real a (0, 1). Es la forma de cumplir el rango del ``score``."""
    return 1.0 / (1.0 + math.exp(-x))


class QAPredictor:
    """Modelo cargado en memoria. Una instancia por proceso (1 worker).

    Concurrencia
    ------------
    ``/predict`` corre ``predict`` en el *threadpool* de FastAPI, así que dos
    peticiones simultáneas pueden llamarlo a la vez desde hilos distintos. Un
    ``threading.Lock`` hace que solo una use el predictor a la vez; las demás
    esperan su turno (se encolan). Se hace por tres razones:

    1. **CPU:** una inferencia ya usa todos los núcleos. Dos en paralelo no
       terminan antes, compiten por los mismos núcleos y las dos tardan más.
       En serie, cada una tarda lo que tarda sola y la latencia es predecible.
    2. **Tokenizer:** el tokenizer *fast* (escrito en Rust) cambia su estado interno
       al truncar y rellenar; usarlo desde dos hilos a la vez puede fallar con
       ``RuntimeError: Already borrowed``. Por eso el lock cubre la tokenización
       y no solo el modelo.
    3. **Memoria:** una sola inferencia a la vez acota el uso de RAM.

    Costo: si llegaran muchas peticiones juntas, la última esperaría la suma de
    todas. El harness manda una por una, así que en la práctica no hay cola.
    """

    def __init__(self, model_dir, tau):
        """Carga tokenizer y modelo desde ``model_dir`` y deja todo listo para inferir.

        ``tau`` es el umbral de "sin respuesta" (ver ``src/postprocess.py``).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        self.model.eval()  # apaga el dropout: en inferencia las respuestas deben ser deterministas
        self.tau = tau
        # Total real de parámetros del modelo servido; /info lo reporta tal cual.
        self.params = sum(p.numel() for p in self.model.parameters())
        # Mismas ventanas que en el entrenamiento (384 tokens, solapamiento 128).
        self._features = ValidationFeatures(self.tokenizer, max_length=MAX_SEQ_LENGTH, stride=DOC_STRIDE)
        # Serializa tokenización + inferencia. Motivos en el docstring de la clase.
        self._lock = threading.Lock()

    def predict(self, example_id, context, question):
        """Responde una pregunta. Devuelve ``{"answer": str|None, "start": int|None, "score": float}``.

        ``answer`` es un fragmento literal de ``context`` y ``start`` su posición
        en caracteres; ambos son ``None`` cuando la respuesta es "sin respuesta".
        ``score`` está en [0, 1] (ver ``_confidence``).
        """
        # El lock cubre tokenización e inferencia (ver el docstring de la clase).
        with self._lock:
            # 1. Ventanas con offsets. ValidationFeatures espera un lote (dict de listas), aquí de 1 ejemplo.
            feats = self._features({"id": [example_id], "question": [question], "context": [context]})
            # 2. Relleno dinámico: no se rellena a 384, sino al largo de la ventana más larga
            #    de ESTA petición. Con un contexto corto hay una sola ventana y no se rellena nada;
            #    con uno largo, solo la última ventana (más corta) se completa con ceros.
            #    attention_mask marca qué posiciones son relleno para que el modelo las ignore.
            batch = self.tokenizer.pad(
                {"input_ids": feats["input_ids"], "attention_mask": feats["attention_mask"]},
                return_tensors="pt",
            )
            # inference_mode: sin gradientes ni memoria extra, solo se calcula la salida.
            with torch.inference_mode():
                out = self.model(**batch)

        padded_len = batch["input_ids"].shape[1]
        windows = []
        # offsets[j] = (car_inicio, car_fin) del token j DENTRO DEL TEXTO ORIGINAL del contexto, no
        # su posición en la ventana. Los tokens de la pregunta y los especiales tienen None. Como
        # los caracteres se miden en el contexto, el mismo ``start`` vale para cualquier ventana.
        for i, offsets in enumerate(feats["offset_mapping"]):
            # Los offsets se rellenan con None hasta la longitud del lote, igual que los logits:
            # sin esto, un candidato en la zona de relleno daría IndexError.
            windows.append(
                {
                    "start_logits": out.start_logits[i].numpy(),
                    "end_logits": out.end_logits[i].numpy(),
                    "offsets": offsets + [None] * (padded_len - len(offsets)),
                }
            )

        # 3. Mejor span entre todas las ventanas vs. puntaje nulo, con el umbral τ.
        pred = predict_example(
            windows, context, tau=self.tau, n_best=N_BEST, max_answer_len=MAX_ANSWER_LEN
        )
        answered = pred["text"] != ""
        return {
            "answer": pred["text"] if answered else None,
            "start": pred["start"] if answered else None,
            "score": self._confidence(pred["margin"], answered),
        }

    def _confidence(self, margin, answered):
        """``score`` en [0, 1]: qué tan lejos del umbral quedó la decisión tomada.

        ``margin = mejor_span - puntaje_nulo`` y la decisión es "span" si ``margin > τ``.
        Con ``d = margin - τ``, el score es ``sigmoide(d)`` si se respondió con span y
        ``sigmoide(-d)`` si se devolvió null. Así significa lo mismo en los dos casos:
        0.5 es un empate con el umbral y se acerca a 1 cuanto más clara es la decisión.
        Sin ningún span válido (``margin`` es None) el null es seguro: score 1.0.
        No es una probabilidad calibrada.
        """
        if margin is None:
            return 1.0
        d = margin - self.tau
        return round(_sigmoid(d if answered else -d), 4)
