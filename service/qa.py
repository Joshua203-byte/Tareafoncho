"""Predictor: carga el modelo fusionado y responde una pregunta sobre un contexto.

Es la misma cadena que en la evaluación, en orden:

1. ``ValidationFeatures`` parte la pregunta y el contexto en ventanas de 384 tokens
   con solapamiento de 128, y guarda el offset en caracteres de cada token del contexto.
2. El modelo devuelve logits de inicio y fin para cada ventana.
3. ``predict_example`` (src/postprocess.py) decide el span o ``null`` con τ.

Las ventanas de una misma pregunta se pasan en un solo lote con relleno. El
relleno va al final: los tokens rellenados no tienen offset, así que el
post-procesamiento los descarta.
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
    """Modelo cargado en memoria. Una instancia por proceso (1 worker)."""

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
        # El modelo es único y la inferencia en CPU no gana con concurrencia: se serializa.
        self._lock = threading.Lock()

    def predict(self, example_id, context, question):
        """Responde una pregunta. Devuelve ``{"answer": str|None, "start": int|None, "score": float}``.

        ``answer`` es un fragmento literal de ``context`` y ``start`` su posición
        en caracteres; ambos son ``None`` cuando la respuesta es "sin respuesta".
        ``score`` está en [0, 1] (ver ``_confidence``).
        """
        # 1. Ventanas con offsets. ValidationFeatures espera un lote (dict de listas), aquí de 1 ejemplo.
        feats = self._features({"id": [example_id], "question": [question], "context": [context]})
        # 2. Las ventanas miden distinto: se rellenan al largo de la más larga para formar un tensor.
        batch = self.tokenizer.pad(
            {"input_ids": feats["input_ids"], "attention_mask": feats["attention_mask"]},
            return_tensors="pt",
        )
        # inference_mode: sin gradientes ni memoria extra, solo se calcula la salida.
        with self._lock, torch.inference_mode():
            out = self.model(**batch)

        padded_len = batch["input_ids"].shape[1]
        windows = []
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
