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
    return 1.0 / (1.0 + math.exp(-x))


class QAPredictor:
    """Modelo cargado en memoria. Una instancia por proceso (1 worker)."""

    def __init__(self, model_dir, tau):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        self.model.eval()
        self.tau = tau
        self.params = sum(p.numel() for p in self.model.parameters())
        self._features = ValidationFeatures(self.tokenizer, max_length=MAX_SEQ_LENGTH, stride=DOC_STRIDE)
        # El modelo es único y la inferencia en CPU no gana con concurrencia: se serializa.
        self._lock = threading.Lock()

    def predict(self, example_id, context, question):
        """Devuelve ``{"answer": str|None, "start": int|None, "score": float}``."""
        feats = self._features({"id": [example_id], "question": [question], "context": [context]})
        batch = self.tokenizer.pad(
            {"input_ids": feats["input_ids"], "attention_mask": feats["attention_mask"]},
            return_tensors="pt",
        )
        with self._lock, torch.inference_mode():
            out = self.model(**batch)

        padded_len = batch["input_ids"].shape[1]
        windows = []
        for i, offsets in enumerate(feats["offset_mapping"]):
            # Los offsets se rellenan con None hasta la longitud del lote, igual que los logits.
            windows.append(
                {
                    "start_logits": out.start_logits[i].numpy(),
                    "end_logits": out.end_logits[i].numpy(),
                    "offsets": offsets + [None] * (padded_len - len(offsets)),
                }
            )

        pred = predict_example(
            windows, context, tau=self.tau, n_best=N_BEST, max_answer_len=MAX_ANSWER_LEN
        )
        answered = pred["text"] != ""
        return {
            "answer": pred["text"] if answered else None,
            "start": pred["start"] if answered else None,
            "score": round(_sigmoid(pred["score"]), 4),
        }
