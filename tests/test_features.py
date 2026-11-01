"""Pruebas del preprocesamiento de QA (src/features.py).

Usan el tokenizer real de DistilBERT, porque los offsets dependen de su
normalización. Son pruebas de comportamiento: cada una describe un caso que el
servicio tiene que manejar bien.
"""

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from src.features import CLS_INDEX, TrainFeatures, ValidationFeatures, make_collator

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "distilbert-base-uncased"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def public_items():
    return {i["id"]: i for i in json.loads((ROOT / "public_sample.json").read_text(encoding="utf8"))["items"]}


def as_example(item):
    """Convierte un ítem de public_sample al formato de SQuAD que espera el preprocesamiento."""
    answers = {"text": [], "answer_start": []}
    for a in item["answers"]:
        answers["text"].append(a["text"])
        answers["answer_start"].append(a["start"])
    return {"id": item["id"], "question": item["question"], "context": item["context"], "answers": answers}


def window_span_text(enc, w, start_pos, end_pos, context):
    """Texto que recuperaría la post-procesamiento a partir de las etiquetas de la ventana."""
    offsets = enc["offset_mapping"][w]
    return context[offsets[start_pos][0] : offsets[end_pos][1]]


def run_train(tok, example):
    feat = TrainFeatures(tok)
    out = feat({k: [v] for k, v in example.items()})
    return out


# --- Etiquetas: el texto decodificado debe ser la respuesta -----------------


def test_label_span_decodes_to_the_answer(tok, public_items):
    """Caso general: en cada ventana con etiqueta, el span recuperado es la respuesta."""
    item = public_items["sample-01"]
    example = as_example(item)
    feat = TrainFeatures(tok)
    enc = tok(
        [example["question"]], [example["context"]],
        truncation="only_second", max_length=384, stride=128,
        return_overflowing_tokens=True, return_offsets_mapping=True,
    )
    out = feat({k: [v] for k, v in example.items()})
    for w, (s, e) in enumerate(zip(out["start_positions"], out["end_positions"])):
        if s == CLS_INDEX:
            continue
        recovered = window_span_text(enc, w, s, e, example["context"])
        assert recovered == item["answers"][0]["text"]


# --- Sin respuesta: todas las ventanas al [CLS] --------------------------


def test_unanswerable_all_windows_point_to_cls(tok, public_items):
    example = as_example(public_items["sample-03"])
    out = run_train(tok, example)
    assert all(s == CLS_INDEX for s in out["start_positions"])
    assert all(e == CLS_INDEX for e in out["end_positions"])


# --- Contexto largo: solo la ventana con la respuesta está etiquetada ------


def test_long_context_only_the_window_with_answer_is_labeled(tok, public_items):
    """sample-05 genera 3 ventanas; la respuesta está en la última."""
    example = as_example(public_items["sample-05"])
    feat = TrainFeatures(tok)
    out = feat({k: [v] for k, v in example.items()})
    labeled = [w for w, s in enumerate(out["start_positions"]) if s != CLS_INDEX]
    assert len(out["start_positions"]) == 3
    assert labeled == [2]


# --- Respuesta a mitad de token: opción B, se marca como [CLS] -------------


MID_TOKEN_CONTEXT = "The capital is Tegucigalpa, in Honduras."


def test_answer_ending_mid_token_is_labeled_cls(tok):
    """'Tegucig' termina dentro del token '##gal' de 'Tegucigalpa': el token completo no coincide."""
    answer = "Tegucig"
    example = {
        "id": "mid-token",
        "question": "What is the capital?",
        "context": MID_TOKEN_CONTEXT,
        "answers": {"text": [answer], "answer_start": [MID_TOKEN_CONTEXT.index(answer)]},
    }
    # La prueba solo tiene sentido si el tokenizer parte la palabra. Lo comprobamos.
    enc = tok(example["question"], MID_TOKEN_CONTEXT, return_offsets_mapping=True)
    end_char = MID_TOKEN_CONTEXT.index(answer) + len(answer)
    token_ends = [b for a, b in enc["offset_mapping"] if b is not None and a < end_char <= b]
    assert token_ends and token_ends[0] != end_char, "el tokenizer no partió la palabra como se esperaba"

    out = run_train(tok, example)
    assert out["start_positions"] == [CLS_INDEX]
    assert out["end_positions"] == [CLS_INDEX]


def test_answer_on_full_word_is_labeled(tok):
    """Control: la palabra completa, con los mismos tokens, sí debe etiquetarse."""
    answer = "Tegucigalpa"
    example = {
        "id": "full-word",
        "question": "What is the capital?",
        "context": MID_TOKEN_CONTEXT,
        "answers": {"text": [answer], "answer_start": [MID_TOKEN_CONTEXT.index(answer)]},
    }
    out = run_train(tok, example)
    assert out["start_positions"][0] != CLS_INDEX
    assert out["end_positions"][0] != CLS_INDEX


# --- Validación: offsets de la pregunta y los especiales quedan en None ----


def test_validation_offsets_only_for_context_tokens(tok, public_items):
    feat = ValidationFeatures(tok)
    example = as_example(public_items["sample-01"])
    out = feat({k: [v] for k, v in example.items()})
    for window_offsets in out["offset_mapping"]:
        # El primer token es [CLS] y el último [SEP]: ninguno pertenece al contexto.
        assert window_offsets[0] is None
        assert window_offsets[-1] is None
        # Debe haber al menos un offset del contexto.
        assert any(o is not None for o in window_offsets)


def test_validation_keeps_every_window_and_its_example_id(tok, public_items):
    feat = ValidationFeatures(tok)
    example = as_example(public_items["sample-05"])
    out = feat({k: [v] for k, v in example.items()})
    assert len(out["example_id"]) == 3
    assert set(out["example_id"]) == {"sample-05"}


# --- Padding dinámico: el collator rellena al lote, no a 384 ---------------


def test_dynamic_padding_pads_to_longest_in_batch_not_384(tok, public_items):
    """Dos ventanas cortas: el lote debe quedar a la longitud de la más larga, no a 384."""
    feat = TrainFeatures(tok)
    features = []
    for item_id in ["sample-01", "sample-03"]:
        out = feat({k: [v] for k, v in as_example(public_items[item_id]).items()})
        features.append({
            "input_ids": out["input_ids"][0],
            "attention_mask": out["attention_mask"][0],
            "start_positions": out["start_positions"][0],
            "end_positions": out["end_positions"][0],
        })
    batch = make_collator(tok)(features)
    longest = max(len(f["input_ids"]) for f in features)
    assert batch["input_ids"].shape == (2, longest)
    assert longest < 384
    assert batch["start_positions"].dtype == torch.long
