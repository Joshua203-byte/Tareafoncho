"""Pruebas del modelo: conteo de entrenables y equivalencia tras el merge de LoRA.

La prueba de merge es la más importante: si fusionar cambia las predicciones,
el modelo que sirve la API no es el que entrenamos.
"""

import torch

from src.model import ATTENTION_MODULES, build_model, count_parameters

MODEL_ID = "distilbert-base-uncased"


def test_lora_trains_only_adapters_and_qa_head():
    model = build_model(MODEL_ID, lora_rank=8)
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert "lora_" in name or "qa_outputs" in name, f"entrena algo inesperado: {name}"


def test_lora_targets_only_attention_matrices():
    model = build_model(MODEL_ID, lora_rank=8)
    lora_layers = {n.split(".lora_")[0].split(".")[-1] for n, _ in model.named_parameters() if ".lora_A" in n}
    assert lora_layers == set(ATTENTION_MODULES)


def test_full_finetuning_trains_everything():
    model = build_model(MODEL_ID, lora_rank=0)
    counts = count_parameters(model)
    assert counts["trainable"] == counts["total"]


def test_merge_preserves_predictions():
    model = build_model(MODEL_ID, lora_rank=4)
    # lora_B arranca en cero, así que sin esto el merge sería trivial. Lo aleatorizamos
    # para que la prueba compare una actualización real.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, std=0.02)
    model.eval()
    ids = torch.randint(1000, 2000, (2, 20))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        before = model(input_ids=ids, attention_mask=mask)
        merged = model.merge_and_unload()
        after = merged(input_ids=ids, attention_mask=mask)
    assert torch.allclose(before.start_logits, after.start_logits, atol=1e-4)
    assert torch.allclose(before.end_logits, after.end_logits, atol=1e-4)
    assert not any("lora_" in n for n, _ in merged.named_parameters())
