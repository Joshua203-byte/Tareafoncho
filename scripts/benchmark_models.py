"""Compara los candidatos de encoder: parámetros, arquitectura y latencia en CPU.

No descarga pesos: solo config.json. Los modelos se crean con pesos aleatorios
porque la latencia depende de la arquitectura, no de los valores de los pesos.
Los parámetros cuentan la cabeza de QA (start/end) que añadiremos en la Fase 3.

Uso:
    .venv\\Scripts\\python.exe scripts\\benchmark_models.py
Guarda el resultado en logs/model_benchmark.json.
"""

import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForQuestionAnswering

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "microsoft/deberta-v3-base",
    "answerdotai/ModernBERT-base",
]
SEQ_LEN = 384  # la longitud de ventana que usaremos
RUNS = 10  # medimos 10 veces y tomamos la mediana, para no depender de un pico


def describe(model_id):
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForQuestionAnswering.from_config(config).eval()
    params = sum(p.numel() for p in model.parameters())
    return model, {
        "model": model_id,
        "params_million": round(params / 1e6, 1),
        "layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "attention_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "architecture": config.model_type,
    }


def latency_ms(model):
    ids = torch.randint(low=1000, high=20000, size=(1, SEQ_LEN))
    mask = torch.ones_like(ids)
    times = []
    with torch.inference_mode():
        model(input_ids=ids, attention_mask=mask)  # calentamiento
        for _ in range(RUNS):
            t0 = time.perf_counter()
            model(input_ids=ids, attention_mask=mask)
            times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 1)


def main():
    print(f"torch {torch.__version__} | threads: {torch.get_num_threads()} | CPU forward, batch 1, seq {SEQ_LEN}")
    rows = []
    for model_id in CANDIDATES:
        try:
            model, info = describe(model_id)
            info["latency_ms_cpu_median"] = latency_ms(model)
            print(info)
            rows.append(info)
            del model
        except Exception as exc:  # un candidato roto no debe tumbar la comparación
            print(f"{model_id}: ERROR {type(exc).__name__}: {exc}")
            rows.append({"model": model_id, "error": f"{type(exc).__name__}: {exc}"})

    out = ROOT / "logs" / "model_benchmark.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf8")
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
