"""Explora SQuAD 2.0 y el public_sample.json.

Responde a tres preguntas antes de entrenar:
1. ¿Qué proporción de preguntas no tiene respuesta?
2. ¿Cuántos ejemplos superan la ventana de 384 tokens que usaremos?
3. ¿Los offsets (answer_start) de los datos son consistentes con el texto?

Uso:
    .venv\\Scripts\\python.exe scripts\\explore_data.py
Guarda el resumen en logs/data_stats.json.
"""

import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_ID = "distilbert-base-uncased"  # mismo vocabulario que bert-base-uncased
MAX_SEQ_LENGTH = 384


def percentile(values, q):
    values = sorted(values)
    idx = min(len(values) - 1, int(q * len(values)))
    return values[idx]


def check_offsets(examples):
    """Cuenta respuestas cuyo texto no coincide con context[start:start+len]."""
    mismatches = 0
    total = 0
    for ex in examples:
        for text, start in zip(ex["answers"]["text"], ex["answers"]["answer_start"]):
            total += 1
            if ex["context"][start : start + len(text)] != text:
                mismatches += 1
    return {"answers_checked": total, "mismatches": mismatches}


def summarize_squad(split_name, ds, tok):
    n = len(ds)
    unanswerable = sum(1 for ex in ds if len(ex["answers"]["text"]) == 0)

    answer_chars = [len(a) for ex in ds for a in ex["answers"]["text"]]
    context_chars = [len(c) for c in ds["context"]]

    # Longitud real de entrada: pregunta + contexto, como la verá el modelo.
    enc = tok(list(ds["question"]), list(ds["context"]), truncation=False)
    input_lengths = [len(ids) for ids in enc["input_ids"]]
    over_window = sum(1 for L in input_lengths if L > MAX_SEQ_LENGTH)

    summary = {
        "examples": n,
        "unanswerable": unanswerable,
        "unanswerable_fraction": round(unanswerable / n, 4),
        "context_chars_median": percentile(context_chars, 0.5),
        "context_chars_p95": percentile(context_chars, 0.95),
        "answer_chars_median": percentile(answer_chars, 0.5),
        "input_tokens_median": percentile(input_lengths, 0.5),
        "input_tokens_p95": percentile(input_lengths, 0.95),
        f"examples_over_{MAX_SEQ_LENGTH}_tokens": over_window,
        "offsets": check_offsets(ds),
    }
    print(f"\n== {split_name} ==")
    for k, v in summary.items():
        print(f"{k:>32}: {v}")
    return summary


def summarize_public_sample(tok):
    data = json.loads((ROOT / "public_sample.json").read_text(encoding="utf8"))
    items = data["items"]
    print("\n== public_sample.json ==")
    report = []
    for item in items:
        n_tokens = len(tok(item["question"], item["context"])["input_ids"])
        # Cada respuesta de referencia debe cumplir el contrato: context[start:start+len] == text
        ok = all(
            item["context"][a["start"] : a["start"] + len(a["text"])] == a["text"]
            for a in item["answers"]
        )
        row = {
            "id": item["id"],
            "is_impossible": item["is_impossible"],
            "input_tokens": n_tokens,
            "over_window": n_tokens > MAX_SEQ_LENGTH,
            "offsets_ok": ok,
            "references": [a["text"] for a in item["answers"]],
        }
        report.append(row)
        print(row)
    return report


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    squad = load_dataset("rajpurkar/squad_v2")

    stats = {
        "tokenizer": TOKENIZER_ID,
        "max_seq_length": MAX_SEQ_LENGTH,
        "train": summarize_squad("train", squad["train"], tok),
        "validation": summarize_squad("validation", squad["validation"], tok),
        "public_sample": summarize_public_sample(tok),
    }

    # Baseline "siempre null": si la respuesta correcta es null, acierta; si no, falla.
    # Su puntaje esperado es la proporción de preguntas sin respuesta.
    stats["always_null_baseline"] = {
        "validation_score": stats["validation"]["unanswerable_fraction"],
        "note": "La guía fija 0.25 como baseline porque el set oculto tiene ~25 % sin respuesta.",
    }
    print("\nBaseline always-null en validation:", stats["always_null_baseline"]["validation_score"])

    out = ROOT / "logs" / "data_stats.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf8")
    print(f"\nResumen guardado en {out}")


if __name__ == "__main__":
    main()
