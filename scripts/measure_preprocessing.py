"""Mide el preprocesamiento de SQuAD 2.0 antes de decidir sus detalles.

Responde con números:
1. ¿Cuánto tarda preprocesar el train completo?
2. ¿Cuántas ventanas genera y cuántos ejemplos tienen más de una?
3. ¿Cuántas respuestas caen a mitad de token (cuando la respuesta termina o empieza
   dentro de una subpalabra)? Esto decide la opción A o B de la Fase 2.
4. ¿Cuánto ahorra el padding dinámico frente a rellenar todo a 384?
5. ¿Alguna respuesta answerable no queda completa en ninguna ventana?

Uso:
    .venv\\Scripts\\python.exe scripts\\measure_preprocessing.py
Guarda el resumen en logs/preprocessing_stats.json.
"""

import json
import sys
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # para importar src/ desde scripts/

from src.features import TrainFeatures  # noqa: E402

MODEL_ID = "distilbert-base-uncased"
MAX_LENGTH = 384
STRIDE = 128
BATCH = 2000


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    squad = load_dataset("rajpurkar/squad_v2")
    train = squad["train"]
    feat = TrainFeatures(tok, max_length=MAX_LENGTH, stride=STRIDE)

    # 1) Tiempo del preprocesamiento completo, como lo haría datasets.map.
    # load_from_cache_file=False obliga a recalcularlo: si no, el tiempo sale de la caché.
    t0 = time.perf_counter()
    mapped = train.map(
        feat,
        batched=True,
        remove_columns=train.column_names,
        load_from_cache_file=False,
        desc="train features",
    )
    elapsed = round(time.perf_counter() - t0, 1)

    # 2 a 5) Recorremos el train en lotes para medir alineación y ventanas.
    n_examples = len(train)
    n_windows = 0
    examples_multi_window = 0
    answerable = 0
    answerable_no_window = 0
    mid_token_end = 0  # la respuesta termina dentro de un token
    mid_token_start = 0  # la respuesta empieza dentro de un token
    total_real_tokens = 0
    for lo in range(0, n_examples, BATCH):
        batch = train[lo : lo + BATCH]
        enc = feat.tokenize(batch)
        windows = len(enc["input_ids"])
        n_windows += windows
        seq_ids = [enc.sequence_ids(i) for i in range(windows)]
        sample_map = enc["overflow_to_sample_mapping"]
        offsets = enc["offset_mapping"]
        total_real_tokens += sum(len(ids) for ids in enc["input_ids"])

        # Ventanas por ejemplo, para contar cuántos tienen más de una.
        per_example = {}
        for ex_idx in sample_map:
            per_example[ex_idx] = per_example.get(ex_idx, 0) + 1
        examples_multi_window += sum(1 for c in per_example.values() if c > 1)

        # Para cada respuesta answerable, buscamos la ventana que la contiene completa.
        found = {}
        for w, ex_idx in enumerate(sample_map):
            ans = batch["answers"][ex_idx]
            if len(ans["text"]) == 0:
                continue
            start_char = ans["answer_start"][0]
            end_char = start_char + len(ans["text"][0])
            ctx = [i for i, sid in enumerate(seq_ids[w]) if sid == 1]
            if not ctx:
                continue
            win_start = offsets[w][ctx[0]][0]
            win_end = offsets[w][ctx[-1]][1]
            if not (win_start <= start_char and end_char <= win_end):
                continue
            # La respuesta cabe completa en esta ventana. Medimos la alineación directamente:
            # el inicio debe coincidir con el comienzo de algún token del contexto, y el fin
            # con el final de algún token. Así la medición no depende de la regla que estamos
            # evaluando (la función de etiquetas ya usa esta misma regla).
            found[ex_idx] = True
            token_starts = {offsets[w][i][0] for i in ctx}
            token_ends = {offsets[w][i][1] for i in ctx}
            if start_char not in token_starts:
                mid_token_start += 1
            if end_char not in token_ends:
                mid_token_end += 1

        for ex_idx, ans in enumerate(batch["answers"]):
            if len(ans["text"]) > 0:
                answerable += 1
                if ex_idx not in found:
                    answerable_no_window += 1

    # Padding fijo a 384 contra longitud real: qué fracción del cómputo es útil.
    fixed_tokens = n_windows * MAX_LENGTH
    summary = {
        "model": MODEL_ID,
        "max_length": MAX_LENGTH,
        "stride": STRIDE,
        "examples": n_examples,
        "windows": n_windows,
        "windows_per_example": round(n_windows / n_examples, 4),
        "examples_with_more_than_one_window": examples_multi_window,
        "map_seconds": elapsed,
        "mapped_rows": len(mapped),
        "answerable": answerable,
        "answerable_without_full_window": answerable_no_window,
        "answers_end_mid_token": mid_token_end,
        "answers_start_mid_token": mid_token_start,
        "real_tokens_fraction_of_fixed_padding": round(total_real_tokens / fixed_tokens, 4),
    }
    for k, v in summary.items():
        print(f"{k:>42}: {v}")

    out = ROOT / "logs" / "preprocessing_stats.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
