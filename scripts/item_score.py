"""Puntaje por ítem sobre dev_select, con la escala de la guía (sección XIV.1).

Por qué existe
--------------
La nota en vivo no es F1. Cada pregunta vale 1, 0.5 o 0, y el equipo recibe el
promedio. Un ``null`` correcto cuenta igual que una respuesta exacta, y un
``null`` incorrecto cuenta como fallo. Este script calcula esa escala sobre
``dev_select`` para ver dónde quedan nuestros modelos frente al ancla de 0.70.

Limitación importante
---------------------
El juez LLM de la guía no está disponible aquí. El crédito parcial (0.5) se
asigna con una regla: una predicción con solapamiento de tokens F1 > 0 frente a
la referencia, pero que no es exacta, cuenta como parcial. Es un proxy. Por eso
el script reporta también el puntaje sin crédito parcial (solo exactas), para
acotar el resultado entre dos cotas.

Reponderación
-------------
Igual que en las corridas anteriores, el puntaje se reporta con 25 % de
preguntas sin respuesta (``0.75 * HasAns + 0.25 * NoAns``), porque es la
proporción esperada en el conjunto oculto. La versión sin reponderar sirve de
referencia.

Control de cordura
------------------
Con τ = +∞ el modelo responde siempre ``null``. El puntaje reponderado debe dar
0.25, que es el baseline de la guía. Si no da 0.25, el cálculo está mal.

Protocolo
---------
Solo usa ``dev_select``. ``dev_report`` no se toca aquí.

Uso
---
    .venv\\Scripts\\python.exe scripts\\item_score.py --run-name full_data_r16_ep2 --checkpoint epoch_2

Salida: tabla en consola y ``logs/runs/<run>/item_score_<checkpoint>.json``.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # para importar src/ desde scripts/
sys.path.insert(0, str(ROOT / "scripts"))  # para reusar load_model y TAU_GRID

from sweep_tau import TAU_GRID, load_model  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from src.data import eval_features, load_squad  # noqa: E402
from src.metrics import exact_match, f1, normalize_answer  # noqa: E402
from src.postprocess import predict_example  # noqa: E402
from src.splits import split_indices  # noqa: E402
from src.train import MODEL_ID, predict_logits  # noqa: E402

SHARE_NULL = 0.25  # proporción de preguntas sin respuesta en el conjunto oculto


def item_score(pred, golds, partial_credit=True):
    """Puntaje de una pregunta según la escala de la guía.

    - Sin respuesta en la referencia: 1 si la predicción es ``null``, 0 si no.
    - Con respuesta: 0 si la predicción es ``null``. 1 si coincide exacto con alguna
      referencia. 0.5 si solapa parcialmente (solo con ``partial_credit``). 0 si no.
    """
    if not golds:
        return float(normalize_answer(pred) == "")
    if normalize_answer(pred) == "":
        return 0.0
    if max(exact_match(pred, g) for g in golds):
        return 1.0
    if partial_credit and max(f1(pred, g) for g in golds) > 0:
        return 0.5
    return 0.0


def summarize(items, tau, null_count, n):
    """Promedios por tipo de pregunta y puntaje con y sin reponderación."""
    has = [s for s in items if s["has_answer"]]
    no = [s for s in items if not s["has_answer"]]
    has_score = sum(s["score"] for s in has) / len(has)
    no_score = sum(s["score"] for s in no) / len(no)
    has_exact = sum(s["exact"] for s in has) / len(has)
    return {
        "tau": tau,
        "null_pct": round(100 * null_count / n, 2),
        "score_raw": round(sum(s["score"] for s in items) / n, 4),
        "score_reweighted": round((1 - SHARE_NULL) * has_score + SHARE_NULL * no_score, 4),
        "score_exact_only_reweighted": round((1 - SHARE_NULL) * has_exact + SHARE_NULL * no_score, 4),
        "HasAns_score": round(has_score, 4),
        "NoAns_score": round(no_score, 4),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Puntaje por ítem (escala de la guía) sobre dev_select.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", required=True, help="epoch_<n> o merged")
    args = p.parse_args(argv)

    raw = load_squad()
    dev = raw["validation"]
    select_idx, _ = split_indices(dev["title"])
    dev_select = dev.select(select_idx)
    raw_by_id = {ex["id"]: {"context": ex["context"], "answers": ex["answers"]} for ex in dev_select}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_model(args.run_name, args.checkpoint)
    enc = eval_features(tokenizer, dev_select)

    logits = predict_logits(model, tokenizer, enc)
    windows_by_id = defaultdict(list)
    for i, ex_id in enumerate(enc["example_id"]):
        start, end = logits[i]
        windows_by_id[ex_id].append(
            {"start_logits": start, "end_logits": end, "offsets": enc["offset_mapping"][i]}
        )

    n = len(raw_by_id)
    rows = []
    print(f"{'tau':>10} | {'raw':>6} {'reweight':>8} {'exact-only':>10} | {'HasAns':>6} {'NoAns':>6} | {'null%':>6}")
    for tau in TAU_GRID:
        items, null_count = [], 0
        for ex_id, ex in raw_by_id.items():
            pred = predict_example(windows_by_id[ex_id], ex["context"], tau=tau)["text"]
            golds = ex["answers"]["text"]
            null_count += pred == ""
            items.append(
                {
                    "has_answer": bool(golds),
                    "score": item_score(pred, golds, partial_credit=True),
                    "exact": item_score(pred, golds, partial_credit=False),
                }
            )
        row = summarize(items, tau, null_count, n)
        rows.append(row)
        label = f"{tau:.1e}" if abs(tau) >= 1e6 else f"{tau:g}"
        print(f"{label:>10} | {row['score_raw']:>6} {row['score_reweighted']:>8} "
              f"{row['score_exact_only_reweighted']:>10} | {row['HasAns_score']:>6} {row['NoAns_score']:>6} | "
              f"{row['null_pct']:>6}", flush=True)

    always_null = rows[-1]["score_reweighted"]
    print(f"\nControl: siempre null (tau = +inf) reponderado = {always_null} (debe ser 0.25)")

    out = ROOT / "logs" / "runs" / args.run_name / f"item_score_{args.checkpoint}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf8")
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
