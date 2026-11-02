"""Barrido del umbral τ de no respuesta sobre dev_select, para una corrida ya entrenada.

Por qué existe
--------------
La decisión "responder o null" depende de τ. Un modelo puede tener buenos spans
y aun así parecer malo si τ está mal ubicado. Este script calcula los logits una
sola vez y luego evalúa muchos valores de τ, para separar dos diagnósticos:

- Si HasAns F1 sube al bajar τ, el problema era el umbral.
- Si HasAns F1 sigue bajo con cualquier τ, el problema es el modelo.

Protocolo
---------
Solo usa ``dev_select``. ``dev_report`` no se toca aquí.

Uso
---
    .venv\\Scripts\\python.exe scripts\\sweep_tau.py --run-name piloto_r8 --checkpoint epoch_1
    .venv\\Scripts\\python.exe scripts\\sweep_tau.py --run-name piloto_r8 --checkpoint merged

``--checkpoint`` acepta ``epoch_<n>`` (adaptador LoRA, se carga sobre el modelo
base) o ``merged`` (modelo completo).

Salida: tabla en consola y ``logs/runs/<run>/tau_sweep_<checkpoint>.json``.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # para importar src/ desde scripts/

from src.data import eval_features, load_squad  # noqa: E402
from src.metrics import aggregate, score_example  # noqa: E402
from src.postprocess import predict_example  # noqa: E402
from src.splits import split_indices  # noqa: E402
from src.train import MODEL_ID, predict_logits  # noqa: E402

# Valores de τ a probar. Van de "siempre span" (muy negativo) a "casi siempre null".
TAU_GRID = [-1e9, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 1e9]


def load_model(run_name, checkpoint):
    """Carga el modelo de una corrida: adaptador LoRA sobre la base, o modelo completo.

    Una carpeta con ``adapter_config.json`` es un adaptador LoRA. Cualquier otra
    (``merged`` o un ``epoch_<n>`` de fine-tuning completo) es un modelo entero.
    """
    model_dir = ROOT / "models" / run_name / checkpoint
    if (model_dir / "adapter_config.json").exists():
        base = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID)
        return PeftModel.from_pretrained(base, model_dir)
    return AutoModelForQuestionAnswering.from_pretrained(model_dir)


def main(argv=None):
    p = argparse.ArgumentParser(description="Barrido de τ sobre dev_select.")
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

    # Los logits se calculan una sola vez; τ solo cambia la decisión.
    logits = predict_logits(model, tokenizer, enc)
    windows_by_id = defaultdict(list)
    for i, ex_id in enumerate(enc["example_id"]):
        start, end = logits[i]
        windows_by_id[ex_id].append(
            {"start_logits": start, "end_logits": end, "offsets": enc["offset_mapping"][i]}
        )

    rows = []
    print(f"{'tau':>10} | {'EM':>6} {'F1':>6} | {'HasAns_F1':>9} | {'NoAns_acc':>9} | {'null%':>6}")
    for tau in TAU_GRID:
        scores, null_count = [], 0
        for ex_id, ex in raw_by_id.items():
            pred = predict_example(windows_by_id[ex_id], ex["context"], tau=tau)
            null_count += pred["text"] == ""
            scores.append(score_example(pred["text"], ex["answers"]["text"]))
        summary = aggregate(scores)
        summary["tau"] = tau
        summary["null_pct"] = round(100 * null_count / len(raw_by_id), 2)
        rows.append(summary)
        label = f"{tau:.1e}" if abs(tau) >= 1e6 else f"{tau:g}"
        print(f"{label:>10} | {summary['EM']:>6} {summary['F1']:>6} | {summary['HasAns_F1']:>9} | "
              f"{summary['NoAns_acc']:>9} | {summary['null_pct']:>6}", flush=True)

    out = ROOT / "logs" / "runs" / args.run_name / f"tau_sweep_{args.checkpoint}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf8")
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
