"""Evaluación final sobre dev_report. Se corre UNA sola vez por modelo elegido.

Por qué existe una sola vez
---------------------------
``dev_report`` es el número limpio que va al reporte. Si lo miramos para elegir
checkpoint o τ, deja de ser limpio. Por eso el script no sobrescribe: si el
archivo de salida ya existe, se niega a correr otra vez.

Protocolo
---------
Modelo y τ ya están fijados antes de correr esto (ver docs/experimentos_fase3.md).
No se usa ``dev_select`` aquí.

Uso
---
    .venv\\Scripts\\python.exe scripts\\eval_report.py --run-name full_ft_4ep --checkpoint epoch_3 --tau -2

Salida: ``logs/report/<run>_<checkpoint>_tau<tau>.json``.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # para importar src/ desde scripts/

from transformers import AutoModelForQuestionAnswering, AutoTokenizer  # noqa: E402

from src.data import eval_features, load_squad  # noqa: E402
from src.splits import split_indices  # noqa: E402
from src.train import MODEL_ID, SEED, evaluate_dev  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluación única sobre dev_report.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", required=True, help="epoch_<n> (modelo completo) o merged")
    p.add_argument("--tau", type=float, required=True)
    args = p.parse_args(argv)

    out = ROOT / "logs" / "report" / f"{args.run_name}_{args.checkpoint}_tau{args.tau:g}.json"
    if out.exists():
        sys.exit(f"Ya existe {out}. dev_report se evalúa una sola vez; no se vuelve a correr.")

    raw = load_squad()
    dev = raw["validation"]
    _, report_idx = split_indices(dev["title"], seed=SEED)
    dev_report = dev.select(report_idx)
    raw_by_id = {ex["id"]: {"context": ex["context"], "answers": ex["answers"]} for ex in dev_report}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model_dir = ROOT / "models" / args.run_name / args.checkpoint
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    enc = eval_features(tokenizer, dev_report)

    metrics, _ = evaluate_dev(model, tokenizer, enc, raw_by_id, args.tau)
    record = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "tau": args.tau,
        "split": "dev_report",
        "questions": len(dev_report),
        **metrics,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf8")
    for key in ("EM", "F1", "HasAns_F1", "NoAns_acc", "guide_raw", "guide_reweighted", "guide_exact_reweighted"):
        print(f"{key}: {record[key]}")
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
