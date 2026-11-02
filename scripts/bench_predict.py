"""Mide la latencia de ``predict`` en CPU y muestra las respuestas del set público.

Uso (desde la raíz del repo, con el modelo en ``QA_MODEL_DIR``):

    python scripts/bench_predict.py --repeats 5

Carga el predictor igual que el servicio (CPU, sin GPU) y reporta la media, el
p95 y el máximo por pregunta. No toca dev_select ni dev_report.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.config import DEFAULT_MODEL_DIR, DEFAULT_TAU  # noqa: E402
from service.qa import QAPredictor  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "public_sample.json"


def main():
    """Carga el predictor, calienta una vez y mide ``--repeats`` veces cada pregunta del set público."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("QA_MODEL_DIR", DEFAULT_MODEL_DIR))
    parser.add_argument("--tau", type=float, default=float(os.environ.get("QA_TAU", DEFAULT_TAU)))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    load_start = time.perf_counter()
    predictor = QAPredictor(args.model_dir, tau=args.tau, tokenizer_dir=os.environ.get("QA_TOKENIZER_DIR"))
    load_s = time.perf_counter() - load_start
    size_mb = sum(p.stat().st_size for p in Path(args.model_dir).glob("*.safetensors")) / 1e6
    print(f"modelo: {args.model_dir} | params={predictor.params:,} | size={size_mb:.0f} MB | load={load_s:.1f} s")

    items = json.loads(SAMPLE.read_text(encoding="utf-8"))["items"]
    predictor.predict(items[0]["id"], items[0]["context"], items[0]["question"])  # calentamiento

    all_ms = []
    for it in items:
        times, pred = [], None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            pred = predictor.predict(it["id"], it["context"], it["question"])
            times.append((time.perf_counter() - t0) * 1000)
        all_ms.extend(times)
        times.sort()
        p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
        print(
            f"{it['id']} ctx={len(it['context'])}ch media={statistics.mean(times):.0f} ms "
            f"p95={p95:.0f} ms max={times[-1]:.0f} ms | answer={pred['answer']!r} start={pred['start']} score={pred['score']}"
        )

    all_ms.sort()
    print(
        f"global: media={statistics.mean(all_ms):.0f} ms p95={all_ms[int(0.95 * len(all_ms))]:.0f} ms "
        f"max={all_ms[-1]:.0f} ms (n={len(all_ms)})"
    )


if __name__ == "__main__":
    main()
