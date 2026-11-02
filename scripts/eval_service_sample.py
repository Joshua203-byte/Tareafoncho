"""Compara el servicio con la ruta de evaluación del entrenamiento y simula sets de 50 preguntas.

Qué hace
--------
1. Toma una muestra de ``dev_select`` (nunca ``dev_report``, que se usó una sola vez):
   ``--n`` preguntas con ``--null-share`` sin respuesta, como el set oculto (≈ 25 %).
2. Pasa cada pregunta por ``QAPredictor`` (el mismo código que sirve la API) y también
   por la ruta de evaluación de ``src/train.py``. Cuenta las diferencias: debe ser 0.
3. Comprueba el contrato: ``context[start:start+len(answer)] == answer`` y que
   ``start`` sea ``None`` solo si ``answer`` lo es.
4. Mide la latencia por pregunta.
5. Calcula el puntaje con la escala de la guía y lo repite en miles de sets de 50
   preguntas (38 con respuesta y 12 sin), para ver cuánto varía un set oculto por azar.

Cómo leer el resultado
----------------------
``dev_select`` se usó para elegir la época y τ, así que su puntaje es algo optimista
(sesgo de selección). La cifra limpia del modelo es la de ``dev_report``. Lo útil de este
script es la comparación servicio vs. referencia y la dispersión de los sets de 50.

Uso (desde la raíz del repo; el modelo sale de QA_MODEL_DIR / QA_TOKENIZER_DIR / QA_TAU):

    python scripts/eval_service_sample.py --n 1000
"""

import argparse
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # para importar service/ y src/ desde scripts/

import numpy as np  # noqa: E402
from transformers import AutoModelForQuestionAnswering, AutoTokenizer  # noqa: E402

from service.config import BASE_MODEL_ID, DEFAULT_MODEL_DIR, DEFAULT_TAU  # noqa: E402
from service.qa import QAPredictor  # noqa: E402
from src.data import eval_features, load_squad  # noqa: E402
from src.metrics import item_score  # noqa: E402
from src.postprocess import predict_example  # noqa: E402
from src.splits import split_indices  # noqa: E402
from src.train import predict_logits  # noqa: E402

SET_SIZE = 50  # tamaño del set oculto
SET_NULL = 12  # 12 de 50 sin respuesta = 24 %, lo más cerca de 25 % que permite un entero
BASELINE, REFERENCE = 0.25, 0.70  # anclas de la guía (sección XIV.1)


def sample_dev_select(n, null_share, seed):
    """Muestra reproducible de ``dev_select`` con la proporción pedida de preguntas sin respuesta."""
    dev = load_squad()["validation"]
    select_idx, _ = split_indices(dev["title"])  # solo dev_select: dev_report no se toca
    dev_select = dev.select(select_idx)
    has = [i for i, ex in enumerate(dev_select) if ex["answers"]["text"]]
    no = [i for i, ex in enumerate(dev_select) if not ex["answers"]["text"]]
    n_no = round(n * null_share)
    rng = random.Random(seed)
    return dev_select.select(sorted(rng.sample(has, n - n_no) + rng.sample(no, n_no)))


def reference_windows(model, tokenizer, examples):
    """Ventanas con logits calculados como en la evaluación del entrenamiento (``predict_logits``)."""
    enc = eval_features(tokenizer, examples)
    logits = predict_logits(model, tokenizer, enc)
    windows = defaultdict(list)
    for i, ex_id in enumerate(enc["example_id"]):
        start, end = logits[i]
        windows[ex_id].append({"start_logits": start, "end_logits": end, "offsets": enc["offset_mapping"][i]})
    return windows


def main():
    """Corre la comparación y escribe el resumen en pantalla. Sale con código 1 si hay diferencias."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", default=os.environ.get("QA_MODEL_DIR", DEFAULT_MODEL_DIR))
    parser.add_argument("--tokenizer-dir", default=os.environ.get("QA_TOKENIZER_DIR") or None)
    parser.add_argument("--tau", type=float, default=float(os.environ.get("QA_TAU", DEFAULT_TAU)))
    parser.add_argument("--n", type=int, default=1000, help="preguntas de dev_select a evaluar")
    parser.add_argument("--null-share", type=float, default=0.25)
    parser.add_argument("--sets", type=int, default=5000, help="cuántos sets de 50 simular")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    examples = sample_dev_select(args.n, args.null_share, args.seed)
    print(f"preguntas: {len(examples)}, sin respuesta: {sum(1 for ex in examples if not ex['answers']['text'])}")

    # Tokenizer: el pedido, o el de la carpeta del modelo si lo trae, o el del checkpoint base.
    tok_src = args.tokenizer_dir or (args.model_dir if (Path(args.model_dir) / "tokenizer.json").exists() else BASE_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir).eval()
    reference = reference_windows(model, tokenizer, examples)
    service = QAPredictor(args.model_dir, args.tau, tokenizer_dir=tok_src)

    differences = invalid = 0
    times, rows = [], []
    for ex in examples:
        t0 = time.perf_counter()
        served = service.predict(ex["id"], ex["context"], ex["question"])
        times.append((time.perf_counter() - t0) * 1000)
        ref = predict_example(reference[ex["id"]], ex["context"], tau=args.tau)
        if (ref["text"] or None) != served["answer"] or ref["start"] != served["start"]:
            differences += 1
        answer, start = served["answer"], served["start"]
        if (answer is None) != (start is None):
            invalid += 1
        elif answer is not None and ex["context"][start : start + len(answer)] != answer:
            invalid += 1
        golds = ex["answers"]["text"]
        rows.append((bool(golds), item_score(answer or "", golds, True), item_score(answer or "", golds, False)))

    times.sort()
    print(f"servicio vs referencia: {differences} diferencias de {len(examples)} | offsets/null inválidos: {invalid}")
    print(f"latencia por pregunta: media {statistics.mean(times):.0f} ms, p95 {times[int(0.95 * len(times))]:.0f} ms, máx {times[-1]:.0f} ms")

    has_rows = [r for r in rows if r[0]]
    no_rows = [r for r in rows if not r[0]]
    has_score = np.mean([r[1] for r in has_rows])
    no_score = np.mean([r[1] for r in no_rows])
    print(f"HasAns {has_score:.3f} (solo exactas {np.mean([r[2] for r in has_rows]):.3f}) | NoAns {no_score:.3f}")
    print(f"puntaje reponderado a 25 % sin respuesta: {0.75 * has_score + 0.25 * no_score:.3f}")

    # Sets de 50 como el oculto: cada set es un sorteo distinto de preguntas.
    rng = random.Random(args.seed + 4)
    sets = np.array(
        [np.mean([r[1] for r in rng.sample(has_rows, SET_SIZE - SET_NULL) + rng.sample(no_rows, SET_NULL)]) for _ in range(args.sets)]
    )
    points = np.clip((sets - BASELINE) / (REFERENCE - BASELINE), 0, 1) * 60  # fórmula de la guía
    print(f"sets de 50: puntaje medio {sets.mean():.3f}, desv {sets.std():.3f}, p5 {np.percentile(sets, 5):.3f}, p95 {np.percentile(sets, 95):.3f}")
    print(f"puntos de 60: media {points.mean():.1f}, p5 {np.percentile(points, 5):.1f}, p95 {np.percentile(points, 95):.1f}")
    sys.exit(1 if differences or invalid else 0)


if __name__ == "__main__":
    main()
