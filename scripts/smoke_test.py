"""Prueba de humo que imita al harness de la evaluación en vivo.

Hace N peticiones secuenciales a ``/predict`` con timeout de 5 s por petición y
cronometra el total contra el tope de 150 s. Los ítems salen de
``public_sample.json`` (5 preguntas repetidas, con ids nuevos), así que no
evalúa calidad: comprueba forma, latencia, autenticación y offsets.

Uso (la variable del token no va en la línea de comandos):

    $env:QA_API_TOKEN = "..."   # PowerShell
    python scripts/smoke_test.py --base-url https://<tunel> --n 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

SAMPLE = Path(__file__).resolve().parent.parent / "public_sample.json"
PER_REQUEST_TIMEOUT_S = 5.0
GLOBAL_BUDGET_S = 150.0


def check_item(response, item_id, context):
    """Devuelve una lista de problemas; vacía si la respuesta cumple el contrato."""
    if response.status_code != 200:
        return [f"status {response.status_code}"]
    try:
        body = response.json()
    except ValueError:
        return ["JSON mal formado"]
    problems = []
    if body.get("id") != item_id:
        problems.append(f"id distinto: {body.get('id')!r}")
    answer, start = body.get("answer"), body.get("start")
    if answer is None:
        if start is not None:
            problems.append("start debe ser null cuando answer es null")
    elif not isinstance(start, int) or context[start : start + len(answer)] != answer:
        problems.append("context[start:start+len] no coincide con answer")
    return problems


def main():
    """Revisa /health y /info, lanza N peticiones a /predict y sale con código 1 si algo falla."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    token = os.environ.get("QA_API_TOKEN", "")
    if not token:
        sys.exit("Falta QA_API_TOKEN en el entorno.")
    base = args.base_url.rstrip("/")
    auth = {"Authorization": f"Bearer {token}"}
    items = json.loads(SAMPLE.read_text(encoding="utf-8"))["items"]

    with httpx.Client(base_url=base, timeout=PER_REQUEST_TIMEOUT_S) as client:
        health = client.get("/health")
        print(f"/health -> {health.status_code} {health.text}")
        info = client.get("/info", headers=auth)
        print(f"/info   -> {info.status_code} {info.text}")
        no_token = client.get("/info")
        print(f"/info sin token -> {no_token.status_code} (debe ser 401)")

        failures, slowest_ms = 0, 0.0
        started = time.perf_counter()
        for i in range(args.n):
            item = items[i % len(items)]
            item_id = f"smoke-{i:03d}"
            body = {"id": item_id, "context": item["context"], "question": item["question"]}
            t0 = time.perf_counter()
            try:
                response = client.post("/predict", json=body, headers=auth)
                problems = check_item(response, item_id, item["context"])
            except httpx.HTTPError as exc:
                problems = [f"error de red o timeout: {type(exc).__name__}"]
            elapsed_ms = (time.perf_counter() - t0) * 1000
            slowest_ms = max(slowest_ms, elapsed_ms)
            if problems:
                failures += 1
                print(f"[FALLA] {item_id}: {'; '.join(problems)}")
        total_s = time.perf_counter() - started

    print(f"\n{args.n} peticiones en {total_s:.1f} s (tope {GLOBAL_BUDGET_S:.0f} s), "
          f"la más lenta {slowest_ms:.0f} ms, fallas {failures}")
    ok = failures == 0 and total_s < GLOBAL_BUDGET_S and no_token.status_code == 401
    print("RESULTADO:", "OK" if ok else "REVISAR")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
