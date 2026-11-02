"""Batería de casos raros contra una URL (local o del túnel): ¿alguna respuesta rompe el contrato?

Qué prueba (cada caso debe dar 200, JSON válido, ``id`` igual, ``score`` en [0, 1] y
``context[start:start+len(answer)] == answer``):

- Cabeceras de cliente: ``User-Agent`` de navegador (ngrok gratuito muestra una página de
  advertencia a los navegadores; no debe afectar a un cliente de API), ``python-requests``,
  ``Content-Type`` distinto de JSON y sin ``Content-Type``.
- Texto: acentos, ñ, emojis, CJK, tabuladores y saltos de línea; ids con comillas y barra.
- Tamaños: contexto de 5000 caracteres, pregunta de 500, id de 64.
- Entradas degeneradas: contexto de 1 carácter, solo espacios, solo saltos, solo puntuación,
  pregunta de 1 carácter o en blanco.
- Los 5 ítems del set público y 16 peticiones en paralelo.

Uso (el token va en el entorno, no en la línea de comandos):

    $env:QA_API_TOKEN = "..."           # PowerShell
    python scripts/robustness_check.py https://<url-publica>

Sale con código 1 si algún caso falla. Los tiempos que imprime incluyen la red del túnel.
"""
import json, os, sys, time, concurrent.futures as cf
import httpx
U = sys.argv[1]; T = os.environ["QA_API_TOKEN"]
A = {"Authorization": f"Bearer {T}"}
sample = json.load(open(__import__("pathlib").Path(__file__).resolve().parent.parent / "public_sample.json", encoding="utf-8"))["items"]
long_ctx = sample[4]["context"]
def post(body, headers=None, content=None, timeout=5.0):
    h = {**A, "Content-Type": "application/json", **(headers or {})}
    t = time.perf_counter()
    try:
        r = httpx.post(U + "/predict", headers=h, timeout=timeout, **({"content": content} if content is not None else {"json": body}))
    except Exception as e:
        return None, f"EXC {type(e).__name__}", (time.perf_counter() - t) * 1000
    return r, None, (time.perf_counter() - t) * 1000
def check(name, body, **kw):
    r, exc, ms = post(body, **kw)
    if exc: print(f"[FALLA] {name}: {exc} {ms:.0f} ms"); return False
    if r.status_code != 200: print(f"[FALLA] {name}: {r.status_code} {r.text[:120]}"); return False
    try: j = r.json()
    except Exception: print(f"[FALLA] {name}: respuesta no es JSON: {r.text[:80]!r}"); return False
    a, s, ctx = j.get("answer"), j.get("start"), body["context"]
    ok = j.get("id") == body["id"] and (a is None and s is None or (isinstance(s, int) and ctx[s:s+len(a)] == a)) and 0 <= j.get("score", -1) <= 1
    print(f"[{'OK' if ok else 'FALLA'}] {name}: {ms:.0f} ms answer={a!r}"); return ok
res = []
q = "How many attention heads does BERT-large have?"
base = {"id": "b-1", "context": sample[0]["context"], "question": q}
# 1. cabecera de navegador (ngrok gratuito muestra una página de advertencia a navegadores)
res.append(check("User-Agent de navegador", base, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36", "Accept": "text/html,application/xhtml+xml"}))
res.append(check("User-Agent python-requests", base, headers={"User-Agent": "python-requests/2.31"}))
# 2. Content-Type distinto o ausente
res.append(check("Content-Type text/plain", base, headers={"Content-Type": "text/plain"}))
r, exc, ms = post(None, content=json.dumps(base).encode(), headers={"Content-Type": ""}); print("[OK]" if r is not None and r.status_code == 200 else "[FALLA]", "sin Content-Type", r.status_code if r is not None else exc); res.append(r is not None and r.status_code == 200)
# 3. Unicode, saltos de línea, emojis, acentos, CJK, espacios raros
uni = "Café Müller – the “Zürich” branch 🚀 opened in 1999 in São Paulo.\nIt employs 250 people (東京 office: 40).\tThe CEO is Zoë Álvarez-Ñandú."
for i, qq in enumerate(["Where did the branch open?", "How many people does it employ?", "Who is the CEO?", "When was the first dam built?"]):
    res.append(check(f"unicode {i}", {"id": f"üñí-{i}-😀", "context": uni, "question": qq}))
res.append(check("pregunta unicode", {"id": "u", "context": uni, "question": "¿Quién es el CEO? 🤔"}))
# 4. Largo máximo: 5000 caracteres de texto real, pregunta de 500
big = (long_ctx + " ") * 3
big = big[:5000]; res.append(check("contexto 5000 car.", {"id": "big", "context": big, "question": q}))
res.append(check("pregunta 500 car.", {"id": "bq", "context": sample[0]["context"], "question": ("How many attention heads does BERT-large have? " * 11)[:500]}))
res.append(check("contexto 5000 de una palabra", {"id": "big2", "context": ("a" * 5000), "question": q}))
res.append(check("contexto 5000 sin espacios raros (guiones)", {"id": "big3", "context": ("word-" * 1000)[:5000], "question": q}))
# 5. Degenerados
for name, ctx, qq in [("contexto de 1 car.", "a", "What?"), ("contexto solo espacios", "     ", "What is it?"), ("contexto solo saltos", "\n\n\n", "What?"),
                      ("pregunta de 1 car.", sample[0]["context"], "?"), ("contexto con solo puntuación", "!!! ??? ...", "Who?"), ("pregunta vacía de sentido", sample[0]["context"], "     ")]:
    res.append(check(name, {"id": "d", "context": ctx, "question": qq}))
res.append(check("id de 64 car.", {"id": "x" * 64, "context": sample[0]["context"], "question": q}))
res.append(check("id con comillas y barra", {"id": 'a"b\c/d', "context": sample[0]["context"], "question": q}))
# 6. Todos los ítems del set público
for it in sample: res.append(check("público " + it["id"], {k: it[k] for k in ("id", "context", "question")}))
# 7. Paralelo (el harness manda uno por uno, pero por si acaso)
def one(i): return check(f"paralelo {i}", {"id": f"p{i}", "context": sample[i % 5]["context"], "question": sample[i % 5]["question"]})
with cf.ThreadPoolExecutor(8) as ex: res += list(ex.map(one, range(16)))
print(f"\n{sum(res)}/{len(res)} correctas")
sys.exit(0 if all(res) else 1)
