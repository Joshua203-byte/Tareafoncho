"""División del dev oficial de SQuAD 2.0 en dos conjuntos: dev_select y dev_report.

Por qué existen dos conjuntos de desarrollo
-------------------------------------------
El split oficial ``validation`` de SQuAD 2.0 (11 873 preguntas, 35 artículos) no
entrena nunca, así que lo usamos como validación. Lo dividimos en dos:

- ``dev_select``: se mira durante el entrenamiento. Sirve para ver si hay
  sobreajuste, para elegir el checkpoint y para calibrar el umbral τ.
- ``dev_report``: se usa una sola vez, al final, para los números del reporte.

Si ambos fueran el mismo conjunto, cada decisión tomada mirándolo (checkpoint,
τ, learning rate) inflaría el número que reportamos. Separarlos mantiene limpio
el número final.

Por qué la división es por artículo
-----------------------------------
Las preguntas de un mismo artículo comparten el contexto. Si repartiéramos por
pregunta, un contexto podría aparecer en ``dev_select`` y en ``dev_report``, y el
modelo llegaría a ``dev_report`` ya conociendo ese texto. Repartir por artículo
evita esa filtración. La semilla fija (42) hace la división reproducible; el
manifiesto con los artículos de cada conjunto se guarda en ``logs/splits.json``.
"""

import json
import random
from pathlib import Path

SEED = 42


def split_titles(titles, seed=SEED):
    """Reparte los títulos únicos de artículos en dos mitades reproducibles.

    Se ordenan antes de barajar, para que el resultado no dependa del orden en
    que aparecen las preguntas en el dataset.

    Devuelve ``(select_titles, report_titles)`` como conjuntos disjuntos.
    """
    unique = sorted(set(titles))
    random.Random(seed).shuffle(unique)
    half = len(unique) // 2
    return set(unique[:half]), set(unique[half:])


def split_indices(titles, seed=SEED):
    """Índices de las preguntas de cada conjunto, según el título de su artículo."""
    select_titles, _ = split_titles(titles, seed)
    select_idx, report_idx = [], []
    for i, title in enumerate(titles):
        (select_idx if title in select_titles else report_idx).append(i)
    return select_idx, report_idx


def write_manifest(titles, select_idx, report_idx, path):
    """Guarda qué artículos quedaron en cada conjunto, para poder auditar la división."""
    select_set = {titles[i] for i in select_idx}
    report_set = {titles[i] for i in report_idx}
    manifest = {
        "seed": SEED,
        "dev_select": {"questions": len(select_idx), "articles": sorted(select_set)},
        "dev_report": {"questions": len(report_idx), "articles": sorted(report_set)},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    return manifest
