"""Genera las gráficas del reporte a partir de los registros de entrenamiento guardados en ``logs/``.

No entrena ni evalúa nada: solo lee archivos que ya existen, así que se puede correr cuando se quiera
y da siempre las mismas imágenes. Escribe tres figuras (en inglés, porque van al ``REPORT.md``):

``logs/loss.png``        pérdida de entrenamiento por paso, modelo servido (fine-tuning completo) y control LoRA.
``logs/dev_select.png``  métricas de ``dev_select`` por época del modelo servido (la época 3 es la elegida).
``logs/tau_sweep.png``   puntaje con la escala de la guía contra el umbral τ, para el checkpoint servido.

Datos de entrada (todos versionados):

- ``logs/runs/full_ft_4ep/train_log.jsonl``         pérdida cada 50 pasos, épocas 1 a 4 (reanuda la corrida de 2 épocas).
- ``logs/runs/r16_es/train_log.jsonl``              pérdida de LoRA r = 16, épocas 1 a 5 (control).
- ``logs/runs/full_ft_2ep/dev_select_metrics.jsonl`` y ``logs/runs/full_ft_4ep/dev_select_metrics.jsonl``
  métricas de ``dev_select`` de las épocas 1–2 y 3–4, todas con τ = −2.
- ``logs/runs/full_ft_4ep/item_score_epoch_3.json`` barrido de τ de la época 3 (escala de la guía, 25 % sin respuesta).

``dev_report`` no aparece aquí: se evaluó una sola vez y su resultado está en ``logs/report/``.

Uso (desde la raíz del repo; necesita ``matplotlib``, ver ``requirements.txt``):

    python scripts/plot_training.py
    python scripts/plot_training.py --out logs --dpi 200
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo escribe archivos
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "logs" / "runs"

# Paleta categórica de referencia (primeras tres posiciones, validadas entre todos los pares) y tintas.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

SERVED_EPOCH = 3  # checkpoint que sirve la API
GUIDE_BASELINE, GUIDE_REFERENCE = 0.25, 0.70  # anclas de la guía, sección XIV.1


def read_jsonl(path):
    """Lista de dicts, uno por línea no vacía."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def style(ax, xlabel, ylabel, title):
    """Aspecto común: superficie clara, rejilla tenue, solo ejes izquierdo e inferior, texto en tinta."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def new_figure(width, height, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE)
    return fig, axes


def label_end(ax, x, y, text, color, dy=0.0):
    """Etiqueta directa al final de una serie: la identidad no depende solo del color."""
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=INK, fontsize=9,
                va="center", ha="left", annotation_clip=False,
                bbox={"boxstyle": "square,pad=0.1", "fc": SURFACE, "ec": "none"})
    ax.plot([x], [y], marker="o", markersize=5, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5)


def plot_loss(out, dpi):
    """Pérdida de entrenamiento por época (cada punto es el promedio reportado cada 50 pasos)."""
    served = [r for r in read_jsonl(RUNS / "full_ft_4ep" / "train_log.jsonl") if "loss" in r and "step" in r]
    lora = [r for r in read_jsonl(RUNS / "r16_es" / "train_log.jsonl") if "loss" in r and "step" in r]
    fig, ax = new_figure(8, 4.2)
    ax.plot([r["epoch"] for r in lora], [r["loss"] for r in lora], color=ORANGE, linewidth=1.6)
    ax.plot([r["epoch"] for r in served], [r["loss"] for r in served], color=BLUE, linewidth=1.6)
    ax.set_ylim(0, 3)  # los primeros pasos (pérdida ≈ 5.7) se salen: la forma de la curva importa más
    ax.set_xlim(0, 5)
    ax.axvline(SERVED_EPOCH, color=INK_SECONDARY, linestyle=":", linewidth=1)
    ax.annotate("epoch 3: served checkpoint", (SERVED_EPOCH, 2.9), xytext=(5, 0),
                textcoords="offset points", color=INK_SECONDARY, fontsize=8, va="top")
    label_end(ax, served[-1]["epoch"], served[-1]["loss"], "Full fine-tuning (served)", BLUE, dy=-8)
    label_end(ax, lora[-1]["epoch"], lora[-1]["loss"], "LoRA r=16 (control)", ORANGE, dy=8)
    ax.set_xticks(range(0, 6))
    style(ax, "Epoch", "Training loss (cross-entropy, start + end)", "Training loss per step")
    fig.subplots_adjust(right=0.72, left=0.1, bottom=0.14)
    fig.savefig(out / "loss.png", dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def plot_dev_select(out, dpi):
    """Métricas de dev_select por época, todas con τ = −2. Dos paneles: no se mezclan escalas en un eje."""
    rows = read_jsonl(RUNS / "full_ft_2ep" / "dev_select_metrics.jsonl") + read_jsonl(
        RUNS / "full_ft_4ep" / "dev_select_metrics.jsonl"
    )
    epochs = [r["epoch"] for r in rows]
    fig, (left, right) = new_figure(10, 4.2, ncols=2)
    # Al final de la curva los tres valores están a ~1 punto: cada etiqueta lleva un desplazamiento vertical propio.
    series = [("F1", "F1, all questions", BLUE, 0), ("HasAns_F1", "F1, answerable", ORANGE, 11),
              ("NoAns_acc", "Accuracy, unanswerable", AQUA, -11)]
    for key, label, color, dy in series:
        values = [r[key] for r in rows]
        left.plot(epochs, values, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE)
        left.annotate(label, (epochs[-1], values[-1]), xytext=(8, dy), textcoords="offset points",
                      color=INK, fontsize=8.5, va="center", annotation_clip=False)
    left.set_xticks(epochs)
    left.set_xlim(0.8, 4.0)
    style(left, "Epoch", "Score (0–100)", "dev_select by epoch, τ = −2")

    proxy = [r["guide_reweighted"] for r in rows]
    right.plot(epochs, proxy, color=BLUE, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE)
    for e, p in zip(epochs, proxy):
        below = e == epochs[0]  # el primer punto queda cerca del eje y de la línea de referencia: etiqueta debajo
        right.annotate(f"{p:.3f}", (e, p), xytext=(12 if below else 0, -16 if below else 9), textcoords="offset points",
                       ha="center", color=INK, fontsize=8.5, fontweight="bold" if e == SERVED_EPOCH else "normal")
    right.axhline(GUIDE_REFERENCE, color=INK_SECONDARY, linestyle="--", linewidth=1)
    right.annotate("guide reference 0.70", (epochs[-1], GUIDE_REFERENCE), xytext=(0, -12),
                   textcoords="offset points", ha="right", color=INK_SECONDARY, fontsize=8)
    right.axvline(SERVED_EPOCH, color=INK_SECONDARY, linestyle=":", linewidth=1)
    right.annotate("served", (SERVED_EPOCH, 0.665), xytext=(5, 0), textcoords="offset points",
                   color=INK_SECONDARY, fontsize=8)
    right.set_xticks(epochs)
    right.set_ylim(0.66, 0.74)
    style(right, "Epoch", "Guide-scale score (25 % unanswerable)", "Selection metric (epoch 3 is the best)")
    fig.subplots_adjust(wspace=0.75, right=0.95, left=0.07, bottom=0.14)
    fig.savefig(out / "dev_select.png", dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def plot_tau_sweep(out, dpi):
    """Puntaje contra τ para la época 3. τ = ±∞ son los extremos: siempre responde / siempre null."""
    sweep = json.loads((RUNS / "full_ft_4ep" / "item_score_epoch_3.json").read_text(encoding="utf-8"))
    labels = ["−∞" if r["tau"] < -1e6 else "+∞" if r["tau"] > 1e6 else f"{r['tau']:g}".replace("-", "−") for r in sweep]
    positions = list(range(len(sweep)))
    full = [r["score_reweighted"] for r in sweep]
    exact = [r["score_exact_only_reweighted"] for r in sweep]
    fig, ax = new_figure(8, 4.2)
    ax.plot(positions, full, color=BLUE, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE)
    ax.plot(positions, exact, color=ORANGE, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE)
    best = max(range(len(sweep)), key=lambda i: full[i] if abs(sweep[i]["tau"]) < 1e6 else -1)
    ax.axvline(positions[labels.index("−2")], color=INK_SECONDARY, linestyle=":", linewidth=1)
    ax.annotate("chosen τ = −2", (positions[labels.index("−2")], 0.28), xytext=(5, 0), textcoords="offset points",
                color=INK_SECONDARY, fontsize=8)
    for y, text in ((GUIDE_REFERENCE, "reference 0.70"), (GUIDE_BASELINE, "baseline 0.25 (always null)")):
        ax.axhline(y, color=INK_SECONDARY, linestyle="--", linewidth=1)
        ax.annotate(text, (positions[-1], y), xytext=(-4, 5), textcoords="offset points", ha="right",
                    color=INK_SECONDARY, fontsize=8)
    ax.annotate(f"{full[best]:.3f}", (positions[best], full[best]), xytext=(0, 9), textcoords="offset points",
                ha="center", color=INK, fontsize=9, fontweight="bold")
    ax.annotate("Partial credit counted", (positions[6], full[6]), xytext=(0, 12), textcoords="offset points",
                ha="center", color=INK, fontsize=8.5)
    ax.annotate("Exact matches only", (positions[6], exact[6]), xytext=(0, -16), textcoords="offset points",
                ha="center", color=INK, fontsize=8.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.2, 0.78)
    style(ax, "Threshold τ (answer if best span − null score > τ)", "Guide-scale score, reweighted to 25 % unanswerable",
          "No-answer threshold sweep on dev_select, epoch 3")
    fig.subplots_adjust(left=0.1, right=0.96, bottom=0.16)
    fig.savefig(out / "tau_sweep.png", dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Genera las tres figuras en ``--out`` (por defecto ``logs/``)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "logs")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_loss(args.out, args.dpi)
    plot_dev_select(args.out, args.dpi)
    plot_tau_sweep(args.out, args.dpi)
    for name in ("loss.png", "dev_select.png", "tau_sweep.png"):
        print("escrito", args.out / name)


if __name__ == "__main__":
    main()
