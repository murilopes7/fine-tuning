"""Gera visualizações dos resultados da avaliação offline."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


METRICS = {
    "exact_action_match": "Exact Action Match",
    "next_room_accuracy": "Next Room Accuracy",
    "format_validity": "Format Validity",
    "non_idle_exact_action_match": "Non-idle Exact Match",
    "macro_f1": "Macro-F1",
}


def load_results(path):
    """Carrega os resultados da avaliação."""
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def get_approaches(results):
    """Identifica as abordagens presentes no resultado."""
    return {
        name: values
        for name, values in results.items()
        if isinstance(values, dict) and "exact_action_match" in values
    }


def save_figure(fig, out_dir, filename):
    """Salva a figura em alta resolução."""
    fig.tight_layout()
    fig.savefig(
        out_dir / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_metrics(approaches, out_dir):
    """Compara as métricas gerais das abordagens."""
    fig, ax = plt.subplots(figsize=(12, 6))

    positions = list(range(len(METRICS)))
    width = 0.8 / len(approaches)

    for i, (name, values) in enumerate(approaches.items()):
        offset = (i - (len(approaches) - 1) / 2) * width

        pairs = [
            (position + offset, values[key])
            for position, key in enumerate(METRICS)
            if values.get(key) is not None
        ]

        bars = ax.bar(
            [position for position, _ in pairs],
            [value for _, value in pairs],
            width=width,
            label=name,
        )

        ax.bar_label(
            bars,
            labels=[f"{value:.1%}" for _, value in pairs],
            padding=3,
            fontsize=9,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        METRICS.values(),
        rotation=15,
        ha="right",
    )

    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Desempenho")
    ax.set_title("Comparação das métricas offline")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    save_figure(fig, out_dir, "metrics_comparison.png")


def plot_action_distribution(approaches, out_dir):
    """Mostra a proporção de exemplos com e sem ações."""
    reference = next(iter(approaches.values()))

    total = reference["n"]
    non_idle = reference["non_idle_n"]
    idle = total - non_idle

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.pie(
        [idle, non_idle],
        labels=[
            f"Sem ações ({idle})",
            f"Com ações ({non_idle})",
        ],
        autopct="%1.2f%%",
        startangle=90,
        wedgeprops={"width": 0.4},
    )

    ax.text(
        0,
        0,
        f"{total}\nexemplos",
        ha="center",
        va="center",
        fontsize=15,
    )

    ax.set_title("Distribuição de ações na validação")

    save_figure(fig, out_dir, "action_distribution.png")


def plot_f1_per_class(approaches, out_dir):
    """Gera um heatmap do F1 por classe de ação."""
    classes = sorted(
        {
            action
            for values in approaches.values()
            for action in values.get("f1_per_class", {})
        }
    )

    if not classes:
        return

    names = list(approaches)

    matrix = [
        [
            approaches[name].get("f1_per_class", {}).get(action, float("nan"))
            for name in names
        ]
        for action in classes
    ]

    fig, ax = plt.subplots(
        figsize=(
            max(6, len(names) * 1.2),
            max(6, len(classes) * 0.45),
        )
    )

    image = ax.imshow(
        matrix,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            label = "N/A" if value != value else f"{value:.2f}"

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, label="F1")
    ax.set_title("F1 por classe de ação")
    ax.set_xlabel("Abordagem")

    save_figure(fig, out_dir, "f1_per_class.png")


def main():
    parser = argparse.ArgumentParser(description="Visualização da avaliação offline")

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/baselines_val.json"),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/figures"),
    )

    args = parser.parse_args()

    results = load_results(args.input)
    approaches = get_approaches(results)

    if not approaches:
        raise ValueError("Nenhuma abordagem encontrada")

    args.out.mkdir(parents=True, exist_ok=True)

    plot_metrics(approaches, args.out)
    plot_action_distribution(approaches, args.out)
    plot_f1_per_class(approaches, args.out)

    print(f"Gráficos salvos em: {args.out}")


if __name__ == "__main__":
    main()
