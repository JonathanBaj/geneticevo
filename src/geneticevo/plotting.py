"""Optional plotting from incremental JSONL records."""
import json


def plot_history(metrics_path, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(metrics_path, encoding="utf-8") as file:
        history = [json.loads(line) for line in file if line.strip()]
    figure, axes = plt.subplots(figsize=(7, 4))
    generations = [row["generation"] for row in history]
    for key in ("best", "mean", "archive_best"):
        axes.plot(generations, [row[key] for row in history], label=key)
    axes.set(xlabel="Evaluated generation (starts at 0)", ylabel="Fitness")
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
