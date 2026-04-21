"""Plot benchmark comparison figures from benchmark JSON results."""
import glob
import json
import os

MPL_DIR = "/tmp/matplotlib"
os.makedirs(MPL_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_DIR)

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.size"] = 11

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")
OUT_DIR = RESULTS_DIR
X_TICKS = [1, 2, 4, 8, 16, 32, 63]

COLORS = {
    "VGGT": "#1f77b4",
    "CUT3R": "#ff7f0e",
    "InfiniteVGGT": "#2ca02c",
    "Pi3X": "#d62728",
    "WinT3R": "#9467bd",
    "SLAM-Former": "#8c564b",
}
MARKERS = {
    "VGGT": "o",
    "CUT3R": "s",
    "InfiniteVGGT": "^",
    "Pi3X": "D",
    "WinT3R": "v",
    "SLAM-Former": "P",
}
MODEL_ORDER = ["CUT3R", "WinT3R", "VGGT", "InfiniteVGGT", "Pi3X", "SLAM-Former"]


def load_data():
    jsons = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    if not jsons:
        raise FileNotFoundError(f"No benchmark JSON files found in {RESULTS_DIR}")

    all_data = {}
    skipped = {}
    for jf in jsons:
        with open(jf) as f:
            payload = json.load(f)
        model = payload["model"]
        valid_results = []
        skipped_results = []
        for item in payload["results"]:
            if "time_sec" in item and "peak_mem_gb" in item:
                valid_results.append(item)
            else:
                skipped_results.append(item)
        all_data[model] = sorted(valid_results, key=lambda x: x["num_images"])
        if skipped_results:
            skipped[model] = sorted(skipped_results, key=lambda x: x["num_images"])
    return all_data, skipped


def sorted_models(all_data):
    preferred = [m for m in MODEL_ORDER if m in all_data]
    remaining = sorted(m for m in all_data if m not in MODEL_ORDER)
    return preferred + remaining


def plot_series(ax, all_data, value_key, ylabel, title, transform=None, annotate_last=False):
    for model in sorted_models(all_data):
        results = all_data[model]
        if not results:
            continue
        counts = [r["num_images"] for r in results]
        values = [r[value_key] for r in results]
        if transform is not None:
            values = [transform(v, n) for v, n in zip(values, counts)]

        color = COLORS.get(model, "gray")
        marker = MARKERS.get(model, "o")
        ax.plot(
            counts,
            values,
            marker=marker,
            color=color,
            label=model,
            linewidth=2,
            markersize=6,
        )
        if annotate_last:
            ax.annotate(
                model,
                (counts[-1], values[-1]),
                textcoords="offset points",
                xytext=(6, 0),
                va="center",
                fontsize=9,
                color=color,
            )

    ax.set_xlabel("Number of Input Images")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(X_TICKS)


def save_primary_figure(all_data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax1, ax2, ax3, ax4 = axes.flat

    plot_series(ax1, all_data, "time_sec", "Inference Time (s)", "Inference Time vs. Input Images")
    plot_series(ax2, all_data, "peak_mem_gb", "Peak GPU Memory (GB)", "Peak GPU Memory vs. Input Images")
    plot_series(
        ax3,
        all_data,
        "time_sec",
        "Time Per Image (s/image)",
        "Per-Image Runtime vs. Input Images",
        transform=lambda time_sec, count: time_sec / count,
    )
    plot_series(
        ax4,
        all_data,
        "time_sec",
        "Throughput (images/s)",
        "Throughput vs. Input Images",
        transform=lambda time_sec, count: count / time_sec,
        annotate_last=True,
    )

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = os.path.join(OUT_DIR, "benchmark_comparison.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


def save_log_figure(all_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_series(ax1, all_data, "time_sec", "Inference Time (s)", "Inference Time vs. Input Images (Log Y)")
    plot_series(ax2, all_data, "peak_mem_gb", "Peak GPU Memory (GB)", "Peak GPU Memory vs. Input Images (Log Y)")
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = os.path.join(OUT_DIR, "benchmark_comparison_log.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


def print_skipped_summary(skipped):
    if not skipped:
        return
    print("\nSkipped points:")
    for model in sorted(skipped):
        for item in skipped[model]:
            reason = item.get("reason", "unknown")
            print(f"  {model:12s} n={item['num_images']:>2}: {reason}")


def main():
    all_data, skipped = load_data()
    save_primary_figure(all_data)
    save_log_figure(all_data)
    print_skipped_summary(skipped)


if __name__ == "__main__":
    main()
