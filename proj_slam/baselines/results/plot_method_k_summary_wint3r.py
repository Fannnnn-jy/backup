#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "active-gs": "Active-GS",
    "active-gs-gt-depth": "Active-GS GT Depth",
    "active-gs-pred-depth": "Active-GS Pred Depth",
    "random": "Random",
}

METHOD_COLORS = {
    "active-gs": "#9467bd",
    "active-gs-gt-depth": "#1f77b4",
    "active-gs-pred-depth": "#ff7f0e",
    "random": "#2ca02c",
}

METHOD_ORDER = [
    "active-gs",
    "active-gs-gt-depth",
    "active-gs-pred-depth",
    "random",
]

RAW_METRICS = [
    ("voxel_coverage", "coverage", "Coverage"),
    ("chamfer_distance", "cd", "Chamfer Distance"),
    ("depth_l2_voxel", "l2", "Depth L2"),
]


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate *_wint3r.json results into summary CSV/JSON files and "
            "generate method-vs-k plots."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_dir,
        help=f"Directory containing *_wint3r.json files. Default: {default_dir}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_dir,
        help=f"Directory for summary files and plots. Default: {default_dir}",
    )
    parser.add_argument(
        "--glob",
        default="*_wint3r.json",
        help="Glob for input result files. Default: *_wint3r.json",
    )
    parser.add_argument(
        "--dataset-prefix",
        default="replicacad",
        help=(
            "Only include files whose scene_id matches this prefix. "
            "Examples: replicacad, procthor, replicacad/apt_0. "
            "Use an empty string to disable filtering."
        ),
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=None,
        help="Target max k. Default: infer from the largest max_views among matched files.",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Output filename stem. Default: method_k_summary_<dataset-prefix>_wint3r",
    )
    return parser.parse_args()


def scene_matches(scene_id: str, dataset_prefix: str) -> bool:
    if not dataset_prefix:
        return True
    normalized = dataset_prefix.rstrip("/")
    return scene_id == normalized or scene_id.startswith(f"{normalized}/")


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def compute_mean(values: list[float]) -> float:
    return statistics.fmean(values)


def compute_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values)


def load_file_records(input_dir: Path, input_glob: str, dataset_prefix: str) -> list[dict]:
    records = []
    for path in sorted(input_dir.glob(input_glob)):
        payload = json.loads(path.read_text())
        scene_id = payload.get("scene_id")
        method = payload.get("method")
        metrics_by_k = payload.get("metrics_by_k")
        if not scene_id or not method or not isinstance(metrics_by_k, list) or not metrics_by_k:
            raise ValueError(f"Unexpected wint3r result format: {path}")
        if not scene_matches(scene_id, dataset_prefix):
            continue

        metrics_map = {}
        for item in metrics_by_k:
            k = int(item["k"])
            metrics_map[k] = {raw_key: float(item[raw_key]) for raw_key, _, _ in RAW_METRICS}

        max_available_k = max(metrics_map)
        max_views = int(payload.get("max_views", max_available_k))
        records.append(
            {
                "path": path,
                "scene_id": scene_id,
                "method": method,
                "metrics_map": metrics_map,
                "max_available_k": max_available_k,
                "max_views": max_views,
            }
        )

    return records


def metric_values_for_file(record: dict, target_k: int) -> dict[str, float]:
    metrics_map = record["metrics_map"]
    max_available_k = record["max_available_k"]
    if target_k in metrics_map:
        return metrics_map[target_k]
    if target_k > max_available_k:
        return metrics_map[max_available_k]
    raise ValueError(
        f"Missing intermediate k={target_k} in {record['path'].name}; "
        f"available ks: {sorted(metrics_map)}"
    )


def build_summary(records: list[dict], resolved_max_k: int, dataset_prefix: str, input_glob: str) -> dict:
    methods: dict[str, dict] = {}
    per_method_values: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for record in records:
        method = record["method"]
        method_entry = methods.setdefault(method, {"files": []})
        method_entry["files"].append(record["path"].name)

        for k in range(1, resolved_max_k + 1):
            metrics = metric_values_for_file(record, k)
            for _, short_name, _ in RAW_METRICS:
                raw_key = next(raw for raw, short, _ in RAW_METRICS if short == short_name)
                per_method_values[method][k][short_name].append(metrics[raw_key])

    for method, method_entry in methods.items():
        method_entry["files"].sort()
        method_entry["num_files"] = len(method_entry["files"])
        rows = []
        for k in range(1, resolved_max_k + 1):
            row = {
                "k": k,
                "num_files": len(per_method_values[method][k]["coverage"]),
            }
            for _, short_name, _ in RAW_METRICS:
                values = per_method_values[method][k][short_name]
                row[f"{short_name}_mean"] = compute_mean(values)
                row[f"{short_name}_std"] = compute_std(values)
            rows.append(row)
        method_entry["rows"] = rows

    return {
        "dataset_filter": f"{dataset_prefix}/*" if dataset_prefix else "*",
        "input_glob": input_glob,
        "fill_rule": (
            f"for each file, extend missing k in [1,{resolved_max_k}] "
            "using metrics from the largest existing k in that file"
        ),
        "max_k": resolved_max_k,
        "methods": dict(sorted(methods.items(), key=lambda item: method_sort_key(item[0]))),
    }


def save_summary_csv(summary: dict, csv_path: Path) -> None:
    fieldnames = [
        "method",
        "k",
        "num_files",
        "coverage_mean",
        "coverage_std",
        "cd_mean",
        "cd_std",
        "l2_mean",
        "l2_std",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, method_data in summary["methods"].items():
            for row in method_data["rows"]:
                writer.writerow(
                    {
                        "method": method,
                        "k": row["k"],
                        "num_files": row["num_files"],
                        "coverage_mean": f"{row['coverage_mean']:.6f}",
                        "coverage_std": f"{row['coverage_std']:.6f}",
                        "cd_mean": f"{row['cd_mean']:.6f}",
                        "cd_std": f"{row['cd_std']:.6f}",
                        "l2_mean": f"{row['l2_mean']:.6f}",
                        "l2_std": f"{row['l2_std']:.6f}",
                    }
                )


def plot_metric(ax, summary: dict, metric_key: str, title: str) -> None:
    max_k = summary["max_k"]
    tick_step = 1 if max_k <= 10 else 2

    for method, method_data in summary["methods"].items():
        rows = method_data["rows"]
        ks = [row["k"] for row in rows]
        means = [row[f"{metric_key}_mean"] for row in rows]
        stds = [row[f"{metric_key}_std"] for row in rows]
        lower = [max(0.0, mean - std) for mean, std in zip(means, stds)]
        upper = [mean + std for mean, std in zip(means, stds)]
        color = METHOD_COLORS.get(method)

        ax.plot(
            ks,
            means,
            marker="o",
            linewidth=2.2,
            markersize=4.5,
            color=color,
            label=METHOD_LABELS.get(method, method),
        )
        ax.fill_between(ks, lower, upper, color=color, alpha=0.16)

    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_xlim(1, max_k)
    ax.set_xticks(range(1, max_k + 1, tick_step))
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)


def save_combined_figure(summary: dict, output_path: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, (_, short_name, title) in zip(axes, RAW_METRICS):
        plot_metric(ax, summary, short_name, title)
    axes[0].set_ylabel("Value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"{title_prefix} Method Comparison by k", fontsize=14, y=1.08)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_single_metric_figures(summary: dict, output_dir: Path, stem: str) -> list[Path]:
    outputs = []
    for _, short_name, title in RAW_METRICS:
        fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
        plot_metric(ax, summary, short_name, title)
        ax.set_ylabel(title)
        ax.legend(frameon=False)
        output_path = output_dir / f"{stem}_{short_name}.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output_path)
    return outputs


def dataset_tag(dataset_prefix: str) -> str:
    if not dataset_prefix:
        return "all"
    return dataset_prefix.replace("/", "_").replace("*", "all")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_file_records(input_dir, args.glob, args.dataset_prefix)
    if not records:
        raise FileNotFoundError(
            f"No wint3r result files matched glob={args.glob!r} with dataset_prefix={args.dataset_prefix!r} in {input_dir}"
        )

    inferred_max_k = max(record["max_views"] for record in records)
    resolved_max_k = args.max_k or inferred_max_k
    stem = args.stem or f"method_k_summary_{dataset_tag(args.dataset_prefix)}_wint3r"

    summary = build_summary(records, resolved_max_k, args.dataset_prefix, args.glob)

    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    combined_path = output_dir / f"{stem}_combined.png"

    save_summary_csv(summary, csv_path)
    json_path.write_text(json.dumps(summary, indent=2))
    single_paths = save_single_metric_figures(summary, output_dir, stem)
    save_combined_figure(summary, combined_path, title_prefix=dataset_tag(args.dataset_prefix))

    print(f"Matched files: {len(records)}")
    print(f"Dataset prefix: {args.dataset_prefix or '*'}")
    print(f"Resolved max k: {resolved_max_k}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {combined_path}")
    for path in single_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
