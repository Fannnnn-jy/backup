"""Print a summary table from benchmark JSON results."""
import os, json, glob

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")

jsons = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
if not jsons:
    print("No benchmark results found.")
    exit()

all_data = {}
for jf in jsons:
    with open(jf) as f:
        d = json.load(f)
    all_data[d["model"]] = {r["num_images"]: r for r in d["results"]}

models = list(all_data.keys())
counts = sorted({n for m in all_data.values() for n in m})

# --- Time table ---
print("\n" + "=" * 70)
print("  Inference Time (seconds)")
print("=" * 70)
header = f"{'Images':>8}" + "".join(f"{m:>16}" for m in models)
print(header)
print("-" * len(header))
for n in counts:
    row = f"{n:>8}"
    for m in models:
        v = all_data[m].get(n)
        row += f"{v['time_sec']:>16.2f}" if v and 'time_sec' in v else f"{'N/A':>16}"
    print(row)

# --- Memory table ---
print("\n" + "=" * 70)
print("  Peak GPU Memory (GB)")
print("=" * 70)
header = f"{'Images':>8}" + "".join(f"{m:>16}" for m in models)
print(header)
print("-" * len(header))
for n in counts:
    row = f"{n:>8}"
    for m in models:
        v = all_data[m].get(n)
        row += f"{v['peak_mem_gb']:>16.2f}" if v and 'peak_mem_gb' in v else f"{'N/A':>16}"
    print(row)

print()
