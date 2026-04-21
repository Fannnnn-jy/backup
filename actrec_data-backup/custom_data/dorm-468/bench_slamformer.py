"""Benchmark SLAM-Former with varying number of input images."""
import glob
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


os.environ["CUDA_VISIBLE_DEVICES"] = "4"

SLAM_ROOT = "/home/ghr/fs/Junyi/SLAM-Former"
CKPT_PATH = "/home/ghr/fs/Junyi/data/proj/model_weights/checkpoint-10.pth.model"
IMG_DIR = "/home/ghr/fs/Junyi/data/proj/custom_data/dorm-468/pics"
COUNTS = [1, 2, 4, 8, 16, 32, 63]
# COUNTS = [63]
RESULTS_DIR = "/home/ghr/fs/Junyi/data/proj/custom_data/dorm-468/benchmark_results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "slamformer.json")
PLY_PATH = os.path.join(RESULTS_DIR, "slamformer_reconstruction.ply")
TARGET_SIZE = 518
RETENTION_RATIO = 0.5
KF_TH = 0.1


def image_sort_key(path):
    stem = Path(path).stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def query_gpu_mem_mb(pid):
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    peak = 0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            proc_pid = int(parts[0])
            used_mem = int(parts[1])
        except ValueError:
            continue
        if proc_pid == pid:
            peak = max(peak, used_mem)
    return peak


def run_demo(image_paths, output_dir):
    input_dir = tempfile.mkdtemp(prefix="slamformer_inputs_")
    try:
        for src in image_paths:
            shutil.copy2(src, os.path.join(input_dir, os.path.basename(src)))

        cmd = [
            "python",
            "slam/demo.py",
            "--ckpt_path",
            CKPT_PATH,
            "--image_folder",
            input_dir,
            "--output_dir",
            output_dir,
            "--target_size",
            str(TARGET_SIZE),
            "--retention_ratio",
            str(RETENTION_RATIO),
            "--kf_th",
            str(KF_TH),
            "--no_save_gmem",
        ]

        # SLAM-Former's attention.py forces FLASH_ATTENTION backend for bf16
        # which may not be available on all GPUs. Disable flash SDP globally
        # before the demo script imports anything.
        patch_and_run = (
            "import torch; torch.backends.cuda.enable_flash_sdp(False); "
            "import runpy, sys; sys.argv = sys.argv[1:]; "
            "runpy.run_path(sys.argv[0], run_name='__main__')"
        )
        patched_cmd = ["python", "-c", patch_and_run] + cmd[1:]

        start = time.time()
        proc = subprocess.Popen(patched_cmd, cwd=SLAM_ROOT)
        peak_mem_mb = 0.0

        while proc.poll() is None:
            used_mem = query_gpu_mem_mb(proc.pid)
            if used_mem is not None:
                peak_mem_mb = max(peak_mem_mb, float(used_mem))
            time.sleep(0.2)

        used_mem = query_gpu_mem_mb(proc.pid)
        if used_mem is not None:
            peak_mem_mb = max(peak_mem_mb, float(used_mem))

        elapsed = time.time() - start
        if proc.returncode != 0:
            raise RuntimeError(f"SLAM-Former demo failed with exit code {proc.returncode}")

        return elapsed, peak_mem_mb / 1024.0
    finally:
        shutil.rmtree(input_dir, ignore_errors=True)


all_images = sorted(glob.glob(os.path.join(IMG_DIR, "*")), key=image_sort_key)
assert len(all_images) >= 63, f"Expected >=63 images, found {len(all_images)}"

skip_benchmark = os.path.exists(RESULTS_FILE)
skip_ply = os.path.exists(PLY_PATH)

if skip_benchmark and skip_ply:
    print("Benchmark JSON and PLY both exist, nothing to do.")
    raise SystemExit(0)

if skip_benchmark:
    print(f"Benchmark JSON already exists: {RESULTS_FILE}")
    print("Skipping benchmark, only running full reconstruction...")
    run_counts = [COUNTS[-1]]
else:
    run_counts = COUNTS

# Force all later frames to be treated as keyframes so small subsets are benchmarkable.
# The default demo threshold can leave `self.map` empty for short sequences.
print("Warmup...")
warmup_dir = tempfile.mkdtemp(prefix="slamformer_warmup_")
try:
    _elapsed, _peak_mem = run_demo(all_images[:2], warmup_dir)
finally:
    shutil.rmtree(warmup_dir, ignore_errors=True)

results = []
for n in run_counts:
    print(f"\n{'=' * 50}")
    print(f"  SLAM-Former: {n} images")
    print(f"{'=' * 50}")

    if n < 2:
        print("  Skipped: SLAM-Former demo cannot terminate cleanly on a single image.")
        results.append(
            {
                "num_images": n,
                "status": "skipped",
                "reason": "SLAM-Former requires at least 2 images for a valid SLAM run",
            }
        )
        continue

    run_dir = tempfile.mkdtemp(prefix=f"slamformer_run_{n}_")
    try:
        elapsed, peak_mem = run_demo(all_images[:n], run_dir)
        print(f"  Time: {elapsed:.2f}s | Peak GPU Mem: {peak_mem:.2f} GB")

        results.append(
            {
                "num_images": n,
                "time_sec": round(elapsed, 3),
                "peak_mem_gb": round(peak_mem, 3),
            }
        )

        if n == COUNTS[-1] and not skip_ply:
            src_ply = os.path.join(run_dir, "final.ply")
            if not os.path.exists(src_ply):
                raise FileNotFoundError(f"Expected reconstruction file not found: {src_ply}")
            os.makedirs(os.path.dirname(PLY_PATH), exist_ok=True)
            shutil.copy2(src_ply, PLY_PATH)
            print(f"  Saved to {PLY_PATH}")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

if not skip_benchmark:
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump({"model": "SLAM-Former", "results": results}, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
else:
    print("\nBenchmark skipped (already exists). Reconstruction only.")
