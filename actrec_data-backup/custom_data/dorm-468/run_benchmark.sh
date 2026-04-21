#!/bin/bash
# ==============================================================================
# run_benchmark.sh - Benchmark 3D reconstruction models
#
# Usage:
#   ./run_benchmark.sh [model]    # all | vggt | cut3r | infinitevggt | pi3x | wint3r
# ==============================================================================
set -euo pipefail


BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-all}"

run_vggt() {
    echo "=============================================="
    echo "  Benchmarking VGGT (env: actrec-new)"
    echo "=============================================="
    conda run --no-capture-output -n actrec-new python "${BENCH_DIR}/bench_vggt.py"
}

run_cut3r() {
    echo "=============================================="
    echo "  Benchmarking CUT3R (env: cut3r)"
    echo "=============================================="
    conda run --no-capture-output -n cut3r python "${BENCH_DIR}/bench_cut3r.py"
}

run_infinitevggt() {
    echo "=============================================="
    echo "  Benchmarking InfiniteVGGT (env: infinitevggt)"
    echo "=============================================="
    conda run --no-capture-output -n infinitevggt python "${BENCH_DIR}/bench_infinitevggt.py"
}

run_pi3x() {
    echo "=============================================="
    echo "  Benchmarking Pi3X (env: vggt)"
    echo "=============================================="
    conda run --no-capture-output -n vggt python "${BENCH_DIR}/bench_pi3x.py"
}

run_wint3r() {
    echo "=============================================="
    echo "  Benchmarking WinT3R (env: WinT3R)"
    echo "=============================================="
    conda run --no-capture-output -n WinT3R python "${BENCH_DIR}/bench_wint3r.py"
}

run_slamformer() {
    echo "=============================================="
    echo "  Benchmarking SLAM-Former (env: SLAM-Former)"
    echo "=============================================="
    export PYTHONPATH=/home/ghr/fs/Junyi/SLAM-Former/src/croco:${PYTHONPATH:-}
    conda run --no-capture-output -n SLAM-Former python "${BENCH_DIR}/bench_slamformer.py"
}

case "$MODEL" in
    all)
        run_vggt
        run_cut3r
        run_infinitevggt
        run_pi3x
        run_wint3r
        run_slamformer
        ;;
    vggt)          run_vggt ;;
    cut3r)         run_cut3r ;;
    infinitevggt)  run_infinitevggt ;;
    pi3x)          run_pi3x ;;
    wint3r)        run_wint3r ;;
    slamformer)    run_slamformer ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Usage: $0 [all|vggt|cut3r|infinitevggt|pi3x|wint3r|slamformer]"
        exit 1 ;;
esac

# Print summary table
echo ""
echo "=============================================="
echo "  Generating summary..."
echo "=============================================="
conda run --no-capture-output -n actrec-new python "${BENCH_DIR}/bench_summary.py"
