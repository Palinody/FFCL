#!/bin/bash
# Runs the C++ reference libraries over the whole matrix: every dimension, every library, every
# workload it supports, at 1 and 6 threads where the library is actually parallel.
#
#   benchmark/references/run_references.sh <build_dir> <output_dir>
#
# build_dir is benchmark/references/build. One JSON file per (library, dimension, threads) is written
# to output_dir, in the same schema as the project harness, so aggregate_references.py can merge it
# with the FFCL and Python-reference results.
set -euo pipefail

BUILD="${1:?build directory}"
OUT="${2:?output directory}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p "$OUT"
: > "$OUT/stderr.log"

spin() { taskset -c 0 bash -c 'end=$((SECONDS+1)); while [ $SECONDS -lt $end ]; do :; done'; }

# Single-threaded libraries: one run per workload, threads recorded as 1.
SERIAL_LIBS="nanoflann flann pico_tree mlpack"
# Libraries that scale with threads: measured at 1 and 6.
PARALLEL_LIBS="arborx pdbscan pargeo"

workloads_for() {
    case "$1" in
        nanoflann) echo "build knn" ;;
        flann)     echo "build knn" ;;
        pico_tree) echo "build knn radius" ;;
        mlpack)    echo "build knn dbscan mst" ;;
        arborx)    echo "build knn radius dbscan mst dendrogram hdbscan" ;;
        pdbscan)   echo "dbscan" ;;
        pargeo)    echo "mst dendrogram" ;;
    esac
}

binary_for() {
    echo "$BUILD/refbench_$1"
}

run_one() {
    local lib="$1" workload="$2" dim="$3" threads="$4"
    local inputs="$REPO/benchmark/bin/datasets_${dim}d/inputs"
    local out="$OUT/${lib}_${dim}d_t${threads}.json"
    local binary; binary="$(binary_for "$lib")"
    [ -x "$binary" ] || { echo "missing $binary" >&2; return 1; }

    spin
    OMP_NUM_THREADS="$threads" PARLAY_NUM_THREADS="$threads" \
        taskset -c "0-$((threads - 1))" "$binary" \
        --datasets "$inputs" --library "$lib" --workload "$workload" --threads "$threads" \
        >> "$out" 2>> "$OUT/stderr.log"
}

for dim in ${FFCL_MATRIX_DIMS:-2 3 4}; do
    for lib in $SERIAL_LIBS; do
        for workload in $(workloads_for "$lib"); do
            run_one "$lib" "$workload" "$dim" 1
        done
        echo "  ${lib} ${dim}d done"
    done
    for lib in $PARALLEL_LIBS; do
        for threads in ${FFCL_MATRIX_THREADS:-1 6}; do
            for workload in $(workloads_for "$lib"); do
                run_one "$lib" "$workload" "$dim" "$threads"
            done
        done
        echo "  ${lib} ${dim}d done"
    done
done
