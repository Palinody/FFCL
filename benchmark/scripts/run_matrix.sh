#!/bin/bash
# Runs the whole comparison matrix: FFCL serial and parallel, single and dual tree, at the tuned leaf
# size and at the reference leaf size, over 2-, 3- and 4-dimensional datasets, plus the reference
# libraries on the same data.
#
#   benchmark/scripts/run_matrix.sh <build_dir> <output_dir>
#
# Expects benchmark/bin/datasets_<d>d/inputs to exist for d in 2 3 4 (see MakeClusteringDatasets.py
# and FFCL_N_FEATURES / FFCL_DATASETS_ROOT).
set -euo pipefail

BUILD="${1:?build directory}"
OUT="${2:?output directory}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORKLOADS="kdtree dbscan mst hdbscan kmeans"
LEAF_REFERENCE=40
LEAF_TUNED=16
THREADS=6

mkdir -p "$OUT"

spin() { taskset -c 0 bash -c 'end=$((SECONDS+1)); while [ $SECONDS -lt $end ]; do :; done'; }

for dim in ${FFCL_MATRIX_DIMS:-2 3 4}; do
    inputs="$REPO/benchmark/bin/datasets_${dim}d/inputs"
    [ -d "$inputs" ] || inputs="$REPO/benchmark/bin/datasets/inputs"

    for leaf in ${FFCL_MATRIX_LEAVES:-"$LEAF_REFERENCE $LEAF_TUNED"}; do
        for threads in 1 "$THREADS"; do
            label="ffcl_${dim}d_l${leaf}_t${threads}"
            out="$OUT/$label.json"
            : > "$out"

            spin

            for workload in $WORKLOADS; do
                # OpenMP reads OMP_NUM_THREADS, and the Boruvka step only goes parallel when the option is
                # set, so both have to follow the row being measured.
                OMP_NUM_THREADS="$threads" FFCL_THREADS="$threads" \
                    FFCL_DATASETS_DIR="$inputs" FFCL_RESULTS_DIR="$OUT" FFCL_LEAF_SIZE="$leaf" \
                    taskset -c 0-$((threads - 1)) "$BUILD/ffcl_benchmark" "$workload" >/dev/null 2>&1

                python3 - "$OUT/ffcl.json" "$threads" <<'PY' >> "$out"
import json, sys
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    row = json.loads(line)
    row["threads"] = int(sys.argv[2])
    print(json.dumps(row))
PY
            done
            echo "  $label: $(wc -l < "$out") measurements"
        done
    done

    FFCL_DATASETS_DIR="$inputs" FFCL_RESULTS_DIR="$OUT" \
        python3 "$REPO/benchmark/scripts/benchmark_reference.py" >/dev/null 2>&1
    mv "$OUT/reference.json" "$OUT/reference_${dim}d.json"
    echo "  reference ${dim}d done"
done
