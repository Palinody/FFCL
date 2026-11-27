#!/bin/bash
# FFCL DBSCAN at min_samples=4, the value that matches the reference libraries' min_pts=5.
#
#   benchmark/references/run_ffcl_dbscan4.sh <build_dir> <output_dir>
#
# FFCL's core test is strictly `size() > min_samples` and its radius buffer includes the query
# sample, so FFCL min_samples=5 means 6 points including itself while every other library's
# min_pts=5 means 5 including itself. The rest of the FFCL matrix is measured by
# benchmark/scripts/run_matrix.sh; this only re-runs the DBSCAN workload at the matched value.
set -euo pipefail

BUILD="${1:?build directory}"
OUT="${2:?output directory}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LEAF=40
MIN_SAMPLES=4

mkdir -p "$OUT"

spin() { taskset -c 0 bash -c 'end=$((SECONDS+1)); while [ $SECONDS -lt $end ]; do :; done'; }

for dim in ${FFCL_MATRIX_DIMS:-2 3 4}; do
    inputs="$REPO/benchmark/bin/datasets_${dim}d/inputs"
    for threads in ${FFCL_MATRIX_THREADS:-1 6}; do
        out="$OUT/ffcl_${dim}d_l${LEAF}_t${threads}.json"
        : > "$out"

        spin
        OMP_NUM_THREADS="$threads" FFCL_THREADS="$threads" \
            FFCL_DATASETS_DIR="$inputs" FFCL_RESULTS_DIR="$OUT" FFCL_LEAF_SIZE="$LEAF" \
            FFCL_DBSCAN_MIN_SAMPLES="$MIN_SAMPLES" \
            taskset -c "0-$((threads - 1))" "$BUILD/ffcl_benchmark" dbscan >/dev/null 2>&1

        python3 - "$OUT/ffcl.json" "$threads" >> "$out" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    row = json.loads(line)
    row["threads"] = int(sys.argv[2])
    print(json.dumps(row))
PY
        echo "  ffcl dbscan min_samples=${MIN_SAMPLES} ${dim}d t${threads}: $(wc -l < "$out") measurements"
    done
done
