"""Side-by-side benchmark: FFCL (sequential / parallel) against hdbscan (Python).

Both libraries are driven at the same stage granularity:

  * "MST"  is FFCL `BoruvkasAlgorithm::make_tree_2` (core distances + Boruvka)
           versus hdbscan `KDTreeBoruvkaAlgorithm.__init__` + `spanning_tree()`
           (hdbscan computes the core distances in the constructor, so both are
           included to keep the row comparable).
  * "full" is FFCL `HDBSCAN::predict` versus `hdbscan.HDBSCAN.fit_predict`.

Both use the same leaf size (`sqrt(n_samples) / 3`, which is what hdbscan uses
internally for its Boruvka tree when `leaf_size = sqrt(n_samples)`).

FFCL is driven through two pre-built binaries so the sequential and the OpenMP
build can be compared without recompiling:

    cmake .. -DMODE=gtest -DTHREADS_ENABLED=false && make BoruvkasRunner
    cp build/BoruvkasRunner /tmp/ffcl_seq
    cmake .. -DMODE=gtest -DTHREADS_ENABLED=true  && make BoruvkasRunner
    cp build/BoruvkasRunner /tmp/ffcl_par

    python3 bin/CompareHDBSCAN.py --seq-binary /tmp/ffcl_seq --par-binary /tmp/ffcl_par --repeats 5

Every repeat is recorded, so the report shows the mean and the median.
"""

import argparse
import math
import os
import statistics
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

try:
    import hdbscan
    from hdbscan._hdbscan_boruvka import KDTreeBoruvkaAlgorithm
    from sklearn.neighbors import KDTree
except ImportError:  # pragma: no cover
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hdbscan", "scikit-learn"])
    import hdbscan
    from hdbscan._hdbscan_boruvka import KDTreeBoruvkaAlgorithm
    from sklearn.neighbors import KDTree

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clustering/inputs")
NAMES = [
    "noisy_circles",
    "noisy_moons",
    "varied",
    "aniso",
    "blobs",
    "no_structure",
    "unbalanced_blobs",
]
MIN_SAMPLES = 1
MIN_CLUSTER_SIZE = 20


def samples_of(fn, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return times


def run_ffcl(binary, path, mode, bucket, repeats):
    if not binary or not os.path.exists(binary):
        return [math.nan] * repeats

    def once():
        subprocess.run([binary, path, "-", mode, "1", str(bucket)], capture_output=True, check=False)

    return samples_of(once, repeats)


def run_hdbscan(data, bucket, repeats, jobs):
    leaf_size = int(math.sqrt(len(data)))
    tree = KDTree(data, leaf_size=leaf_size)

    def mst_once():
        alg = KDTreeBoruvkaAlgorithm(
            tree,
            MIN_SAMPLES,
            metric="euclidean",
            leaf_size=bucket,
            approx_min_span_tree=False,
            n_jobs=jobs,
        )
        return alg.spanning_tree()

    def full_once():
        cl = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES,
            core_dist_n_jobs=jobs,
            allow_single_cluster=True,
            algorithm="boruvka_kdtree",
            gen_min_span_tree=True,
            approx_min_span_tree=False,
            leaf_size=leaf_size,
        )
        return cl.fit_predict(data)

    return samples_of(mst_once, repeats), samples_of(full_once, repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-binary", default="", help="BoruvkasRunner built with THREADS_ENABLED=false")
    parser.add_argument("--par-binary", default="", help="BoruvkasRunner built with THREADS_ENABLED=true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bucket", type=int, default=0, help="override the leaf size for both libraries")
    args = parser.parse_args()

    rows = []
    for name in NAMES:
        path = os.path.join(ROOT, name + ".txt")
        data = np.loadtxt(path, dtype=np.float32, delimiter=" ")
        n = len(data)
        bucket = args.bucket if args.bucket else max(16, int(math.sqrt(n)) // 3)

        row = {"name": name, "n": n, "bucket": bucket}
        row["ffcl_mst_seq"] = run_ffcl(args.seq_binary, path, "dual", bucket, args.repeats)
        row["ffcl_mst_par"] = run_ffcl(args.par_binary, path, "dual", bucket, args.repeats)
        row["ffcl_full_seq"] = run_ffcl(args.seq_binary, path, "full", bucket, args.repeats)
        row["ffcl_full_par"] = run_ffcl(args.par_binary, path, "full", bucket, args.repeats)

        sk_seq_mst, sk_seq_full = run_hdbscan(data, bucket, args.repeats, 1)
        sk_par_mst, sk_par_full = run_hdbscan(data, bucket, args.repeats, -1)

        row["sk_mst_seq"], row["sk_mst_par"] = sk_seq_mst, sk_par_mst
        row["sk_full_seq"], row["sk_full_par"] = sk_seq_full, sk_par_full
        rows.append(row)

        print(
            f"{name:20s} ffcl {statistics.fmean(row['ffcl_mst_seq']):.3f}s  "
            f"hdbscan {statistics.fmean(row['sk_mst_seq']):.3f}s",
            flush=True,
        )

    def report(title, prefix, sk_prefix):
        # Each cell holds the per-run samples, so aggregate per dataset first and then across datasets.
        def avg(key):
            return statistics.fmean([statistics.fmean(r[key]) for r in rows])

        def med(key):
            return statistics.median([statistics.median(r[key]) for r in rows])

        print()
        print("=" * 126)
        print(f"{title} -- seconds, mean/median over {args.repeats} runs per dataset")
        print("=" * 126)
        header = (
            f"{'dataset':20s} {'n':>7s} {'leaf':>5s} "
            f"{'FFCL seq':>17s} {'FFCL par':>17s} {'FFCL x':>7s} "
            f"{'hdbscan seq':>17s} {'hdbscan par':>17s} {'hdbs x':>7s} "
            f"{'F/sk seq':>9s} {'F/sk par':>9s}"
        )
        print(header)
        print("-" * len(header))

        for r in rows:
            fs, fp = r[prefix + "_seq"], r[prefix + "_par"]
            hs, hp = r["sk_" + sk_prefix + "_seq"], r["sk_" + sk_prefix + "_par"]
            print(
                f"{r['name']:20s} {r['n']:7d} {r['bucket']:5d} "
                f"{statistics.fmean(fs):8.3f}/{statistics.median(fs):8.3f} "
                f"{statistics.fmean(fp):8.3f}/{statistics.median(fp):8.3f} "
                f"{statistics.fmean(fs) / statistics.fmean(fp):6.2f}x "
                f"{statistics.fmean(hs):8.3f}/{statistics.median(hs):8.3f} "
                f"{statistics.fmean(hp):8.3f}/{statistics.median(hp):8.3f} "
                f"{statistics.fmean(hs) / statistics.fmean(hp):6.2f}x "
                f"{statistics.fmean(fs) / statistics.fmean(hs):8.2f}x "
                f"{statistics.fmean(fp) / statistics.fmean(hp):8.2f}x"
            )
        print("-" * len(header))
        f_seq, f_par = avg(prefix + "_seq"), avg(prefix + "_par")
        h_seq, h_par = avg("sk_" + sk_prefix + "_seq"), avg("sk_" + sk_prefix + "_par")
        print(
            f"{'mean of means':20s} {'':7s} {'':5s} "
            f"{f_seq:8.3f}/{med(prefix + '_seq'):8.3f} {f_par:8.3f}/{med(prefix + '_par'):8.3f} "
            f"{f_seq / f_par:6.2f}x "
            f"{h_seq:8.3f}/{med('sk_' + sk_prefix + '_seq'):8.3f} "
            f"{h_par:8.3f}/{med('sk_' + sk_prefix + '_par'):8.3f} "
            f"{h_seq / h_par:6.2f}x {f_seq / h_seq:8.2f}x {f_par / h_par:8.2f}x"
        )
        print("  mean of means / median of medians across the datasets")
        print("  'x' = sequential -> parallel speedup; 'F/sk' = FFCL / hdbscan (below 1.00 means FFCL faster)")

    report("MST construction (core distances + Boruvka)", "ffcl_mst", "mst")
    report("Full HDBSCAN pipeline", "ffcl_full", "full")


if __name__ == "__main__":
    main()
