"""Side-by-side benchmark: FFCL (sequential / parallel) against hdbscan (Python).

Both libraries are driven at the same stage granularity:

  * "MST"  is FFCL `BoruvkasAlgorithm::make_tree_2` versus hdbscan's
           `KDTreeBoruvkaAlgorithm.spanning_tree()`.
  * "full" is FFCL `HDBSCAN::predict` versus `hdbscan.HDBSCAN.fit_predict`.

The leaf size is the same for both libraries (`sqrt(n_samples) / 3`, which is
what hdbscan uses internally for its Boruvka tree when `leaf_size =
sqrt(n_samples)`), so the numbers are directly comparable.

FFCL is run through two pre-built binaries so that the sequential and the
OpenMP build can be compared without recompiling:

    build_seq/BoruvkasRunner   (-DTHREADS_ENABLED=false)
    build_par/BoruvkasRunner   (-DTHREADS_ENABLED=true)

Usage:
    python3 bin/CompareHDBSCAN.py --seq-binary ... --par-binary ... [--repeats N]
"""

import argparse
import math
import os
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
    import subprocess as _sp

    _sp.check_call([sys.executable, "-m", "pip", "install", "hdbscan", "scikit-learn"])
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


def best_of(fn, repeats):
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def run_ffcl(binary, path, mode, bucket, repeats):
    if not binary or not os.path.exists(binary):
        return math.nan

    def once():
        subprocess.run(
            [binary, path, "-", mode, "1", str(bucket)],
            capture_output=True,
            check=False,
        )

    return best_of(once, repeats)


def run_hdbscan(data, bucket, repeats, jobs):
    leaf_size = int(math.sqrt(len(data)))
    tree = KDTree(data, leaf_size=leaf_size)

    def mst_once():
        # `KDTreeBoruvkaAlgorithm.__init__` computes the core distances and `spanning_tree()` runs the
        # Boruvka loop. FFCL's `make_tree_2` does both, so both are timed here to keep the row comparable.
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

    return best_of(mst_once, repeats), best_of(full_once, repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-binary", default="")
    parser.add_argument("--par-binary", default="")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--bucket", type=int, default=0, help="override the leaf size for both libraries")
    args = parser.parse_args()

    rows = []
    for name in NAMES:
        path = os.path.join(ROOT, name + ".txt")
        data = np.loadtxt(path, dtype=np.float32, delimiter=" ")
        n = len(data)
        bucket = args.bucket if args.bucket else max(16, int(math.sqrt(n)) // 3)

        ffcl_seq_mst = run_ffcl(args.seq_binary, path, "dual", bucket, args.repeats)
        ffcl_par_mst = run_ffcl(args.par_binary, path, "dual", bucket, args.repeats)
        ffcl_seq_full = run_ffcl(args.seq_binary, path, "full", bucket, args.repeats)
        ffcl_par_full = run_ffcl(args.par_binary, path, "full", bucket, args.repeats)

        sk_seq_mst, sk_seq_full = run_hdbscan(data, bucket, args.repeats, 1)
        sk_par_mst, sk_par_full = run_hdbscan(data, bucket, args.repeats, -1)

        rows.append(
            dict(
                name=name,
                n=n,
                bucket=bucket,
                ffcl_seq_mst=ffcl_seq_mst,
                ffcl_par_mst=ffcl_par_mst,
                ffcl_seq_full=ffcl_seq_full,
                ffcl_par_full=ffcl_par_full,
                sk_seq_mst=sk_seq_mst,
                sk_par_mst=sk_par_mst,
                sk_seq_full=sk_seq_full,
                sk_par_full=sk_par_full,
            )
        )
        r = rows[-1]
        print(
            f"{name:20s} ffcl {r['ffcl_seq_mst']:.3f}/{r['ffcl_par_mst']:.3f} s  "
            f"hdbscan {r['sk_seq_mst']:.3f}/{r['sk_par_mst']:.3f} s",
            flush=True,
        )

    def mean(key):
        values = [r[key] for r in rows if not math.isnan(r[key])]
        return sum(values) / len(values) if values else math.nan

    print()
    print("=" * 108)
    print("MST construction, seconds (best of {})".format(args.repeats))
    print("=" * 108)
    header = (
        f"{'dataset':20s} {'n':>7s} {'leaf':>5s} "
        f"{'FFCL seq':>10s} {'FFCL par':>10s} {'FFCL x':>7s} "
        f"{'sk seq':>10s} {'sk par':>10s} {'sk x':>7s} "
        f"{'seq F/sk':>9s} {'par F/sk':>9s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        fx = r["ffcl_seq_mst"] / r["ffcl_par_mst"]
        sx = r["sk_seq_mst"] / r["sk_par_mst"]
        print(
            f"{r['name']:20s} {r['n']:7d} {r['bucket']:5d} "
            f"{r['ffcl_seq_mst']:10.3f} {r['ffcl_par_mst']:10.3f} {fx:6.2f}x "
            f"{r['sk_seq_mst']:10.3f} {r['sk_par_mst']:10.3f} {sx:6.2f}x "
            f"{r['ffcl_seq_mst'] / r['sk_seq_mst']:8.2f}x {r['ffcl_par_mst'] / r['sk_par_mst']:8.2f}x"
        )
    print("-" * len(header))
    print(
        f"{'mean':20s} {'':7s} {'':5s} "
        f"{mean('ffcl_seq_mst'):10.3f} {mean('ffcl_par_mst'):10.3f} "
        f"{mean('ffcl_seq_mst') / mean('ffcl_par_mst'):6.2f}x "
        f"{mean('sk_seq_mst'):10.3f} {mean('sk_par_mst'):10.3f} "
        f"{mean('sk_seq_mst') / mean('sk_par_mst'):6.2f}x "
        f"{mean('ffcl_seq_mst') / mean('sk_seq_mst'):8.2f}x "
        f"{mean('ffcl_par_mst') / mean('sk_par_mst'):8.2f}x"
    )

    print()
    print("=" * 108)
    print("Full HDBSCAN pipeline, seconds (best of {})".format(args.repeats))
    print("=" * 108)
    print(header.replace("MST", "full").replace("seq F/sk", "seq F/sk").replace("par F/sk", "par F/sk"))
    print("-" * len(header))
    for r in rows:
        fx = r["ffcl_seq_full"] / r["ffcl_par_full"]
        sx = r["sk_seq_full"] / r["sk_par_full"]
        print(
            f"{r['name']:20s} {r['n']:7d} {r['bucket']:5d} "
            f"{r['ffcl_seq_full']:10.3f} {r['ffcl_par_full']:10.3f} {fx:6.2f}x "
            f"{r['sk_seq_full']:10.3f} {r['sk_par_full']:10.3f} {sx:6.2f}x "
            f"{r['ffcl_seq_full'] / r['sk_seq_full']:8.2f}x {r['ffcl_par_full'] / r['sk_par_full']:8.2f}x"
        )
    print("-" * len(header))
    print(
        f"{'mean':20s} {'':7s} {'':5s} "
        f"{mean('ffcl_seq_full'):10.3f} {mean('ffcl_par_full'):10.3f} "
        f"{mean('ffcl_seq_full') / mean('ffcl_par_full'):6.2f}x "
        f"{mean('sk_seq_full'):10.3f} {mean('sk_par_full'):10.3f} "
        f"{mean('sk_seq_full') / mean('sk_par_full'):6.2f}x "
        f"{mean('ffcl_seq_full') / mean('sk_seq_full'):8.2f}x "
        f"{mean('ffcl_par_full') / mean('sk_par_full'):8.2f}x"
    )
    print()
    print("'x' columns are the sequential -> parallel speedup; 'F/sk' columns are FFCL time / hdbscan time")
    print("(below 1.00 means FFCL is faster).")


if __name__ == "__main__":
    main()
