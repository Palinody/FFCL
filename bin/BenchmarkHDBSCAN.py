"""Benchmark hdbscan (sklearn-contrib) the same way
test/src/datastruct/graph/spanning_tree/BoruvkasAlgorithmTest.cpp benchmarks FFCL.

Both sides use, per dataset:
  * the same generated inputs (bin/MakeClusteringDatasets.py),
  * the same tree parameter: bucket_size / leaf_size = sqrt(n_samples),
  * the same HDBSCAN parameters: k_nearest_neighbors / min_samples = 1,
    min_cluster_size = 20,
  * the same measured phases: tree build, minimum spanning tree, full pipeline.

hdbscan's public HDBSCAN class does not expose the MST on its own, so the
Boruvka object is driven directly (this is exactly what
hdbscan.hdbscan_._hdbscan_boruvka_kdtree does internally):

    tree = KDTree(X, leaf_size=leaf_size)
    alg  = KDTreeBoruvkaAlgorithm(tree, min_samples, leaf_size=leaf_size // 3, ...)
    mst  = alg.spanning_tree()

Note that hdbscan builds the Boruvka tree with `leaf_size // 3` while the
core-distance tree uses `leaf_size`; both values are reported.

Run the FFCL counterpart with:
    ./build/BoruvkasBenchmark            (or the BoruvkasAlgorithmTest)
"""

import math
import os
import sys
import time

import numpy as np

try:
    import hdbscan
    from hdbscan._hdbscan_boruvka import KDTreeBoruvkaAlgorithm
    from sklearn.neighbors import KDTree
except ImportError:  # pragma: no cover
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "hdbscan", "scikit-learn"])
    import hdbscan
    from hdbscan._hdbscan_boruvka import KDTreeBoruvkaAlgorithm
    from sklearn.neighbors import KDTree

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clustering/inputs/")

FILE_NAMES = [
    "noisy_circles.txt",
    "noisy_moons.txt",
    "varied.txt",
    "aniso.txt",
    "blobs.txt",
    "no_structure.txt",
    "unbalanced_blobs.txt",
]

# Same values as the FFCL tests: k_nearest_neighbors(1), min_cluster_size(20).
MIN_SAMPLES = 1
MIN_CLUSTER_SIZE = 20
N_REPEATS = 3


def read_dataset(filepath: str):
    # hdbscan casts to float64 internally anyway; keep float32 to mirror FFCL.
    return np.loadtxt(filepath, dtype=np.float32, delimiter=" ")


def bench(fn, n_repeats=N_REPEATS):
    """Best-of-N wall time, like the FFCL benchmarks report a single run."""
    best = math.inf
    result = None
    for _ in range(n_repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def run_dataset(name: str):
    path = os.path.join(ROOT_FOLDER, name)
    data = read_dataset(path)
    n_samples, n_features = data.shape

    # FFCL: OptionsType().bucket_size(std::sqrt(n_samples)).max_depth(std::log2(n_samples))
    leaf_size = int(math.sqrt(n_samples))
    boruvka_leaf_size = max(leaf_size // 3, 3)  # what hdbscan does internally

    print("---")
    print(f"Dataset name: '{name}'")
    print(f"Dataset shape: {data.shape}")
    print(f"leaf_size (core-distance tree): {leaf_size}")
    print(f"leaf_size (boruvka tree):       {boruvka_leaf_size}")
    print(f"HDBSCAN parameters: min_samples={MIN_SAMPLES} min_cluster_size={MIN_CLUSTER_SIZE}")

    # --- 1) tree build -------------------------------------------------------
    t_tree, tree = bench(lambda: KDTree(data, leaf_size=leaf_size))
    print(f"  tree build          : {t_tree * 1e3:9.3f} ms")

    # --- 2) minimum spanning tree only --------------------------------------
    def build_mst():
        alg = KDTreeBoruvkaAlgorithm(
            tree,
            MIN_SAMPLES,
            metric="euclidean",
            leaf_size=boruvka_leaf_size,
            approx_min_span_tree=False,
            n_jobs=1,
        )
        return alg.spanning_tree()

    t_mst, mst = bench(build_mst)
    print(f"  MST (boruvka_kdtree): {t_mst * 1e3:9.3f} ms   edges={len(mst)} weight={mst[:, 2].sum():.6f}")

    # --- 3) full pipeline ----------------------------------------------------
    def full_pipeline():
        cl = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES,
            cluster_selection_epsilon=0,
            core_dist_n_jobs=1,
            allow_single_cluster=True,
            algorithm="boruvka_kdtree",
            gen_min_span_tree=True,
            approx_min_span_tree=False,
            leaf_size=leaf_size,
        )
        return cl.fit_predict(data)

    t_full, labels = bench(full_pipeline)
    n_clusters = len(set(labels.tolist()) - {-1})
    print(
        f"  full HDBSCAN        : {t_full * 1e3:9.3f} ms   "
        f"clusters={n_clusters} noise={(labels == -1).sum()}"
    )
    return t_mst, t_full


def run_all():
    print(f"{'dataset':20s} {'n':>6s} {'leaf':>5s} {'MST ms':>9s} {'full ms':>9s}")
    print("-" * 54)
    totals = []
    for name in FILE_NAMES:
        t_mst, t_full = run_dataset(name)
        n = np.loadtxt(os.path.join(ROOT_FOLDER, name), dtype=np.float32, delimiter=" ").shape[0]
        print(f"{name.replace('.txt', ''):20s} {n:6d} {int(math.sqrt(n)):5d} {t_mst * 1e3:9.3f} {t_full * 1e3:9.3f}")
        totals.append((t_mst, t_full))
    print("-" * 54)
    print(
        f"{'mean':20s} {'':6s} {'':5s} "
        f"{1e3 * sum(t[0] for t in totals) / len(totals):9.3f} "
        f"{1e3 * sum(t[1] for t in totals) / len(totals):9.3f}"
    )


if __name__ == "__main__":
    run_all()
