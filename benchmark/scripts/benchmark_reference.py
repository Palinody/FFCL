"""Reference measurements for the FFCL benchmark.

Runs the same workload as benchmark/src/ (same datasets, same parameters, same timing policy: three
tries, mean and median) against the reference implementations, and writes one JSON line per
measurement to benchmark/bin/results/reference.json.

The two sides write the same schema so benchmark/scripts/make_tables.py can merge them:

    {"library", "algorithm", "stage", "dataset", "n", "leaf", "threads",
     "tries", "mean", "median", "note"}

Reference libraries, and what each one stands in for:

    sklearn.neighbors.KDTree  general-purpose KD-tree, kNN
    scipy.spatial.cKDTree     the other widely used C KD-tree, kNN
    FLANN                     measured on the C++ side (benchmark/src/reference)
    sklearn.cluster.DBSCAN    DBSCAN
    hdbscan                   HDBSCAN*, the library FFCL's HDBSCAN is modelled on
    sklearn.cluster.KMeans    KMeans (Lloyd)
    sklearn_extra.KMedoids    KMedoids, if it happens to be installed

Not measured here: PCL (its KD-tree is a FLANN wrapper, so the FLANN row already covers it) and
MLPACK (not installed; the hdbscan row is its Boruvka implementation ported to Cython).
"""

import json
import os
import statistics
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPTS_DIR)
DATASETS_DIR = os.environ.get("FFCL_DATASETS_DIR") or os.path.join(BENCHMARK_DIR, "bin", "datasets", "inputs")
RESULTS_DIR = os.environ.get("FFCL_RESULTS_DIR") or os.path.join(BENCHMARK_DIR, "bin", "results")

# Must match benchmark/src/main.cpp
N_QUERIES = 1000
K_NEAREST_NEIGHBORS = 2
MIN_CLUSTER_SIZE = 20
MIN_SAMPLES = 1
N_CENTROIDS = 8
N_MEDOIDS = 8
DBSCAN_RADIUS = 0.5
DBSCAN_MIN_SAMPLES = 5
TRIES = 3

# One reference leaf size for every dataset and library, as agreed.
REFERENCE_LEAF_SIZE = int(os.environ.get("FFCL_REFERENCE_LEAF_SIZE") or 40)

DATASET_NAMES = [
    "noisy_circles",
    "noisy_moons",
    "varied",
    "aniso",
    "blobs",
    "no_structure",
    "unbalanced_blobs",
]


def time_workload(workload, tries=TRIES):
    samples = []
    for _ in range(tries):
        start = time.perf_counter()
        workload()
        samples.append(time.perf_counter() - start)
    return statistics.fmean(samples), statistics.median(samples)


def record(out, library, algorithm, stage, dataset, n, leaf, tries, mean, median, note):
    out.write(
        json.dumps(
            {
                "library": library,
                "algorithm": algorithm,
                "stage": stage,
                "dataset": dataset,
                "n": n,
                "leaf": leaf,
                "threads": 1,
                "tries": tries,
                "mean": mean,
                "median": median,
                "note": note,
            }
        )
        + "\n"
    )


def load(name):
    return np.loadtxt(os.path.join(DATASETS_DIR, name + ".txt"), dtype=np.float32, delimiter=" ")


def benchmark_kdtree(out):
    from sklearn.neighbors import KDTree as SklearnKDTree
    from scipy.spatial import cKDTree

    for name in DATASET_NAMES:
        data = load(name)
        n = len(data)
        queries = min(N_QUERIES, n)
        query_data = data[:queries]

        # --- sklearn KDTree ---------------------------------------------------
        tree = SklearnKDTree(data, leaf_size=REFERENCE_LEAF_SIZE)

        mean, median = time_workload(lambda: SklearnKDTree(data, leaf_size=REFERENCE_LEAF_SIZE))
        record(out, "sklearn", "kdtree", "tree_build", name, n, REFERENCE_LEAF_SIZE, TRIES, mean, median,
               f"queries={queries} k={K_NEAREST_NEIGHBORS}")

        mean, median = time_workload(
            lambda: tree.query(query_data, k=K_NEAREST_NEIGHBORS, dualtree=False, breadth_first=False)
        )
        record(out, "sklearn", "kdtree", "knn", name, n, REFERENCE_LEAF_SIZE, TRIES, mean, median,
               f"queries={queries} k={K_NEAREST_NEIGHBORS} strategy=single_tree")

        # --- scipy cKDTree ----------------------------------------------------
        tree = cKDTree(data, leafsize=REFERENCE_LEAF_SIZE)

        mean, median = time_workload(lambda: cKDTree(data, leafsize=REFERENCE_LEAF_SIZE))
        record(out, "scipy", "kdtree", "tree_build", name, n, REFERENCE_LEAF_SIZE, TRIES, mean, median,
               f"queries={queries} k={K_NEAREST_NEIGHBORS}")

        mean, median = time_workload(lambda: tree.query(query_data, k=K_NEAREST_NEIGHBORS, workers=1))
        record(out, "scipy", "kdtree", "knn", name, n, REFERENCE_LEAF_SIZE, TRIES, mean, median,
               f"queries={queries} k={K_NEAREST_NEIGHBORS}")


def benchmark_dbscan(out):
    from sklearn.cluster import DBSCAN as SklearnDBSCAN

    for name in DATASET_NAMES:
        data = load(name)

        def run():
            return SklearnDBSCAN(eps=DBSCAN_RADIUS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(data)

        mean, median = time_workload(run)
        record(out, "sklearn", "dbscan", "full", name, len(data), 0, TRIES, mean, median,
               f"radius={DBSCAN_RADIUS} min_samples={DBSCAN_MIN_SAMPLES}")


def benchmark_hdbscan(out):
    import hdbscan
    from hdbscan._hdbscan_boruvka import KDTreeBoruvkaAlgorithm
    from sklearn.neighbors import KDTree

    for name in DATASET_NAMES:
        data = load(name)
        n = len(data)

        # The MST on its own. hdbscan computes the core distances in the Boruvka object's constructor,
        # so construction and spanning_tree() are timed together to match FFCL's make_tree_2, which
        # also includes them.
        tree = KDTree(data, leaf_size=REFERENCE_LEAF_SIZE)

        def run_mst():
            algorithm = KDTreeBoruvkaAlgorithm(
                tree,
                MIN_SAMPLES,
                metric="euclidean",
                leaf_size=REFERENCE_LEAF_SIZE,
                approx_min_span_tree=False,
                n_jobs=1,
            )
            return algorithm.spanning_tree()

        mean, median = time_workload(run_mst)
        record(out, "hdbscan", "hdbscan", "mst", name, n, REFERENCE_LEAF_SIZE, TRIES, mean, median,
               f"k_nearest_neighbors={MIN_SAMPLES + 1} approx=False")

        def run_full():
            return hdbscan.HDBSCAN(
                min_cluster_size=MIN_CLUSTER_SIZE,
                min_samples=MIN_SAMPLES,
                core_dist_n_jobs=1,
                allow_single_cluster=True,
                algorithm="boruvka_kdtree",
                approx_min_span_tree=False,
                leaf_size=REFERENCE_LEAF_SIZE,
            ).fit_predict(data)

        mean, median = time_workload(run_full)
        record(out, "hdbscan", "hdbscan", "full", name, n, REFERENCE_LEAF_SIZE, TRIES, mean, median,
               f"k_nearest_neighbors={MIN_SAMPLES + 1} min_cluster_size={MIN_CLUSTER_SIZE}")


def benchmark_kmeans(out):
    from sklearn.cluster import KMeans as SklearnKMeans

    for name in DATASET_NAMES:
        data = load(name)

        def run():
            # n_init=1 and the same initialisation strategy FFCL uses (k-means++) so the work is the same
            return SklearnKMeans(
                n_clusters=N_CENTROIDS, init="k-means++", n_init=1, max_iter=100, algorithm="lloyd"
            ).fit_predict(data)

        mean, median = time_workload(run)
        record(out, "sklearn", "kmeans", "full", name, len(data), 0, TRIES, mean, median,
               f"n_centroids={N_CENTROIDS} init=k-means++ n_init=1 algorithm=lloyd")


def benchmark_kmedoids(out):
    try:
        from sklearn_extra.cluster import KMedoids as SklearnKMedoids
    except ImportError:
        print("  sklearn_extra not installed: no KMedoids reference, only the FFCL row will be reported")
        return

    for name in DATASET_NAMES:
        data = load(name)

        def run():
            return SklearnKMedoids(
                n_clusters=N_MEDOIDS, method="pam", init="k-medoids++", max_iter=100
            ).fit_predict(data)

        mean, median = time_workload(run)
        record(out, "sklearn_extra", "kmedoids", "full", name, len(data), 0, TRIES, mean, median,
               f"n_medoids={N_MEDOIDS} method=pam")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    stages = {
        "kdtree": benchmark_kdtree,
        "dbscan": benchmark_dbscan,
        "hdbscan": benchmark_hdbscan,
        "kmeans": benchmark_kmeans,
        "kmedoids": benchmark_kmedoids,
    }

    output = os.path.join(RESULTS_DIR, "reference.json")
    with open(output, "w") as out:
        for stage, function in stages.items():
            if which not in ("all", stage):
                continue
            print(f"{stage}...")
            function(out)

    print(f"results written to {output}")


if __name__ == "__main__":
    main()
