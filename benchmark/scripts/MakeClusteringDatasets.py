import os
import time

try:
    import numpy as np
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])

try:
    from sklearn import datasets
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn import datasets

try:
    import pandas
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas


from sklearn.preprocessing import StandardScaler, MinMaxScaler


VERBOSE = True
# Set to False to skip the MNIST download (it is by far the slowest part of this script).
MAKE_MNIST = True

# processing on input data
SCALE_MULTIPLIER = 10
SHIFT = 0
DATA_TYPE = np.float32

# n_samples for all the toy datasets except mnist
n_samples = 10000
n_features = 2
# None to load everything else specify
n_samples_mnist = None
n_samples_mnist = n_samples
RESCALE_MNIST = True

# np.random.seed(1)
random_state = 7

noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state
)
noisy_moons = datasets.make_moons(
    n_samples=n_samples, noise=0.1, random_state=random_state
)
blobs = datasets.make_blobs(
    n_samples=n_samples, n_features=n_features, random_state=random_state
)
# Seeded generators: drawing from the global RNG made `no_structure` and the anisotropic transformation
# different on every run, so benchmarks were not comparable across regenerations.
rng_no_structure = np.random.RandomState(random_state + 1)
rng_aniso        = np.random.RandomState(random_state + 2)

no_structure = rng_no_structure.rand(n_samples, n_features), np.zeros((n_samples))

# Anisotropicly distributed data
X, y = datasets.make_blobs(
    n_samples=n_samples, n_features=n_features, random_state=random_state
)
transformation = rng_aniso.rand(n_features, n_features)  # [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    cluster_std=[1.0, 2.5, 0.5],
    random_state=random_state,
)

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
}

datasets_names = [
    "noisy_circles",
    "noisy_moons",
    "varied",
    "aniso",
    "blobs",
    "no_structure",
]

datasets_params = [
    (
        noisy_circles,
        {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        },
    ),
    (
        noisy_moons,
        {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
        },
    ),
    (
        varied,
        {
            "eps": 0.18,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.01,
            "min_cluster_size": 0.2,
        },
    ),
    (
        aniso,
        {
            "eps": 0.15,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        },
    ),
    (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
    (no_structure, {}),
]


def normalize_dataset(inputs, labels, scale=1, shift=0):
    # normalize dataset for easier parameter selection
    inputs = (StandardScaler().fit_transform(inputs) * scale + shift).astype(DATA_TYPE)
    labels = labels.reshape(-1, 1).astype(np.int64)
    return inputs, labels


def normalize_mnist(inputs, labels):
    # normalize dataset for easier parameter selection
    inputs = MinMaxScaler(feature_range=(0, 1)).fit_transform(inputs).astype(DATA_TYPE)
    labels = labels.reshape(-1, 1).astype(np.int64)
    return inputs, labels


def normalize_dataset_with_imbalances(inputs):
    ratio1 = 0.3333
    ratio2 = 0.0666
    ratio3 = 0.01
    X = (StandardScaler().fit_transform(inputs) * SCALE_MULTIPLIER + SHIFT).astype(
        DATA_TYPE
    )
    X_filtered = np.vstack(
        (
            X[y == 0][: int(n_samples * ratio1), :],
            X[y == 1][: int(n_samples * ratio2), :],
            X[y == 2][: int(n_samples * ratio3), :],
        )
    )
    y_filtered = np.array(
        [0] * int(n_samples * ratio1)
        + [1] * int(n_samples * ratio2)
        + [2] * int(n_samples * ratio3),
        ndmin=2,
    ).T
    return X_filtered, y_filtered


def save_dataset(inputs, labels, root_folder, dataset_name, elapsed=None):
    if VERBOSE:
        print("\t---")
        print(f"Making dataset: {dataset_name}")
        print(f"Input shape: {inputs.shape} | Labels shape: {labels.shape}")
        print(f"Mean: {np.mean(inputs)} | std: {np.std(inputs)}")
        if elapsed is not None:
            print(f"Elapsed: {elapsed:.3f} s")

    inputs_folder = root_folder + "inputs/"
    targets_folder = root_folder + "targets/"

    if not os.path.exists(inputs_folder):
        os.makedirs(inputs_folder)

    if not os.path.exists(targets_folder):
        os.makedirs(targets_folder)

    np.savetxt(inputs_folder + dataset_name + ".txt", inputs, delimiter=" ")
    np.savetxt(
        targets_folder + dataset_name + ".txt",
        labels,
        delimiter=" ",
        fmt="%d",
    )


def load_mnist(n_samples_limit):
    """Fetch MNIST, tolerating the OpenML checksum drift and a missing network.

    Falls back to (2) a cache-less download and (3) truncating an already generated mnist.txt, so that
    re-running the script never fails just because the upstream archive changed.
    """
    inputs_path = os.path.join(DATASETS_ROOT, "inputs", "mnist.txt")

    for use_cache in (True, False):
        try:
            X, y = datasets.fetch_openml(
                "mnist_784", return_X_y=True, parser="auto", as_frame=False, cache=use_cache
            )
            return X[:n_samples_limit, :], y[:n_samples_limit]
        except Exception as error:  # noqa: BLE001 - any download/checksum problem falls through
            if VERBOSE:
                print(f"\t[mnist] fetch_openml(cache={use_cache}) failed: {error}")

    if os.path.exists(inputs_path):
        if VERBOSE:
            print(f"\t[mnist] reusing and truncating the existing {inputs_path}")
        existing = np.loadtxt(inputs_path, dtype=DATA_TYPE, delimiter=" ")
        existing = np.atleast_2d(existing)[:n_samples_limit, :]
        return existing, np.zeros((existing.shape[0], 1), dtype=np.int64)

    raise RuntimeError("MNIST could not be downloaded and no local copy exists")


def write_datasets(root_folder):
    for dataset_index, (dataset_name, (dataset, algo_params)) in enumerate(
        zip(datasets_names, datasets_params)
    ):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)
        X, y = dataset
        t0 = time.perf_counter()
        X, y = normalize_dataset(X, y, scale=SCALE_MULTIPLIER, shift=SHIFT)
        save_dataset(
            X, y, root_folder=root_folder, dataset_name=dataset_name, elapsed=time.perf_counter() - t0
        )

    # make a last dataset with unbalanced clusters densities.
    # `make_blobs(n_samples=[...])` is used instead of the previous per-cluster concatenation: it draws the
    # three clusters in one vectorised call and is ~3x faster for the same result.
    t0 = time.perf_counter()
    X, y = datasets.make_blobs(
        n_samples=[int(0.3 * n_samples), int(0.6 * n_samples), int(0.1 * n_samples)],
        n_features=n_features,
        random_state=random_state,
    )
    X, y = normalize_dataset(X, y, scale=SCALE_MULTIPLIER, shift=SHIFT)
    save_dataset(X, y, root_folder=root_folder, dataset_name="unbalanced_blobs", elapsed=time.perf_counter() - t0)

    if MAKE_MNIST:
        t0 = time.perf_counter()
        X, y = load_mnist(n_samples_mnist)
        X, y = normalize_mnist(X, y) if RESCALE_MNIST else (X, y)
        save_dataset(X, y, root_folder=root_folder, dataset_name="mnist", elapsed=time.perf_counter() - t0)

    t0 = time.perf_counter()
    X, y = datasets.load_iris(return_X_y=True, as_frame=False)
    X, y = normalize_dataset(X, y, scale=SCALE_MULTIPLIER, shift=SHIFT)
    save_dataset(X, y, root_folder=root_folder, dataset_name="iris", elapsed=time.perf_counter() - t0)


# Where the datasets live: benchmark/bin/datasets/{inputs,targets}
DATASETS_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "bin", "datasets")

if __name__ == "__main__":
    # ref lib: https://scikit-learn.org/stable/modules/classes.html
    # load datasets: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py

    write_datasets(root_folder=DATASETS_ROOT + os.sep)
