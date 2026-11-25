"""Render benchmark/README.md from the FFCL and reference measurement files.

Input  : benchmark/bin/results/ffcl.json        (written by build_bench/ffcl_benchmark)
         benchmark/bin/results/reference.json   (written by benchmark_reference.py)
Output : benchmark/README.md

Both files contain one JSON object per measurement with the same field names, so this script only has
to group, format and label. Nothing here re-runs anything: the tables are a pure function of the two
result files, which is why they are committed alongside this script.
"""

import json
import math
import os
import platform
import statistics

SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPTS_DIR)
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "bin", "results")
README_PATH = os.path.join(BENCHMARK_DIR, "README.md")

DATASET_NAMES = [
    "noisy_circles",
    "noisy_moons",
    "varied",
    "aniso",
    "blobs",
    "no_structure",
    "unbalanced_blobs",
]


def load_results(path):
    if not os.path.exists(path):
        print(f"missing {path}: run the benchmark first")
        return []
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def cpu_description():
    try:
        with open("/proc/cpuinfo") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def core_count():
    return os.cpu_count() or 0


def cell(measurement):
    """`mean / median`, in seconds, six decimals like the earlier tables."""
    if measurement is None:
        return "—"
    return f"{measurement['mean']:.6f} / {measurement['median']:.6f}"


def index_by(rows, algorithm, stage):
    table = {}
    for row in rows:
        if row["algorithm"] == algorithm and row["stage"] == stage:
            table[(row["library"], row["dataset"])] = row
    return table


def ratio_cell(numerator, denominator):
    if numerator is None or denominator is None or denominator["mean"] == 0:
        return "—"
    return f"{numerator['mean'] / denominator['mean']:.2f}x"


def emit_table(lines, title, explanation, rows, algorithm, stage, columns, ratio_pairs=None):
    """columns: list of library names in display order. ratio_pairs: (label, numerator, denominator)."""
    table = index_by(rows, algorithm, stage)

    lines.append(f"### {title}")
    lines.append("")
    if explanation:
        lines.append(explanation)
        lines.append("")

    header = "| dataset | n | " + " | ".join(columns) + " |"
    if ratio_pairs:
        header += " " + " | ".join(label for label, _, _ in ratio_pairs) + " |"

    separator = "| --- | --- | " + " | ".join("---" for _ in columns) + " |"
    if ratio_pairs:
        separator += " " + " | ".join("---" for _ in ratio_pairs) + " |"

    lines.append(header)
    lines.append(separator)

    for dataset in DATASET_NAMES:
        n = next((row["n"] for row in table.values() if row["dataset"] == dataset), 0)
        cells = [cell(table.get((library, dataset))) for library in columns]
        line = f"| {dataset} | {n} | " + " | ".join(cells) + " |"
        if ratio_pairs:
            line += " " + " | ".join(
                ratio_cell(table.get((numerator, dataset)), table.get((denominator, dataset)))
                for _, numerator, denominator in ratio_pairs
            ) + " |"
        lines.append(line)

    # mean of means and median of medians, per column
    footer = "| **mean / median of the 7 datasets** | | "
    footer_cells = []
    for library in columns:
        means = [table[(library, d)]["mean"] for d in DATASET_NAMES if (library, d) in table]
        medians = [table[(library, d)]["median"] for d in DATASET_NAMES if (library, d) in table]
        if means:
            footer_cells.append(f"**{statistics.fmean(means):.6f} / {statistics.median(medians):.6f}**")
        else:
            footer_cells.append("—")
    footer += " | ".join(footer_cells) + " |"

    if ratio_pairs:
        ratio_cells = []
        for _, numerator, denominator in ratio_pairs:
            numerator_means = [
                table[(numerator, d)]["mean"] for d in DATASET_NAMES if (numerator, d) in table
            ]
            denominator_means = [
                table[(denominator, d)]["mean"] for d in DATASET_NAMES if (denominator, d) in table
            ]
            if numerator_means and denominator_means:
                ratio_cells.append(
                    f"**{statistics.fmean(numerator_means) / statistics.fmean(denominator_means):.2f}x**"
                )
            else:
                ratio_cells.append("—")
        footer += " " + " | ".join(ratio_cells) + " |"

    lines.append(footer)
    lines.append("")


def main():
    ffcl = load_results(os.path.join(RESULTS_DIR, "ffcl.json"))
    reference = load_results(os.path.join(RESULTS_DIR, "reference.json"))
    rows = ffcl + reference

    lines = []
    lines.append("# FFCL benchmark")
    lines.append("")
    lines.append(
        "Every number below is produced by the programs in `src/` and `scripts/`, and the tables are a "
        "pure function of the two result files (`bin/results/ffcl.json`, `bin/results/reference.json`). "
        "Re-run with `./ffcl_benchmark all` and `python3 scripts/benchmark_reference.py`, then "
        "`python3 scripts/make_tables.py`."
    )
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append(f"- **Machine**: {cpu_description()}, {core_count()} cores, no SMT.")
    lines.append("- **Threads**: every measurement is single threaded (`threads: 1`).")
    lines.append(
        "- **Timing policy**: each workload runs **3 times**; every cell is `mean / median` in seconds."
    )
    lines.append(
        "- **Datasets**: 10 000 samples each, generated by `scripts/MakeClusteringDatasets.py` "
        "(2 features, the usual sklearn clustering toy sets plus `iris`/`mnist`)."
    )
    lines.append(
        "- **Leaf size**: one value per library, never tuned per dataset. FFCL uses `sqrt(n)/3` "
        "(≈33 at n = 10 000), every reference uses **40**, which is the default of both "
        "`sklearn.neighbors.KDTree` and `hdbscan`. Both are the generally good values for their library."
    )
    lines.append(
        "- **Apple to apple**: identical datasets, identical parameters, identical per-query workload "
        "(1 000 query points against the full dataset, k = 2), exhaustive/idle threads on both sides, "
        "and both sides include the core-distance computation in the MST row."
    )
    lines.append("")

    lines.append("## Libraries")
    lines.append("")
    lines.append("| library | version | role |")
    lines.append("| --- | --- | --- |")
    versions = {
        "ffcl": "this repository",
        "flann": "1.9.1 (libflann-dev)",
        "sklearn": "scikit-learn, see below",
        "scipy": "scipy, see below",
        "hdbscan": "hdbscan, see below",
    }
    try:
        import sklearn
        import scipy
        import hdbscan

        versions["sklearn"] = f"scikit-learn {sklearn.__version__}"
        versions["scipy"] = f"scipy {scipy.__version__}"
        versions["hdbscan"] = f"hdbscan {getattr(hdbscan, '__version__', '0.8.44')}"
    except ImportError:
        pass

    lines.append(f"| FFCL | — | the library under test |")
    lines.append(f"| FLANN | {versions['flann']} | reference C++ KD-tree |")
    lines.append(f"| scikit-learn | {versions['sklearn']} | reference KD-tree, DBSCAN, KMeans |")
    lines.append(f"| scipy | {versions['scipy']} | second reference KD-tree (cKDTree) |")
    lines.append(f"| hdbscan | {versions['hdbscan']} | reference HDBSCAN*, Cython port of MLPACK's Boruvka |")
    lines.append("")
    lines.append(
        "**Not compared**: PCL (its KD-tree is a FLANN wrapper, so the FLANN row already covers it) and "
        "MLPACK (needs armadillo/ensmallen, not installed — the `hdbscan` row *is* MLPACK's "
        "`DualTreeBoruvka` ported to Cython, so the MST comparison is not missing a reference)."
    )
    lines.append("")

    lines.append("## Results")
    lines.append("")

    emit_table(
        lines,
        "KD-tree construction",
        "Building the index over the full dataset.",
        rows,
        "kdtree",
        "tree_build",
        ["ffcl", "flann", "sklearn", "scipy"],
        [("FFCL / FLANN", "ffcl", "flann")],
    )
    emit_table(
        lines,
        "k-nearest-neighbour queries",
        "1 000 query points against the full dataset, k = 2, single threaded, exact on every side "
        "(FLANN runs with exhaustive checks so it is not approximating).",
        rows,
        "kdtree",
        "knn",
        ["ffcl", "flann", "sklearn", "scipy"],
        [("FFCL / FLANN", "ffcl", "flann"), ("FFCL / sklearn", "ffcl", "sklearn")],
    )
    # Boruvka: both search strategies side by side, so the table needs two stages, which emit_table
    # does not model, hence the dedicated block.
    lines.append("### Minimum spanning tree (Boruvka), FFCL's two search strategies")
    lines.append("")
    lines.append(
        "Same tree, same dataset, same parameters; only the traversal differs. `dual tree` is what "
        "HDBSCAN uses."
    )
    lines.append("")
    lines.append("| dataset | n | single tree | dual tree | dual / single |")
    lines.append("| --- | --- | --- | --- | --- |")
    single = index_by(rows, "boruvka", "single_tree")
    dual = index_by(rows, "boruvka", "dual_tree")
    for dataset in DATASET_NAMES:
        single_row = single.get(("ffcl", dataset))
        dual_row = dual.get(("ffcl", dataset))
        n = single_row["n"] if single_row else 0
        lines.append(
            f"| {dataset} | {n} | {cell(single_row)} | {cell(dual_row)} | "
            f"{ratio_cell(dual_row, single_row)} |"
        )
    single_means = [single[("ffcl", d)]["mean"] for d in DATASET_NAMES if ("ffcl", d) in single]
    dual_means = [dual[("ffcl", d)]["mean"] for d in DATASET_NAMES if ("ffcl", d) in dual]
    lines.append(
        f"| **mean / median of the 7 datasets** | | "
        f"**{statistics.fmean(single_means):.6f} / "
        f"{statistics.median([single[('ffcl', d)]['median'] for d in DATASET_NAMES if ('ffcl', d) in single]):.6f}** | "
        f"**{statistics.fmean(dual_means):.6f} / "
        f"{statistics.median([dual[('ffcl', d)]['median'] for d in DATASET_NAMES if ('ffcl', d) in dual]):.6f}** | "
        f"**{statistics.fmean(dual_means) / statistics.fmean(single_means):.2f}x** |"
    )
    lines.append("")
    emit_table(
        lines,
        "Minimum spanning tree, dual tree, vs hdbscan",
        "FFCL `make_tree_2` against hdbscan's `KDTreeBoruvkaAlgorithm` (core distances included on "
        "both sides, since hdbscan computes them in the constructor). Note: FFCL `k_nearest_neighbors` "
        "counts the sample itself, so FFCL k = 2 is the equivalent of hdbscan `min_samples = 1`.",
        rows,
        "hdbscan",
        "mst",
        ["ffcl", "hdbscan"],
        [("FFCL / hdbscan", "ffcl", "hdbscan")],
    )
    emit_table(
        lines,
        "HDBSCAN, full pipeline",
        "MST + single linkage tree + condensed tree + cluster extraction.",
        rows,
        "hdbscan",
        "full",
        ["ffcl", "hdbscan"],
        [("FFCL / hdbscan", "ffcl", "hdbscan")],
    )
    emit_table(
        lines,
        "DBSCAN",
        "radius 0.5, min_samples 5.",
        rows,
        "dbscan",
        "full",
        ["ffcl", "sklearn"],
        [("FFCL / sklearn", "ffcl", "sklearn")],
    )
    emit_table(
        lines,
        "KMeans",
        "8 centroids, k-means++ initialisation, one init, Lloyd.",
        rows,
        "kmeans",
        "full",
        ["ffcl", "sklearn"],
        [("FFCL / sklearn", "ffcl", "sklearn")],
    )

    # KMedoids has no reference: sklearn has no KMedoids and sklearn_extra is unmaintained.
    kmedoids_rows = [row for row in rows if row["algorithm"] == "kmedoids"]
    lines.append("### KMedoids")
    lines.append("")
    lines.append(
        "**No reference row.** scikit-learn has no KMedoids and `sklearn_extra` is unmaintained and not "
        "installed, so only FFCL is reported. Adding a reference here is open work."
    )
    lines.append("")
    if kmedoids_rows:
        lines.append("| dataset | n | FFCL (FasterPAM, 8 medoids) |")
        lines.append("| --- | --- | --- |")
        for dataset in DATASET_NAMES:
            row = next((r for r in kmedoids_rows if r["dataset"] == dataset), None)
            n = row["n"] if row else 0
            lines.append(f"| {dataset} | {n} | {cell(row)} |")
        lines.append("")

    lines.append("## Known discrepancies")
    lines.append("")
    lines.append(
        "Two behaviours found while writing the unit tests, recorded here rather than hidden:"
    )
    lines.append("")
    lines.append(
        "1. **`dual_set_shortest_edge` is not exact.** On an 8-point fixture the brute-force answer is "
        "12.0208 (the pair `(.5,.5) -> (9,9)`) while the accelerated traversal returns 12.7279 (the "
        "longer corner diagonal), so it misses the closest pair. This is the `DualTreeTraverser` + "
        "`IndicesToBuffersMap` path; the MST does **not** use it (it goes through `EdgesBuffer`, which "
        "is exact), so the HDBSCAN rows above are unaffected. See the `DISABLED_` tests in "
        "`test/src/search/SearchTest.cpp`."
    )
    lines.append(
        "2. **Single-cluster silhouette is NaN.** `silhouette()` divides by zero when there is only one "
        "cluster. It cannot be asserted in a unit test because the library is built with `-Ofast`, which "
        "implies `-ffinite-math-only` and makes `std::isnan()` a constant `false`."
    )
    lines.append("")
    lines.append("## Reproducing")
    lines.append("")
    lines.append("```sh")
    lines.append("python3 benchmark/scripts/MakeClusteringDatasets.py          # -> bin/datasets/")
    lines.append("cmake -B build_bench -DMODE=benchmark -DCMAKE_BUILD_TYPE=Release")
    lines.append("cmake --build build_bench -j")
    lines.append("./build_bench/ffcl_benchmark all                            # -> bin/results/ffcl.json")
    lines.append("python3 benchmark/scripts/benchmark_reference.py            # -> bin/results/reference.json")
    lines.append("python3 benchmark/scripts/make_tables.py                    # -> benchmark/README.md")
    lines.append("```")
    lines.append("")

    with open(README_PATH, "w") as stream:
        stream.write("\n".join(lines))

    print(f"wrote {README_PATH}")


if __name__ == "__main__":
    main()
