# FFCL reference-library benchmark

FFCL against every library cloned under `benchmark_references/`, on 2-, 3- and
4-dimensional data, for KD-tree build, kNN, radius, DBSCAN, mutual-reachability MST,
dendrogram and HDBSCAN. The tables are a pure function of the JSON the runners write;
nothing here is hand-edited. See [Reproducing](#reproducing) for the commands.

## Context

- **CPU** Intel(R) Core(TM) i5-9600KF CPU @ 3.70GHz, 6 cores, no SMT, governor `performance`.
- **Compiler** g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0, `-O3 -march=native`, OpenMP on.
- **Datasets** the 7 generated sets of `MakeClusteringDatasets.py`, 10 000 samples each,
  at 2, 3 and 4 features. Same seed and generators, only the dimension changes.
- **Leaf size** 40 for every tree that has one (sklearn's and hdbscan's default). mlpack's
  DBSCAN and EMST build their internal tree at mlpack's own default of 20, which the public
  constructors do not expose; that is noted where it applies.
- **Threads** 1 and 6, pinned with `taskset`, `OMP_NUM_THREADS` / `PARLAY_NUM_THREADS` set
  per row. nanoflann, FLANN, pico_tree and mlpack are single-threaded by construction.
- **Timing** 3 tries, geometric mean over the 7 datasets, in ms. Index build is included in
  every clustering row and excluded from the standalone kNN/radius rows.
- **Conventions** kNN 1 000 queries k = 2 with k counting the sample itself; DBSCAN radius
  0.5 with min_pts = 5 including the sample; MST over mutual reachability with k = 2;
  HDBSCAN `min_cluster_size` = 20, `min_samples` = 1 (k = 2).
- **FFCL builds.** `ffcl-ser` is `THREADS_ENABLED=OFF`, the CMake default; `ffcl-par` is
  `THREADS_ENABLED=true`, the build the previously published tables used. They run the same
  algorithm and produce the same MST weight; only the OpenMP loops differ.
- **FFCL DBSCAN** is measured at `min_samples` = 4 here. FFCL's core test is strictly
  `size() > min_samples` and its buffer includes the query sample, so 4 is what matches the
  references' 5. The previously published tables used 5 and are one stricter.
- † marks a row that is not a like-for-like comparison: ArborX stops at the HDBSCAN
  dendrogram and does not extract flat clusters; mlpack's EMST is a plain Euclidean MST,
  not a mutual-reachability one.

## 2-D, 1 thread

| workload | ffcl-ser | ffcl-par | nano | flann | pico | arborx | mlpack | pdbscan | pargeo | sklearn | scipy | hdbscan | winner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| build | 1.04 | 1.12 | 1.34 | 0.76 | 0.52 | 0.68 | 1.21 | — | — | 1.95 | 1.49 | — | **pico** (1.30x over next) |
| kNN | 0.50 | 0.51 | 0.35 | 0.38 | 0.31 | 0.35 | 3.48 | — | — | 1.24 | 0.74 | — | **pico** (1.15x over next) |
| radius | — | — | — | — | 0.79 | 1.80 | — | — | — | — | — | — | **pico** (2.30x over next) |
| MST | 26.4 | 59.3 | — | — | — | 17.9 | 30.9 † | — | 21.3 | — | — | 129 | **arborx** (1.19x over next) |
| dendrogram | — | — | — | — | — | 0.49 | — | — | 0.65 | — | — | — | **arborx** (1.33x over next) |
| HDBSCAN | 27.9 | 56.8 | — | — | — | 19.3 † | — | — | — | — | — | 127 | **ffcl-ser** (4.57x over next) |
| DBSCAN | 3.17 | 3.42 | — | — | — | 9.16 | 24.7 | 7.17 | — | 39.3 | — | — | **ffcl-ser** (2.26x over next) |

## 3-D, 1 thread

| workload | ffcl-ser | ffcl-par | nano | flann | pico | arborx | mlpack | pdbscan | pargeo | sklearn | scipy | hdbscan | winner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| build | 1.11 | 1.14 | 1.45 | 0.86 | 0.58 | 0.76 | 1.33 | — | — | 2.34 | 1.54 | — | **pico** (1.29x over next) |
| kNN | 0.63 | 0.64 | 0.49 | 0.49 | 0.41 | 0.49 | 4.22 | — | — | 1.62 | 0.87 | — | **pico** (1.18x over next) |
| radius | — | — | — | — | 0.68 | 1.66 | — | — | — | — | — | — | **pico** (2.43x over next) |
| MST | 39.7 | 307 | — | — | — | 28.0 | 51.0 † | — | 33.3 | — | — | 189 | **arborx** (1.19x over next) |
| dendrogram | — | — | — | — | — | 0.51 | — | — | 0.67 | — | — | — | **arborx** (1.31x over next) |
| HDBSCAN | 41.4 | 299 | — | — | — | 29.4 † | — | — | — | — | — | 177 | **ffcl-ser** (4.27x over next) |
| DBSCAN | 17.5 | 17.7 | — | — | — | 13.6 | 20.3 | 12.0 | — | 41.5 | — | — | **pdbscan** (1.13x over next) |

## 4-D, 1 thread

| workload | ffcl-ser | ffcl-par | nano | flann | pico | arborx | mlpack | pdbscan | pargeo | sklearn | scipy | hdbscan | winner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| build | 1.14 | 1.20 | 1.54 | 0.97 | 0.61 | 0.83 | 1.35 | — | — | 2.74 | 1.59 | — | **pico** (1.37x over next) |
| kNN | 0.80 | 0.82 | 0.68 | 0.63 | 0.54 | 0.72 | 6.04 | — | — | 2.18 | 1.09 | — | **pico** (1.16x over next) |
| radius | — | — | — | — | 0.53 | 1.34 | — | — | — | — | — | — | **pico** (2.51x over next) |
| MST | 51.8 | 377 | — | — | — | 41.6 | 75.6 † | — | 50.5 | — | — | 234 | **arborx** (1.21x over next) |
| dendrogram | — | — | — | — | — | 0.48 | — | — | 0.67 | — | — | — | **arborx** (1.41x over next) |
| HDBSCAN | 54.5 | 377 | — | — | — | 42.7 † | — | — | — | — | — | 227 | **ffcl-ser** (4.17x over next) |
| DBSCAN | 6.56 | 6.53 | — | — | — | 11.8 | 16.1 | 11.1 | — | 41.5 | — | — | **ffcl-par** (1.71x over next) |

## 6 threads, parallel-capable libraries only

### 2-D

| workload | ffcl-ser | ffcl-par | arborx | pdbscan | pargeo | winner |
|---|---|---|---|---|---|---|
| build | 1.05 | 1.10 | 0.29 | — | — | **arborx** (3.59x over next) |
| kNN | 0.51 | 0.51 | 0.14 | — | — | **arborx** (3.68x over next) |
| radius | — | — | 0.47 | — | — | **arborx** |
| MST | 26.2 | 49.8 | 4.11 | — | 7.21 | **arborx** (1.75x over next) |
| dendrogram | — | — | 0.47 | — | 0.37 | **pargeo** (1.29x over next) |
| HDBSCAN | 28.1 | 53.4 | 5.32 † | — | — | **ffcl-ser** |
| DBSCAN | 3.13 | 3.44 | 2.54 | 2.45 | — | **pdbscan** (1.03x over next) |

### 3-D

| workload | ffcl-ser | ffcl-par | arborx | pdbscan | pargeo | winner |
|---|---|---|---|---|---|---|
| build | 1.11 | 1.13 | 0.32 | — | — | **arborx** (3.41x over next) |
| kNN | 0.63 | 0.65 | 0.16 | — | — | **arborx** (3.98x over next) |
| radius | — | — | 0.42 | — | — | **arborx** |
| MST | 39.7 | 63.5 | 6.01 | — | 11.5 | **arborx** (1.91x over next) |
| dendrogram | — | — | 0.51 | — | 0.39 | **pargeo** (1.31x over next) |
| HDBSCAN | 41.2 | 63.8 | 7.36 † | — | — | **ffcl-ser** |
| DBSCAN | 17.5 | 17.5 | 3.74 | 3.66 | — | **pdbscan** (1.02x over next) |

### 4-D

| workload | ffcl-ser | ffcl-par | arborx | pdbscan | pargeo | winner |
|---|---|---|---|---|---|---|
| build | 1.13 | 1.19 | 0.34 | — | — | **arborx** (3.30x over next) |
| kNN | 0.79 | 0.81 | 0.20 | — | — | **arborx** (3.90x over next) |
| radius | — | — | 0.34 | — | — | **arborx** |
| MST | 51.4 | 75.1 | 8.82 | — | 15.2 | **arborx** (1.73x over next) |
| dendrogram | — | — | 0.53 | — | 0.39 | **pargeo** (1.34x over next) |
| HDBSCAN | 53.5 | 72.7 | 10.4 † | — | — | **ffcl-ser** |
| DBSCAN | 6.46 | 6.54 | 3.31 | 3.35 | — | **arborx** (1.01x over next) |

## By workload, 1 thread

The same numbers as above, regrouped so every library that implements a workload is on
one row. `—` means the library has no implementation for that workload.

### build

| library | 2-D | 3-D | 4-D |
|---|---|---|---|
| ffcl-ser | 1.04 | 1.11 | 1.14 |
| ffcl-par | 1.12 | 1.14 | 1.20 |
| nano | 1.34 | 1.45 | 1.54 |
| flann | 0.76 | 0.86 | 0.97 |
| pico | 0.52 | 0.58 | 0.61 |
| arborx | 0.68 | 0.76 | 0.83 |
| mlpack | 1.21 | 1.33 | 1.35 |
| sklearn | 1.95 | 2.34 | 2.74 |
| scipy | 1.49 | 1.54 | 1.59 |

### kNN

| library | 2-D | 3-D | 4-D |
|---|---|---|---|
| ffcl-ser | 0.50 | 0.63 | 0.80 |
| ffcl-par | 0.51 | 0.64 | 0.82 |
| nano | 0.35 | 0.49 | 0.68 |
| flann | 0.38 | 0.49 | 0.63 |
| pico | 0.31 | 0.41 | 0.54 |
| arborx | 0.35 | 0.49 | 0.72 |
| mlpack | 3.48 | 4.22 | 6.04 |
| sklearn | 1.24 | 1.62 | 2.18 |
| scipy | 0.74 | 0.87 | 1.09 |

### radius

| library | 2-D | 3-D | 4-D |
|---|---|---|---|
| pico | 0.79 | 0.68 | 0.53 |
| arborx | 1.80 | 1.66 | 1.34 |

### MST

| library | 2-D | 3-D | 4-D |
|---|---|---|---|
| ffcl-ser | 26.4 | 39.7 | 51.8 |
| ffcl-par | 59.3 | 307 | 377 |
| arborx | 17.9 | 28.0 | 41.6 |
| mlpack | 30.9 † | 51.0 † | 75.6 † |
| pargeo | 21.3 | 33.3 | 50.5 |
| hdbscan | 129 | 189 | 234 |

### dendrogram

| library | 2-D | 3-D | 4-D |
|---|---|---|---|
| arborx | 0.49 | 0.51 | 0.48 |
| pargeo | 0.65 | 0.67 | 0.67 |

### HDBSCAN

| library | 2-D | 3-D | 4-D |
|---|---|---|---|
| ffcl-ser | 27.9 | 41.4 | 54.5 |
| ffcl-par | 56.8 | 299 | 377 |
| arborx | 19.3 † | 29.4 † | 42.7 † |
| hdbscan | 127 | 177 | 227 |

### DBSCAN

| library | 2-D | 3-D | 4-D |
|---|---|---|---|
| ffcl-ser | 3.17 | 17.5 | 6.56 |
| ffcl-par | 3.42 | 17.7 | 6.53 |
| arborx | 9.16 | 13.6 | 11.8 |
| mlpack | 24.7 | 20.3 | 16.1 |
| pdbscan | 7.17 | 12.0 | 11.1 |
| sklearn | 39.3 | 41.5 | 41.5 |

## By workload, 6 threads

### build

| library | 2-D | 3-D | 4-D |
|---|---|---|---|
| ffcl-ser | 1.05 | 1.11 | 1.13 |
| ffcl-par | 1.10 | 1.13 | 1.19 |
| arborx | 0.29 | 0.32 | 0.34 |

### kNN

| library | 2-D | 3-D | 4-D |
|---|---|---|---|
| ffcl-ser | 0.51 | 0.63 | 0.79 |
| ffcl-par | 0.51 | 0.65 | 0.81 |
| arborx | 0.14 | 0.16 | 0.20 |

### radius

| library | 2-D | 3-D | 4-D |
|---|---|---|---|
| arborx | 0.47 | 0.42 | 0.34 |

### MST

| library | 2-D | 3-D | 4-D |
|---|---|---|---|
| ffcl-ser | 26.2 | 39.7 | 51.4 |
| ffcl-par | 49.8 | 63.5 | 75.1 |
| arborx | 4.11 | 6.01 | 8.82 |
| pargeo | 7.21 | 11.5 | 15.2 |

### dendrogram

| library | 2-D | 3-D | 4-D |
|---|---|---|---|
| arborx | 0.47 | 0.51 | 0.53 |
| pargeo | 0.37 | 0.39 | 0.39 |

### HDBSCAN

| library | 2-D | 3-D | 4-D |
|---|---|---|---|
| ffcl-ser | 28.1 | 41.2 | 53.5 |
| ffcl-par | 53.4 | 63.8 | 72.7 |
| arborx | 5.32 † | 7.36 † | 10.4 † |

### DBSCAN

| library | 2-D | 3-D | 4-D |
|---|---|---|---|
| ffcl-ser | 3.13 | 17.5 | 6.46 |
| ffcl-par | 3.44 | 17.5 | 6.54 |
| arborx | 2.54 | 3.74 | 3.31 |
| pdbscan | 2.45 | 3.66 | 3.35 |

## Findings

**The parallel build is a pessimisation on this workload.** `THREADS_ENABLED=true` turns
on the parallel Boruvka path, and it is slower than the serial dual-tree path at every
dimension and every thread count:

| workload | 2-D ser | 2-D par | 3-D ser | 3-D par | 4-D ser | 4-D par |
|---|---|---|---|---|---|---|
| MST dual, 1 thread | 26.4 | 59.3 | 39.7 | 307 | 51.8 | 377 |
| MST dual, 6 threads | 26.2 | 49.8 | 39.7 | 63.5 | 51.4 | 75.1 |
| HDBSCAN full, 1 thread | 27.9 | 56.8 | 41.4 | 299 | 54.5 | 377 |
| HDBSCAN full, 6 threads | 28.1 | 53.4 | 41.2 | 63.8 | 53.5 | 72.7 |
| DBSCAN, 1 thread | 3.17 | 3.42 | 17.5 | 17.7 | 6.56 | 6.53 |
| DBSCAN, 6 threads | 3.13 | 3.44 | 17.5 | 17.5 | 6.46 | 6.54 |

At 1 thread the parallel MST is
2.2x slower at 2-D, 7.7x slower at 3-D, 7.3x slower at 4-D
than the serial one; at 6 threads it is still
1.90x slower, 1.60x slower, 1.46x slower.
The serial build does not use the extra cores, but it does not need them: it beats the
parallel build at 6 threads while running on one. This is the real dual-tree scaling
problem, and it is in the parallel path, not the traversal. The published tables used the
parallel build, so their FFCL MST and HDBSCAN rows are pessimistic by that factor.

**FFCL is not the leader against the specialised C++ trees, but it is close on MST.**
pico_tree wins the build and the kNN query at every dimension; ArborX wins the MST and the
dendrogram. FFCL's own wins are full HDBSCAN against the Python `hdbscan` at every
dimension, and DBSCAN at 2-D and 4-D.

**kNN.** FFCL (serial) is behind nanoflann by
1.42x at 2-D, 1.29x at 3-D, 1.19x at 4-D and behind pico_tree by
1.64x at 2-D, 1.53x at 3-D, 1.48x at 4-D. It still beats sklearn by
2.48x at 2-D, 2.56x at 3-D, 2.71x at 4-D and scipy by
1.47x at 2-D, 1.38x at 3-D, 1.35x at 4-D. It is 5th of the 8 kNN implementations, also behind FLANN and ArborX.

**Build.** FFCL builds in
1.04 ms at 2-D, 1.11 ms at 3-D, 1.14 ms at 4-D, against
0.52, 0.58, 0.61 ms for pico_tree and
0.68, 0.76, 0.83 ms for ArborX. It is 4th of 8,
behind pico_tree, ArborX and FLANN.

**MST.** With the serial build FFCL takes
26 ms at 2-D, 40 ms at 3-D, 52 ms at 4-D,
against ArborX's
18, 28, 42 ms and pargeo's
21, 33, 50 ms. FFCL is
1.47x, 1.42x, 1.25x
behind ArborX and essentially tied with pargeo, while being
4.9x, 4.8x, 4.5x
faster than the Python `hdbscan`. The earlier claim that FFCL is 1.7x slower than `hdbscan`
at 3-D and 4-D came from the parallel build and does not hold for the default one.

**HDBSCAN.** FFCL (serial) is
4.57x faster at 2-D, 4.27x faster at 3-D, 4.17x faster at 4-D
than the Python `hdbscan` library. ArborX's HDBSCAN row is the dendrogram only, so its
lead is not like-for-like.

**DBSCAN.** With the conventions matched, FFCL wins 2-D (
3.17 ms vs pdbscan 7.17)
and 4-D, but loses 3-D to pdbscan (
17.5 ms vs 12.0). 3-D is
FFCL's weak dimension for DBSCAN, as it was in the published tables. DBSCAN does not use
the extra threads in either build.

**Leaf 16 is only a trap for the parallel build.** Dual-tree MST, leaf 40 against
leaf 16, 1 thread:

| build | 2-D l40 | 2-D l16 | 3-D l40 | 3-D l16 | 4-D l40 | 4-D l16 |
|---|---|---|---|---|---|---|
| serial | 26.4 | 23.7 | 39.7 | 39.0 | 51.8 | 58.5 |
| parallel | 59.3 | 1303 | 307 | 1848 | 377 | 2330 |

With the serial build leaf 16 is flat for the MST (
0.90x at 2-D, 0.98x at 3-D, 1.13x at 4-D
). With the parallel build it is
22x worse at 2-D, 6x worse at 3-D, 6x worse at 4-D
— the 22x 2-D regression from the performance work. So the leaf-16 blowup is another
symptom of the parallel path, not of the traversal itself. Leaf 16 still helps kNN
slightly and hurts the build by 1.5x.

**DBSCAN convention.** The published rows used `min_samples` = 5, one stricter than the
references' `min_pts` = 5. Re-measured at 4, FFCL's DBSCAN time changes by
-2% at 2-D, -1% at 3-D, -5% at 4-D
— within run-to-run noise. The effect is small and does not change any ranking, but the
published rows were not quite like-for-like and are now corrected.

## Reproducing

```sh
cd benchmark/references && cmake -B build . && cmake --build build -j6
./benchmark/references/run_references.sh benchmark/references/build /tmp/perf/refs2

# FFCL serial (default) and parallel builds, leaf 40 and 16
./benchmark/scripts/run_matrix.sh ./build_bench /tmp/perf/matrix_new
./benchmark/scripts/run_matrix.sh ./build_par   /tmp/perf/matrix

# FFCL DBSCAN at min_samples=4, both builds
./benchmark/references/run_ffcl_dbscan4.sh ./build_bench /tmp/perf/matrix4
./benchmark/references/run_ffcl_dbscan4.sh ./build_par   /tmp/perf/matrix4_par

python3 benchmark/references/aggregate_references.py   # writes benchmark/references/README.md
```

The Python reference rows need numpy, sklearn, scipy and hdbscan. The bundled venv at
`benchmark_references/hdbscan_bench_venv` was created for python3.14 and does not run on
this machine's python3.10, so the existing `reference_<d>d.json` files were reused.

## Caveats and what did not work

- **mlpack's MST is a plain Euclidean MST**, not a mutual-reachability one. It is still
  slower than FFCL's MR-MST (
31 vs 26 ms at 2-D, 51 vs 40 ms at 3-D, 76 vs 52 ms at 4-D
) despite doing less work, so it is not the interesting comparison it looks like.

- **ArborX's HDBSCAN stops at the dendrogram.** It has no flat-cluster extraction, so its
  HDBSCAN row is the dendrogram construction only and is marked †.

- **pico_tree and pargeo have no DBSCAN.** pico_tree's radius row is the DBSCAN primitive,
  not a full DBSCAN; pargeo only does MST and dendrogram.

- **mlpack's DBSCAN and EMST use leaf 20**, mlpack's own default. The public constructors do
  not expose the leaf size, so those two rows are not at the reference leaf of 40.

- **One executable per library.** Linking mlpack/armadillo into the same binary as
  nanoflann slowed the nanoflann 4-D kNN loop from 0.68 to 0.86 ms (26%), which would have
  biased the table. Each library now builds and runs on its own.

- **pdbscan and pargeo cannot share a binary.** Both vendor a different parlay scheduler;
  linking both makes the scheduler singleton resolve to the wrong layout and segfaults.

- **FLANN must use `KDTreeSingleIndexParams(leaf)`**, not `KDTreeIndexParams(leaf)`. The
  latter's argument is the number of randomised trees, which is why FFCL's own published
  FLANN build row is ~80 ms instead of ~0.8 ms. `refbench_flann` uses the single-tree params.

- **`perf` is unusable** (`perf_event_paranoid` = 4), so everything here is wall-clock by
  ablation, not a profile.

