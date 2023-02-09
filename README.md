# cpp_clustering
Fast iterator compatible clustering algorithms written in c++.

- [Current features](#current-features)
- [KMedoids algorithms performance](#kmedoids-algorithms-performance)
- [Installation](#installation)
- [Compiling](#compiling)
- [How to use](#how-to-use)

## TODO (in order)

- `cpp_clustering::containers::KDTree`
- Proper unit testing
- DBSCAN
- OPTICS
- DENCLUE

## Current features

- ### KMedoids:
  - FasterMSC (with(out) pairwise distance matrix) [paper](https://arxiv.org/pdf/2209.12553.pdf)
  - FasterPAM (with(out) pairwise distance matrix) [paper](https://arxiv.org/pdf/2008.05171.pdf)

The implementation of the papers was helped by one of the [author's repo](https://github.com/kno10/rust-kmedoids)

- ### KMeans:
  - Lloyd

- ### Distance functions
  - euclidean
  - manhattan
  - cosine similarity

- ### initialization:
  - random uniform
  - spatial uniform
  - pam::build
  - kmeans++

- ### Selecting the number of clusters
  - silhouette method

## KMedoids algorithms performance
- Cpu: `Intel® Core™ i5-9600KF CPU @ 3.70GHz × 6`
- Dataset: MNIST
- initialization: random uniform

The following table summarizes the results of single runs made with different parameters. It provides a rough estimate about what one could expect from the present library. For a more in depth analysis of the performance of the algorithms, refer to the articles [(1)](https://arxiv.org/pdf/2209.12553.pdf) and [(2)](https://arxiv.org/pdf/2008.05171.pdf)

|                         | **type**   |**n_samples**|**n_features**|**n_centroids**|**n_iter (converged)**| **time (s)**|**n_threads**|
------                    | -----      | -----       |---           |------         |---                           |---          |---          |
**FasterPAM**             | float       | 15000     |784| 10|2| 1.7|1|
**FasterPAM**             | float       | 15000     |784|100|4| 4.5|1|
**FasterPAM**             | float       | 30000     |784| 10|3|11.1|1|
**FasterPAM**             | float       | 30000     |784|100|4|  17|1|
**FasterMSC**             | float       | 15000     |784| 10|2| 3.3|1|
**FasterMSC**             | float       | 15000     |784|100|2| 6.4|1|
**FasterMSC**             | float       | 30000     |784| 10|2|12.8|1|
**FasterMSC**             | float       | 30000     |784|100|3|27.1|1|
**PairwiseDistanceMatrix**| float       |   15000    |784|  | |  40|1|
**PairwiseDistanceMatrix**| float       |   30000    |784|  | | 162|1|
**PairwiseDistanceMatrix**| float       |   15000    |784|  | |6.27|6|
**PairwiseDistanceMatrix**| float       |   30000    |784|  | |25.6|6|

## Installation
```sh
cd <your_repo>
git clone <this_repo>
mkdir build
cd build
```

## Compiling
### 1) As a library
```sh
cmake ..
build
```
This will create a `libcpp_clustering.a` file in the `build/` firectory

### 2) To run the benchmarks

Generate the datasets automatically:
```
python3 ../datasets/datasets_maker.py
```
This will create the following files in `../datasets/clustering/`
```sh
.
├── inputs
│   ├── aniso.txt
│   ├── blobs.txt
│   ├── iris.txt
│   ├── mnist.txt
│   ├── noisy_circles.txt
│   ├── noisy_moons.txt
│   ├── no_structure.txt
│   ├── unbalanced_blobs.txt
│   └── varied.txt
└── targets
    ├── aniso.txt
    ├── blobs.txt
    ├── iris.txt
    ├── mnist.txt
    ├── noisy_circles.txt
    ├── noisy_moons.txt
    ├── no_structure.txt
    ├── unbalanced_blobs.txt
    └── varied.txt
```

The average default number of samples for the datasets is ~2000 samples. This can be changed in the `datasets_maker.py` script. MNIST can have up to 70000 samples and 784 features. Setting `n_samples_mnist=None` results in loading mnist with the maximum number of samples.

Compile (this can also be done before generating the datasets except running the benchmarks)

```sh
cmake .. -DMODE="benchmark"
build
```
This will create a `cpp_clustering` executable in the `build` folder. To run the benchmarks, simply:
```
./cpp_clustering
```
The results will be written in `../datasets/clustering/` and the folder structure might now look like this:
```sh
├── centroids
│   ├── aniso.txt
<...>
│   └── varied.txt
├── inputs
│   ├── aniso.txt
<...>
│   └── varied.txt
├── predictions
│   ├── aniso.txt
<...>
│   └── varied.txt
└── targets
    ├── aniso.txt
<...>
    └── varied.txt
```
The clusters can be visualised:
```sh
python3 ../datasets/plot.py
```

### 3) To run the unit tests
```sh
cmake .. -DMODE="gtest"
build
```
Run the gtests:
```sh
ctest -V
```

### \*) All cmake options (the ones with a (\*) can be accumulated)
- `-DMODE="gtest"`
- `-DMODE="benchmark"`
- `-DVERBOSE=true` (\*)
- `-DTHREADS_ENABLED=true` (\*)

## How to use

### Silhouette method

#### Example: select the best number of centroids/medoids k in a range
```c
std::size_t k_min = 2;
std::size_t k_max = 10;

std::vector<float> scores(k_max - k_min);

// range n_centroids/n_medoids in [2, 10[
for (size_t k = k_min; k < k_max; ++k) {
    // use any clustering algorithm that better suits your use case
    common::clustering::KMeans<dType> kmeans(k, n_features);
    // fit the centroids (or medoids if it was KMedoids)
    kmeans.fit(data.begin(), data.end());
    // map the samples to their closest centroid/medoid
    const auto predictions = kmeans.predict(data.begin(), data.end());
    // compute the silhouette scores for each sample
    const auto samples_silhouette_values = common::clustering::silhouette_method::silhouette(data.begin(),
                                                                                             data.end(),
                                                                                             predictions.begin(),
                                                                                             predictions.end(),
                                                                                             /*n_features=*/n_features);
    // get the average score
    const auto mean_silhouette_coefficient = common::clustering::silhouette_method::get_mean_silhouette_coefficient(
        samples_silhouette_values.begin(), samples_silhouette_values.end());
    // accumulate the current scores
    scores[k - k_min] = mean_silhouette_coefficient;
}
// find the k corresponding to the number of centroids/medoids k with the best average silhouette score
const auto best_k = k_min + common::utils::argmax(scores.begin(), scores.end())
```

### KMeans

#### Example with kmeans++ initialization
```c
using KMeans = cpp_clustering::KMeans<float>;

const std::size_t n_features = 3;
// input_data.size() / n_features -> n_samples (4 in this case)
std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
// must be less than n_samples
const std::size_t n_centroids = 2;

cont auto centroids_init = common::clustering::kmeansplusplus::make_centroids(input_data.begin(), input_data.end(), n_centroids, n_features))
// initializing the centroids manually is optional
auto kmeans = KMeans(n_centroids, n_features, centroids_init);

const auto centroids = kmeans.fit(input_data.begin(), input_data.end());
```

### KMedoids

#### Simple example
```c
using KMedoids = cpp_clustering::KMedoids<float>;

const std::size_t n_features = 3;
// input_data.size() / n_features -> n_samples (4 in this case)
std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
// must be less than n_samples
const std::size_t n_medoids = 2;

auto kmedoids = KMedoids(n_medoids, n_features);
// swap medoids and returns the centroids
const auto centroids = kmedoids.fit(input_data.begin(), input_data.end());
```

#### Complete example
```c
#include "cpp_clustering/kmedoids/KMedoids.hpp"
// dType: type of the training samples
// true: PrecomputePairwiseDistanceMatrix (set to true by default)
using KMedoids = cpp_clustering::KMedoids<SomeDataType, true>;

const std::size_t n_features = 3;
// input_data.size() / n_features -> n_samples (4 in this case)
std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
// each elements of the labels vector maps to a sample from input_data
std::vector<std::size_t> labels = {0, 0, 2, 1};
// 3 for class 0, 1 and 2 from the labels vector
const auto n_classes = 3;

// must be less than n_samples
const std::size_t n_medoids = 2;

auto kmedoids = KMedoids(n_medoids, n_features);
// set the options. The ones presented in this example are the same as the ones by default and provide no change
kmedoids.set_options(
            /*KMedoids options=*/KMedoids::Options()
                .max_iter(100)
                .early_stopping(true)
                .patience(0)
                .n_init(1));

// Fit the data (inputs_first, inputs_last) with a PAM algorithm. Default: cpp_clustering::FasterPAM
const auto centroids = kmedoids.fit<cpp_clustering::FasterMSC>(input_data.begin(), input_data.end());

// map each data sample to its medoid
const std::vector<std::size_t> predictions = kmedoids.predict(input_data.begin(), input_data.end());

// swap the medoids if you want them to match your labels.
// Complexity (worst case): O(n_centroids!).
const auto [best_match_count, swapped_centroids] =
            kmedoids.swap_to_best_count_match(input_data.begin(), input_data.end(), labels.begin(), labels.end());

// another way to swap the medoids but much faster for larger n_centroids
// Complexity (worst case): O(n_medoids * n_classes).
const auto [best_match_count, swapped_centroids] =
            kmedoids.remap_centroid_to_label_index(input_data.begin(), input_data.end(), labels.begin(), labels.end(), n_classes);

```
