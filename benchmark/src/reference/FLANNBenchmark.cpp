#include "ffcl/benchmark/Dataset.hpp"
#include "ffcl/benchmark/Harness.hpp"

#include <cstddef>
#include <ostream>

#if defined(FFCL_BENCH_HAS_FLANN)
#include <flann/flann.hpp>

#include <vector>
#endif

namespace ffcl::benchmark {

// FLANN is the reference KD-tree implementation for the neighbour searches. Same workload as the FFCL
// side: the first `n_queries` samples are queried against the whole dataset.
//
// `kReferenceLeafSize` is FLANN's own default (40), which is also the generally good leaf size we use
// for the Python references. It is not tuned per dataset.
void benchmark_flann(std::ostream& results, std::size_t n_queries, std::size_t k_nearest_neighbors) {
#if !defined(FFCL_BENCH_HAS_FLANN)
    (void)results;
    (void)n_queries;
    (void)k_nearest_neighbors;

    auto measurement  = Measurement{};
    measurement.library   = "flann";
    measurement.algorithm = "kdtree";
    measurement.stage     = "unavailable";
    measurement.dataset   = "-";
    measurement.note      = "FLANN was not found at configure time";
    emit_json_line(results, measurement);
#else
    for (const auto& name : all_dataset_names()) {
        const auto dataset   = load_dataset(name);
        const auto n_samples = dataset.n_samples;
        const auto queries   = std::min(n_queries, n_samples);

        auto base = Measurement{};
        base.library   = "flann";
        base.algorithm = "kdtree";
        base.dataset   = name;
        base.n         = n_samples;
        base.leaf      = kReferenceLeafSize;
        base.threads   = 1;
        base.note      = "queries=" + std::to_string(queries) + " k=" + std::to_string(k_nearest_neighbors) +
                    " index=kdtree checks=exhaustive";

        // flann::Matrix does not own its data, and the index keeps a reference to it for its whole
        // lifetime, so the storage has to outlive the index and must not be const
        auto storage = dataset.data;

        auto matrix       = flann::Matrix<float>(storage.data(), n_samples, dataset.n_features);
        auto query_matrix = flann::Matrix<float>(storage.data(), queries, dataset.n_features);

        auto index = flann::Index<flann::L2<float>>(matrix, flann::KDTreeIndexParams(kReferenceLeafSize));

        // FLANN's kNN search takes Matrix<int> for the indices and a SearchParams; one query row at a
        // time, which is the same per-query workload the FFCL side runs.
        auto result_indices = std::vector<std::vector<int>>(queries, std::vector<int>(k_nearest_neighbors));
        auto result_dists   = std::vector<std::vector<float>>(queries, std::vector<float>(k_nearest_neighbors));

        // --- index build -----------------------------------------------------
        {
            auto measurement  = base;
            measurement.stage = "tree_build";

            measure_and_emit(results, measurement, [&] { index.buildIndex(); });
        }
        // --- k nearest neighbours --------------------------------------------
        {
            auto measurement  = base;
            measurement.stage = "knn";

            measure_and_emit(results, measurement, [&] {
                for (std::size_t query_index = 0; query_index < queries; ++query_index) {
                    const auto single_query = flann::Matrix<float>(
                        query_matrix.ptr() + query_index * dataset.n_features, 1, dataset.n_features);

                    auto neighbor_indices = flann::Matrix<int>(result_indices[query_index].data(), 1, k_nearest_neighbors);
                    auto neighbor_dists   = flann::Matrix<float>(result_dists[query_index].data(), 1, k_nearest_neighbors);

                    // -1 checks = exhaustive, so FLANN returns the exact answer like FFCL does.
                    // Any smaller value makes FLANN approximate and the comparison meaningless.
                    index.knnSearch(single_query,
                                    neighbor_indices,
                                    neighbor_dists,
                                    k_nearest_neighbors,
                                    flann::SearchParams(-1));
                }
            });
        }
    }
#endif
}

}  // namespace ffcl::benchmark
