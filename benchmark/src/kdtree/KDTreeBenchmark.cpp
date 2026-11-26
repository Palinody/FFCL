#include "ffcl/benchmark/Dataset.hpp"
#include "ffcl/benchmark/Harness.hpp"
#include "ffcl/benchmark/Indexer.hpp"

#include "ffcl/search/Search.hpp"
#include "ffcl/search/buffer/Unsorted.hpp"

#include <cstddef>
#include <ostream>
#include <vector>

namespace ffcl::benchmark {

// Queries are the first k_n_queries samples, searched against the whole dataset. That is the same
// workload the reference benchmarks run, so the numbers are comparable.
void benchmark_kdtree(std::ostream& results, std::size_t k_n_queries, std::size_t k_nearest_neighbors) {
    for (const auto& name : all_dataset_names()) {
        const auto dataset    = load_dataset(name);
        const auto leaf_size  = ffcl_leaf_size_for(dataset.n_samples);
        const auto n_features = dataset.n_features;
        const auto n_queries  = std::min(k_n_queries, dataset.n_samples);

        auto base = Measurement{};
        base.library   = "ffcl";
        base.algorithm = "kdtree";
        base.dataset   = name;
        base.n         = dataset.n_samples;
        base.leaf      = leaf_size;
        base.threads   = 1;
        base.note = "queries=" + std::to_string(n_queries) + " k=" + std::to_string(k_nearest_neighbors);

        // --- index build -----------------------------------------------------
        {
            auto measurement  = base;
            measurement.stage = "tree_build";

            measure_and_emit(results, measurement, [&] {
                auto indices = make_indices(dataset.n_samples);
                auto samples = dataset.data;
                auto indexer = make_indexer(indices, samples, n_features, leaf_size);
                (void)indexer;
            });
        }
        // --- k nearest neighbours --------------------------------------------
        {
            auto prepared_indices = make_indices(dataset.n_samples);
            auto prepared_samples = dataset.data;
            auto prepared_indexer = make_indexer(prepared_indices, prepared_samples, n_features, leaf_size);

            auto measurement  = base;
            measurement.stage = "knn";

            measure_and_emit(results, measurement, [&] {
                auto searcher = search::Searcher<IndexerType>(prepared_indexer);

                auto results_buffer = std::vector<float>{};
                results_buffer.reserve(n_queries * k_nearest_neighbors);

                for (std::size_t query_index = 0; query_index < n_queries; ++query_index) {
                    auto buffer = searcher(search::buffer::Unsorted<decltype(prepared_samples.begin())>(
                        searcher.features_range_first(query_index),
                        searcher.features_range_last(query_index),
                        k_nearest_neighbors));

                    // keep the traversal from being optimised away
                    results_buffer.emplace_back(buffer.furthest_distance());
                }
                (void)results_buffer;
            });
        }
    }
}

}  // namespace ffcl::benchmark
