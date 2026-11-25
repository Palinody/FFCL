#include "ffcl/benchmark/Dataset.hpp"
#include "ffcl/benchmark/Harness.hpp"
#include "ffcl/benchmark/Indexer.hpp"

#include "ffcl/datastruct/graph/spanning_tree/BoruvkasAlgorithm.hpp"

#include <cmath>
#include <cstddef>
#include <ostream>

namespace ffcl::benchmark {

// Boruvka on its own, for both the single-tree and the dual-tree step, so the two search strategies
// can be compared without the HDBSCAN stages around them.
void benchmark_minimum_spanning_tree(std::ostream& results, std::size_t k_nearest_neighbors) {
    for (const auto& name : all_dataset_names()) {
        const auto dataset    = load_dataset(name);
        const auto leaf_size  = ffcl_leaf_size_for(dataset.n_samples);
        const auto n_features = dataset.n_features;

        for (const auto use_dual_tree : {false, true}) {
            auto measurement  = Measurement{};
            measurement.library   = "ffcl";
            measurement.algorithm = "boruvka";
            measurement.stage     = use_dual_tree ? "dual_tree" : "single_tree";
            measurement.dataset   = name;
            measurement.n         = dataset.n_samples;
            measurement.leaf      = leaf_size;
            measurement.threads   = 1;
            measurement.note      = "k_nearest_neighbors=" + std::to_string(k_nearest_neighbors);

            measure_and_emit(results, measurement, [&] {
                auto indices = make_indices(dataset.n_samples);
                auto samples = dataset.data;
                auto indexer = make_indexer(indices, samples, n_features, leaf_size);

                auto boruvkas_algorithm = BoruvkasAlgorithm<IndexerType>();
                boruvkas_algorithm.set_options(
                    typename BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(k_nearest_neighbors));

                const auto mst = use_dual_tree ? boruvkas_algorithm.make_tree_2(std::move(indexer))
                                               : boruvkas_algorithm.make_tree(std::move(indexer));
                (void)mst;
            });
        }
    }
}

}  // namespace ffcl::benchmark
