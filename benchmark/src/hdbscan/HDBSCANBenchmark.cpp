#include "ffcl/benchmark/Dataset.hpp"
#include "ffcl/benchmark/Harness.hpp"
#include "ffcl/benchmark/Indexer.hpp"

#include "ffcl/datastruct/graph/spanning_tree/BoruvkasAlgorithm.hpp"
#include "ffcl/hdbscan/HDBSCAN.hpp"

#include <cstddef>
#include <ostream>

namespace ffcl::benchmark {

// The HDBSCAN stages are reported separately because the MST dominates FFCL's own time and a single
// number would hide where it goes: "tree_build" is the KDTree, "mst" is Boruvka, "full" is the whole
// predict() path.
void benchmark_hdbscan(std::ostream& results, std::size_t k_nearest_neighbors, std::size_t min_cluster_size) {
    for (const auto& name : all_dataset_names()) {
        const auto dataset    = load_dataset(name);
        const auto leaf_size  = ffcl_leaf_size_for(dataset.n_samples);
        const auto n_features = dataset.n_features;

        auto base = Measurement{};
        base.library   = "ffcl";
        base.algorithm = "hdbscan";
        base.dataset   = name;
        base.n         = dataset.n_samples;
        base.leaf      = leaf_size;
        base.threads   = 1;
        base.note      = "k_nearest_neighbors=" + std::to_string(k_nearest_neighbors) +
                    " min_cluster_size=" + std::to_string(min_cluster_size);

        // --- index build ----------------------------------------------------
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
        // --- minimum spanning tree ------------------------------------------
        {
            auto measurement  = base;
            measurement.stage = "mst";

            measure_and_emit(results, measurement, [&] {
                auto indices = make_indices(dataset.n_samples);
                auto samples = dataset.data;
                auto indexer = make_indexer(indices, samples, n_features, leaf_size);

                auto boruvkas_algorithm = BoruvkasAlgorithm<IndexerType>();
                boruvkas_algorithm.set_options(
                    typename BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(k_nearest_neighbors));

                const auto mst = boruvkas_algorithm.make_tree_2(std::move(indexer));
                (void)mst;
            });
        }
        // --- full pipeline ---------------------------------------------------
        {
            auto measurement  = base;
            measurement.stage = "full";

            measure_and_emit(results, measurement, [&] {
                auto indices = make_indices(dataset.n_samples);
                auto samples = dataset.data;
                auto indexer = make_indexer(indices, samples, n_features, leaf_size);

                auto hdbscan = HDBSCAN<IndexerType>();
                hdbscan.set_options(typename HDBSCAN<IndexerType>::Options()
                                        .k_nearest_neighbors(k_nearest_neighbors)
                                        .min_cluster_size(min_cluster_size)
                                        .allow_single_cluster(true));

                const auto predictions = hdbscan.predict(std::move(indexer));
                (void)predictions;
            });
        }
    }
}

}  // namespace ffcl::benchmark
