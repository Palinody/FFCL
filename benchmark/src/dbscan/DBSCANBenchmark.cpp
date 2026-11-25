#include "ffcl/benchmark/Dataset.hpp"
#include "ffcl/benchmark/Harness.hpp"
#include "ffcl/benchmark/Indexer.hpp"

#include "ffcl/dbscan/DBSCAN.hpp"

#include <cstddef>
#include <ostream>

namespace ffcl::benchmark {

void benchmark_dbscan(std::ostream& results, float radius, std::size_t min_samples) {
    for (const auto& name : all_dataset_names()) {
        const auto dataset    = load_dataset(name);
        const auto leaf_size  = ffcl_leaf_size_for(dataset.n_samples);
        const auto n_features = dataset.n_features;

        auto measurement  = Measurement{};
        measurement.library   = "ffcl";
        measurement.algorithm = "dbscan";
        measurement.stage     = "full";
        measurement.dataset   = name;
        measurement.n         = dataset.n_samples;
        measurement.leaf      = leaf_size;
        measurement.threads   = 1;
        measurement.note = "radius=" + std::to_string(radius) + " min_samples=" + std::to_string(min_samples);

        measure_and_emit(results, measurement, [&] {
            auto indices = make_indices(dataset.n_samples);
            auto samples = dataset.data;
            auto indexer = make_indexer(indices, samples, n_features, leaf_size);

            auto dbscan = DBSCAN<IndexerType>();
            dbscan.set_options(typename DBSCAN<IndexerType>::Options().radius(radius).min_samples(min_samples));

            const auto predictions = dbscan.predict(std::move(indexer));
            (void)predictions;
        });
    }
}

}  // namespace ffcl::benchmark
