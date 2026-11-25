#include "ffcl/benchmark/Dataset.hpp"
#include "ffcl/benchmark/Harness.hpp"

#include "ffcl/kmeans/KMeans.hpp"

#include <cstddef>
#include <ostream>

namespace ffcl::benchmark {

void benchmark_kmeans(std::ostream& results, std::size_t n_centroids) {
    for (const auto& name : all_dataset_names()) {
        const auto dataset = load_dataset(name);

        auto measurement  = Measurement{};
        measurement.library   = "ffcl";
        measurement.algorithm = "kmeans";
        measurement.stage     = "full";
        measurement.dataset   = name;
        measurement.n         = dataset.n_samples;
        measurement.leaf      = 0;  // KMeans does not use a KDTree
        measurement.threads   = 1;
        measurement.note      = "n_centroids=" + std::to_string(n_centroids) + " init=kmeans++";

        measure_and_emit(results, measurement, [&] {
            auto kmeans = KMeans<float>(n_centroids);

            const auto centroids = kmeans.fit(dataset.data.begin(), dataset.data.end(), dataset.n_features);
            (void)centroids;
        });
    }
}

}  // namespace ffcl::benchmark
