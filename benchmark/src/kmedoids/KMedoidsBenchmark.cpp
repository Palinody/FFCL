#include "ffcl/benchmark/Dataset.hpp"
#include "ffcl/benchmark/Harness.hpp"

#include "ffcl/kmedoids/KMedoids.hpp"

#include <cstddef>
#include <ostream>

namespace ffcl::benchmark {

void benchmark_kmedoids(std::ostream& results, std::size_t n_medoids) {
    for (const auto& name : all_dataset_names()) {
        const auto dataset = load_dataset(name);

        auto measurement  = Measurement{};
        measurement.library   = "ffcl";
        measurement.algorithm = "kmedoids";
        measurement.stage     = "full";
        measurement.dataset   = name;
        measurement.n         = dataset.n_samples;
        measurement.leaf      = 0;
        measurement.threads   = 1;
        measurement.note      = "n_medoids=" + std::to_string(n_medoids) + " algorithm=FasterPAM";

        measure_and_emit(results, measurement, [&] {
            auto kmedoids = KMedoids<float>(n_medoids);

            const auto medoids = kmedoids.fit(dataset.data.begin(), dataset.data.end(), dataset.n_features);
            (void)medoids;
        });
    }
}

}  // namespace ffcl::benchmark
