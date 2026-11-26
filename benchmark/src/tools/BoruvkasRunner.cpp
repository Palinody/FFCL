// Standalone driver kept for ad-hoc timing of a single dataset, used by
// benchmark/scripts/CompareHDBSCAN.py.
//
//   BoruvkasRunner <input.txt> <output.txt|-> <dual|single|full> [k] [bucket]

#include "ffcl/benchmark/Dataset.hpp"
#include "ffcl/benchmark/Indexer.hpp"

#include "ffcl/common/Timer.hpp"

#include <fstream>

#include "ffcl/datastruct/graph/spanning_tree/BoruvkasAlgorithm.hpp"
#include "ffcl/hdbscan/HDBSCAN.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace fb = ffcl::benchmark;

int main(int argument_count, char** argument_values) {
    if (argument_count < 4) {
        std::fprintf(stderr, "usage: %s <input.txt> <output|-> <dual|single|full> [k] [bucket]\n", argument_values[0]);
        return 1;
    }
    const auto input  = std::string{argument_values[1]};
    const auto output = std::string{argument_values[2]};
    const auto mode   = std::string{argument_values[3]};

    if (mode != "dual" && mode != "single" && mode != "full") {
        std::fprintf(stderr, "error: unknown mode '%s'\n", mode.c_str());
        return 1;
    }
    const auto k = argument_count > 4 ? std::stoul(argument_values[4]) : std::size_t{1};

    const auto n_features = fb::num_features_in_file(input);
    const auto dataset    = fb::load_dataset_from_path(input, n_features);
    const auto leaf_size  = argument_count > 5 ? std::stoul(argument_values[5]) : fb::ffcl_leaf_size_for(dataset.n_samples);

    auto indices = fb::make_indices(dataset.n_samples);
    auto samples = dataset.data;

    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;
    timer.reset();

    auto indexer = fb::make_indexer(indices, samples, n_features, leaf_size);

    if (mode == "full") {
        auto hdbscan = ffcl::HDBSCAN<fb::IndexerType>();
        hdbscan.set_options(typename ffcl::HDBSCAN<fb::IndexerType>::Options()
                                .k_nearest_neighbors(k)
                                .min_cluster_size(20)
                                .allow_single_cluster(true));

        const auto predictions = hdbscan.predict(std::move(indexer));

        std::printf("RESULT mode=full n=%zu k=%zu bucket=%zu elapsed=%.6f\n",
                    dataset.n_samples, k, leaf_size, static_cast<double>(timer.elapsed()) / 1e9);
        (void)predictions;
        return 0;
    }
    auto boruvkas_algorithm = ffcl::BoruvkasAlgorithm<fb::IndexerType>();
    boruvkas_algorithm.set_options(
        typename ffcl::BoruvkasAlgorithm<fb::IndexerType>::Options().k_nearest_neighbors(k));

    const auto mst = mode == "dual" ? boruvkas_algorithm.make_tree_2(std::move(indexer))
                                    : boruvkas_algorithm.make_tree(std::move(indexer));

    const auto elapsed = static_cast<double>(timer.elapsed()) / 1e9;

    auto total_weight = 0.0;
    for (const auto& edge : mst) {
        total_weight += static_cast<double>(std::get<2>(edge));
    }
    if (output != "-") {
        if (auto stream = std::ofstream(output); stream.is_open()) {
            for (const auto& edge : mst) {
                stream << std::get<0>(edge) << " " << std::get<1>(edge) << " " << std::get<2>(edge) << "\n";
            }
        }
    }
    std::printf("RESULT mode=%s n=%zu k=%zu bucket=%zu mst=%zu total_weight=%.9f elapsed=%.6f\n",
                mode.c_str(), dataset.n_samples, k, leaf_size, mst.size(), total_weight, elapsed);

    return 0;
}
