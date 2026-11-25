// Minimal driver used by bin/CompareHDBSCAN.py.
//
// Build it twice, once per thread configuration, so that the sequential and the OpenMP build can be
// compared without recompiling:
//
//   cmake .. -DMODE=gtest -DTHREADS_ENABLED=false && make BoruvkasRunner   # -> build/BoruvkasRunner
//   cmake .. -DMODE=gtest -DTHREADS_ENABLED=true  && make BoruvkasRunner   # -> build/BoruvkasRunner
//
// Usage:
//   BoruvkasRunner <input.txt> <output.txt|-> <dual|single|full> [k_nearest_neighbors] [bucket_size]
//
// Prints one machine readable RESULT line.
#include "ffcl/common/Timer.hpp"
#include "ffcl/datastruct/graph/spanning_tree/BoruvkasAlgorithm.hpp"
#include "ffcl/datastruct/tree/kdtree/KDTree.hpp"
#include "ffcl/hdbscan/HDBSCAN.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

namespace {

using dType = float;

std::vector<dType> load_data(const fs::path& filename, char delimiter = ' ') {
    std::ifstream filestream(filename);

    if (!filestream.is_open()) {
        std::fprintf(stderr, "error: cannot open input file '%s'\n", filename.string().c_str());
        std::exit(1);
    }
    std::vector<dType> data;
    std::string        row_str, elem_str;

    while (std::getline(filestream, row_str, '\n')) {
        std::stringstream row_str_stream(row_str);
        while (std::getline(row_str_stream, elem_str, delimiter)) {
            if (!elem_str.empty()) {
                data.emplace_back(std::stof(elem_str));
            }
        }
    }
    return data;
}

std::size_t get_num_features(const fs::path& filename, char delimiter = ' ') {
    std::ifstream file(filename);
    std::string   line;

    if (!std::getline(file, line)) {
        std::fprintf(stderr, "error: input file '%s' is empty\n", filename.string().c_str());
        std::exit(1);
    }
    return std::count(line.begin(), line.end(), delimiter) + 1;
}

auto is_valid_mode(const std::string& mode) -> bool {
    return mode == "dual" || mode == "single" || mode == "full";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <input> <output|-> <dual|single|full> [k] [bucket]\n", argv[0]);
        return 1;
    }
    const fs::path    input{argv[1]};
    const std::string output{argv[2]};
    const std::string mode{argv[3]};
    if (!is_valid_mode(mode)) {
        std::fprintf(stderr, "error: unknown mode '%s' (expected dual, single or full)\n", mode.c_str());
        std::fprintf(stderr, "usage: %s <input> <output|-> <dual|single|full> [k] [bucket]\n", argv[0]);
        return 1;
    }
    const std::size_t k = (argc > 4) ? std::stoul(argv[4]) : 1;

    auto              data       = load_data(input);
    const std::size_t n_features = get_num_features(input);
    const std::size_t n_samples  = data.size() / n_features;

    if (n_samples < 2) {
        std::fprintf(stderr,
                     "error: '%s' holds %zu sample(s), at least 2 are needed\n",
                     input.string().c_str(),
                     n_samples);
        return 1;
    }

    // Same default as the tests and the Python benchmark: sqrt(n) / 3, clamped.
    const std::size_t bucket_size = (argc > 5)
                                        ? std::stoul(argv[5])
                                        : std::max<std::size_t>(16, static_cast<std::size_t>(std::sqrt(n_samples) / 3));

    if (bucket_size < 1) {
        std::fprintf(stderr, "error: bucket size must be >= 1\n");
        return 1;
    }

    auto indices = std::vector<std::size_t>(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(bucket_size)
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType{})
                                   .splitting_rule_policy(SplittingRulePolicyType{}));

    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;
    timer.reset();

    if (mode == "full") {
        auto hdbscan = ffcl::HDBSCAN<IndexerType>();
        hdbscan.set_options(ffcl::HDBSCAN<IndexerType>::Options()
                                .k_nearest_neighbors(k)
                                .min_cluster_size(20)
                                .return_leaf_nodes(false)
                                .allow_single_cluster(true));

        const auto predictions = hdbscan.predict(std::move(indexer));

        std::printf("RESULT mode=full n=%zu k=%zu bucket=%zu elapsed=%.6f labels=%zu\n",
                    n_samples,
                    k,
                    bucket_size,
                    static_cast<double>(timer.elapsed()) / 1e9,
                    predictions.size());
        return 0;
    }
    auto boruvkas_algorithm = ffcl::BoruvkasAlgorithm<IndexerType>();
    boruvkas_algorithm.set_options(typename ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(k));

    auto minimum_spanning_tree = (mode == "dual") ? boruvkas_algorithm.make_tree_2(std::move(indexer))
                                                  : boruvkas_algorithm.make_tree(std::move(indexer));

    const auto elapsed_seconds = static_cast<double>(timer.elapsed()) / 1e9;

    double      total_weight = 0;
    std::size_t n_non_finite = 0;

    for (const auto& edge : minimum_spanning_tree) {
        const auto weight = static_cast<double>(std::get<2>(edge));

        if (std::isfinite(weight)) {
            total_weight += weight;
        } else {
            ++n_non_finite;
        }
    }
    if (output != "-") {
        std::ofstream filestream(output);
        for (const auto& edge : minimum_spanning_tree) {
            filestream << std::get<0>(edge) << " " << std::get<1>(edge) << " " << std::get<2>(edge) << "\n";
        }
    }
    std::printf("RESULT mode=%s n=%zu k=%zu bucket=%zu mst=%zu total_weight=%.9f n_non_finite=%zu elapsed=%.6f\n",
                mode.c_str(),
                n_samples,
                k,
                bucket_size,
                minimum_spanning_tree.size(),
                total_weight,
                n_non_finite,
                elapsed_seconds);
    return 0;
}
