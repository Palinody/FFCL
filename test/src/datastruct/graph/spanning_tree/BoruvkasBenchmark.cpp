// Benchmark counterpart of bin/BenchmarkHDBSCAN.py.
//
// Both sides use, per dataset:
//   * the same generated inputs (bin/MakeClusteringDatasets.py),
//   * the same tree parameter: bucket_size / leaf_size = sqrt(n_samples),
//   * the same HDBSCAN parameters: k_nearest_neighbors / min_samples = 1,
//     min_cluster_size = 20,
//   * the same measured phases: tree build, minimum spanning tree, full pipeline.
//
// The Python side drives hdbscan's KDTreeBoruvkaAlgorithm directly so that the
// "MST" row is the same quantity as make_tree / make_tree_2 here.
#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/graph/spanning_tree/BoruvkasAlgorithm.hpp"
#include "ffcl/datastruct/tree/kdtree/KDTree.hpp"
#include "ffcl/hdbscan/HDBSCAN.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using dType = float;

constexpr std::size_t kMinClusterSize = 20;
constexpr std::size_t kNearestNeighbors = 1;
constexpr std::size_t kRepeats          = 3;

const std::vector<std::string> kFileNames = {"noisy_circles.txt",
                                             "noisy_moons.txt",
                                             "varied.txt",
                                             "aniso.txt",
                                             "blobs.txt",
                                             "no_structure.txt",
                                             "unbalanced_blobs.txt"};

const fs::path kRootFolder = fs::path("../bin/clustering/inputs");

std::vector<dType> load_data(const fs::path& filename, char delimiter = ' ') {
    std::ifstream  filestream(filename);
    std::vector<dType> data;
    std::string    row_str, elem_str;

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
    std::getline(file, line);
    return std::count(line.begin(), line.end(), delimiter) + 1;
}

template <typename Function>
double best_of(std::size_t n_repeats, Function&& function) {
    double best = std::numeric_limits<double>::max();
    for (std::size_t repeat = 0; repeat < n_repeats; ++repeat) {
        ffcl::common::Timer<ffcl::common::Nanoseconds> timer;
        timer.reset();
        function();
        best = std::min(best, static_cast<double>(timer.elapsed()) / 1e6);
    }
    return best;
}

struct Result {
    double      tree_build_ms = 0;
    double      mst_single_ms = 0;
    double      mst_dual_ms   = 0;
    double      full_ms       = 0;
    double      mst_weight    = 0;
    std::size_t mst_size      = 0;
    std::size_t n_clusters    = 0;
};

Result run_dataset(const fs::path& filepath) {
    using IndicesIterator = std::vector<std::size_t>::iterator;
    using SamplesIterator = std::vector<dType>::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    const auto data       = load_data(filepath);
    const auto n_features = get_num_features(filepath);
    const auto n_samples  = data.size() / n_features;

    auto indices = std::vector<std::size_t>(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    Result result;

    // --- 1) tree build ------------------------------------------------------
    result.tree_build_ms = best_of(kRepeats, [&] {
        auto local_indices = indices;
        auto indexer       = IndexerType(local_indices.begin(),
                                   local_indices.end(),
                                   const_cast<std::vector<dType>&>(data).begin(),
                                   const_cast<std::vector<dType>&>(data).end(),
                                   n_features,
                                   OptionsType()
                                       .bucket_size(std::sqrt(n_samples))
                                       .max_depth(std::log2(n_samples))
                                       .axis_selection_policy(AxisSelectionPolicyType{})
                                       .splitting_rule_policy(SplittingRulePolicyType{}));
        (void)indexer;
    });

    // --- 2) minimum spanning tree ------------------------------------------
    const auto run_mst = [&](bool dual) {
        auto local_indices = indices;
        auto local_data    = data;

        auto indexer = IndexerType(local_indices.begin(),
                                   local_indices.end(),
                                   local_data.begin(),
                                   local_data.end(),
                                   n_features,
                                   OptionsType()
                                       .bucket_size(std::sqrt(n_samples))
                                       .max_depth(std::log2(n_samples))
                                       .axis_selection_policy(AxisSelectionPolicyType{})
                                       .splitting_rule_policy(SplittingRulePolicyType{}));

        auto boruvkas_algorithm = ffcl::BoruvkasAlgorithm<IndexerType>();
        boruvkas_algorithm.set_options(
            typename ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(kNearestNeighbors));

        return dual ? boruvkas_algorithm.make_tree_2(std::move(indexer))
                    : boruvkas_algorithm.make_tree(std::move(indexer));
    };

    result.mst_single_ms = best_of(kRepeats, [&] { (void)run_mst(false); });
    result.mst_dual_ms   = best_of(kRepeats, [&] { (void)run_mst(true); });

    const auto mst = run_mst(true);
    result.mst_size = mst.size();
    for (const auto& edge : mst) {
        result.mst_weight += static_cast<double>(std::get<2>(edge));
    }

    // --- 3) full pipeline ---------------------------------------------------
    result.full_ms = best_of(kRepeats, [&] {
        auto local_indices = indices;
        auto local_data    = data;

        auto indexer = IndexerType(local_indices.begin(),
                                   local_indices.end(),
                                   local_data.begin(),
                                   local_data.end(),
                                   n_features,
                                   OptionsType()
                                       .bucket_size(std::sqrt(n_samples))
                                       .max_depth(std::log2(n_samples))
                                       .axis_selection_policy(AxisSelectionPolicyType{})
                                       .splitting_rule_policy(SplittingRulePolicyType{}));

        auto hdbscan = ffcl::HDBSCAN<IndexerType>();
        hdbscan.set_options(ffcl::HDBSCAN<IndexerType>::Options()
                                .k_nearest_neighbors(kNearestNeighbors)
                                .min_cluster_size(kMinClusterSize)
                                .return_leaf_nodes(false)
                                .allow_single_cluster(true));

        const auto predictions = hdbscan.predict(std::move(indexer));

        // number of distinct non-noise labels (label 0 is noise in ffcl)
        auto labels = predictions;
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        result.n_clusters = labels.size() - (labels.empty() ? 0 : (labels.front() == 0 ? 1 : 0));
    });


    return result;
}

}  // namespace

int main() {
    std::cout << std::left << std::setw(20) << "dataset" << std::right << std::setw(7) << "n" << std::setw(7)
              << "bucket" << std::setw(11) << "build ms" << std::setw(11) << "STT ms" << std::setw(11) << "DTT ms"
              << std::setw(11) << "full ms" << std::setw(12) << "mst weight" << "\n";
    std::cout << std::string(90, '-') << "\n";

    double sum_build = 0, sum_single = 0, sum_dual = 0, sum_full = 0;

    for (const auto& name : kFileNames) {
        const auto filepath = kRootFolder / name;

        if (!fs::exists(filepath)) {
            std::cerr << "missing " << filepath << "\n";
            continue;
        }
        const auto n_samples = load_data(filepath).size() / get_num_features(filepath);

        const auto result = run_dataset(filepath);

        sum_build += result.tree_build_ms;
        sum_single += result.mst_single_ms;
        sum_dual += result.mst_dual_ms;
        sum_full += result.full_ms;

        auto short_name = name;
        short_name.erase(short_name.size() - 4);

        std::cout << std::left << std::setw(20) << short_name << std::right << std::setw(7) << n_samples
                  << std::setw(7) << static_cast<std::size_t>(std::sqrt(n_samples)) << std::setw(11) << std::fixed
                  << std::setprecision(3) << result.tree_build_ms << std::setw(11) << result.mst_single_ms
                  << std::setw(11) << result.mst_dual_ms << std::setw(11) << result.full_ms << std::setw(12)
                  << std::setprecision(6) << result.mst_weight << "\n";
    }

    const auto n = static_cast<double>(kFileNames.size());

    std::cout << std::string(90, '-') << "\n";
    std::cout << std::left << std::setw(20) << "mean" << std::right << std::setw(7) << "" << std::setw(7) << ""
              << std::setw(11) << std::fixed << std::setprecision(3) << sum_build / n << std::setw(11)
              << sum_single / n << std::setw(11) << sum_dual / n << std::setw(11) << sum_full / n << "\n";

    return 0;
}
