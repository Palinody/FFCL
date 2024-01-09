#pragma once

#include "IO.hpp"
#include "Utils.hpp"

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/tree/kdtree/KDTree.hpp"
#include "ffcl/hdbscan/HDBSCAN.hpp"

#include <sys/types.h>  // std::ssize_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <tuple>
#include <vector>

namespace hdbscan::benchmark {

namespace ffcl_ {

utils::DurationsSummary run_hdbscan(const fs::path&                filepath,
                                    const std::optional<fs::path>& predictions_filepath,
                                    std::size_t                    k_nearest_neighbors,
                                    std::size_t                    min_cluster_size) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    std::vector<bench::io::DataType> data;
    std::size_t                      n_samples, n_features;

    if (filepath.extension().string() == ".bin") {
        std::tie(data, n_samples, n_features) = bench::io::bin::decode(/*n_features=*/4, filepath);

    } else if (filepath.extension().string() == ".txt") {
        data       = bench::io::txt::load_data<bench::io::DataType>(filepath, ' ');
        n_features = bench::io::txt::get_num_features_in_file(filepath);
        n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    } else {
        char message[100];
        std::sprintf(message, "File extension found '%s' but only supports .txt or .bin", filepath.extension().c_str());
        throw std::runtime_error(message);
    }

    utils::DurationsSummary bench_summary;

    static constexpr std::size_t n_features_target = 3;

    bench_summary.n_samples  = n_samples;
    bench_summary.n_features = n_features_target;

    auto data_xyz = std::vector<bench::io::DataType>(n_samples * n_features_target);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // Each point represents one row of the 2D matrix (n_features-dimensional point)
        data_xyz[sample_index * n_features_target]     = data[sample_index * n_features];
        data_xyz[sample_index * n_features_target + 1] = data[sample_index * n_features + 1];
        data_xyz[sample_index * n_features_target + 2] = data[sample_index * n_features + 2];
    }

    auto indices = utils::generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data_xyz)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    timer.reset();

    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data_xyz.begin(),
                               data_xyz.end(),
                               n_features_target,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType())
                                   .splitting_rule_policy(SplittingRulePolicyType()));

    bench_summary.indexer_build_duration = timer.elapsed();

    auto hdbscan = ffcl::HDBSCAN<IndexerType>();

    hdbscan.set_options(ffcl::HDBSCAN<IndexerType>::Options()
                            .k_nearest_neighbors(k_nearest_neighbors)
                            .min_cluster_size(min_cluster_size)
                            .return_leaf_nodes(false)
                            .allow_single_cluster(true));

    timer.reset();

    const auto predictions = hdbscan.predict(std::move(indexer));

    bench_summary.indexer_query_duration = timer.elapsed();

    bench_summary.total_duration = bench_summary.indexer_build_duration + bench_summary.indexer_query_duration;

    if (predictions_filepath.has_value()) {
        if (filepath.extension().string() == ".bin") {
            bench::io::bin::encode(predictions, predictions_filepath.value());

        } else if (filepath.extension().string() == ".txt") {
            bench::io::txt::write_data(predictions, 1, predictions_filepath.value());

        } else {
            char message[100];
            std::sprintf(
                message, "File extension found '%s' but only supports .txt or .bin", filepath.extension().c_str());
            throw std::runtime_error(message);
        }
    }
    return bench_summary;
}

}  // namespace ffcl_

template <typename Function, typename... Args>
void run_pointclouds_sequences_benchmark(const Function&    function,
                                         const fs::path&    relative_path,
                                         bool               write_predictions,
                                         const std::string& library_name,
                                         Args&&... args) {
    // the path to the files from the inputs_folder
    // const auto relative_path = fs::path("pointclouds_sequences/2");
    const auto filenames = bench::io::get_files_names_at_path(bench::io::inputs_folder / relative_path);

    // Conversion factor for nanoseconds to seconds
    long double to_seconds = 1e-9;

    // the sequence object that will be used to compute the variance
    std::vector<utils::DurationsSummary> bench_summary_vector;
    bench_summary_vector.reserve(filenames.size());
    // the object that will be used to compute the mean
    utils::DurationsSummary bench_summary_mean;

    for (std::size_t file_index = 0; file_index < filenames.size(); ++file_index) {
        const auto& filename = filenames[file_index];

        auto bench_summary = function(bench::io::inputs_folder / relative_path / filename,
                                      write_predictions ? static_cast<std::optional<std::filesystem::path>>(
                                                              bench::io::predictions_folder / relative_path / filename)
                                                        : std::nullopt,
                                      std::forward<Args>(args)...);

        bench_summary.apply_timer_multiplier(to_seconds);

        bench_summary_mean += bench_summary;

        bench_summary_vector.emplace_back(std::move(bench_summary));

        utils::print_progress_bar(file_index, filenames.size());
    }
    bench_summary_mean /= filenames.size();

    utils::DurationsSummary bench_summary_variance;

    for (auto& bench_summary : bench_summary_vector) {
        bench_summary -= bench_summary_mean;
        bench_summary *= bench_summary;
        bench_summary_variance += bench_summary;
    }
    bench_summary_variance /= filenames.size();

    printf("%s computation time average ± variance (n_samples/queries: %.2Lf ± %.2Lf | n_features: %ld):"
           "\n\tbuild: %.12Lf ± %.12Lf"
           "\n\tqueries: %.12Lf ± %.12Lf"
           "\n\ttotal: %.12Lf ± %.12Lf\n",
           library_name.c_str(),
           bench_summary_mean.n_samples,
           bench_summary_variance.n_samples,
           static_cast<std::size_t>(bench_summary_mean.n_features),
           bench_summary_mean.indexer_build_duration,
           bench_summary_variance.indexer_build_duration,
           bench_summary_mean.indexer_query_duration,
           bench_summary_variance.indexer_query_duration,
           bench_summary_mean.total_duration,
           bench_summary_variance.total_duration);
}

void run_point_cloud_sequences() {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    // const std::vector<std::size_t> k_nearest_neighbors = {3, 5, 10};
    const std::vector<std::size_t> k_nearest_neighbors = {5};
    // const std::vector<std::size_t> min_cluster_sizes   = {5, 10, 15, 20};
    const std::vector<std::size_t> min_cluster_sizes = {15};

    const std::vector<fs::path> relative_paths = {fs::path("pointclouds_sequences/1"),
                                                  fs::path("pointclouds_sequences/2"),
                                                  fs::path("pointclouds_sequences/0000"),
                                                  fs::path("pointclouds_sequences/0001")};

    for (const auto& knn : k_nearest_neighbors) {
        std::cout << "Running benchmarks with knn: " << knn << "\n";
        for (const auto& min_cluster_size : min_cluster_sizes) {
            std::cout << "Running benchmarks with min_cluster_size: " << min_cluster_size << "\n";
            for (const auto& relative_path : relative_paths) {
                std::cout << "---\nSequence in folder: " << relative_path.c_str() << "\n\n";

                timer.reset();
                run_pointclouds_sequences_benchmark(
                    &ffcl_::run_hdbscan, relative_path, true, "FFCL", knn, min_cluster_size);
                timer.print_elapsed_seconds();

                std::cout << "\n---\n";
            }
        }
    }
}

}  // namespace hdbscan::benchmark