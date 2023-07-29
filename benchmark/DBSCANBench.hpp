#pragma once

#include "IO.hpp"
#include "Utils.hpp"

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDTreeIndexed.hpp"
#include "ffcl/dbscan/DBSCAN.hpp"

#include "ffcl/math/statistics/Statistics.hpp"

#include <sys/types.h>  // std::ssize_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <tuple>
#include <vector>

namespace dbscan::benchmark {

static constexpr float       radius      = 0.5;
static constexpr std::size_t min_samples = 10;

utils::DurationsSummary run_dbscan(const fs::path& filepath, const std::optional<fs::path>& predictions_filepath) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    std::vector<dType> data;
    std::size_t        n_samples, n_features;

    if (filepath.extension().string() == ".bin") {
        std::tie(data, n_samples, n_features) = bench::io::bin::decode(/*n_features=*/4, filepath);

    } else if (filepath.extension().string() == ".txt") {
        data       = bench::io::txt::load_data<dType>(filepath, ' ');
        n_features = bench::io::txt::get_num_features_in_file(filepath);
        n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    } else {
        char message[100];
        std::sprintf(message, "File extension found '%s' but only supports .txt or .bin", filepath.extension().c_str());
        throw std::runtime_error(message);
    }

    utils::DurationsSummary bench_summary;

    bench_summary.n_samples  = n_samples;
    bench_summary.n_features = n_features;

    timer.reset();

    auto indices = utils::generate_indices(n_samples);

    using IndicesIterator         = decltype(indices)::iterator;
    using SamplesIterator         = decltype(data)::iterator;
    using IndexerType             = ffcl::containers::KDTreeIndexed<IndicesIterator, SamplesIterator>;
    using OptionsType             = IndexerType::Options;
    using AxisSelectionPolicyType = kdtree::policy::IndexedHighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType = kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>;

    timer.reset();

    // IndexedHighestVarianceBuild, IndexedMaximumSpreadBuild, IndexedCycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType().feature_mask({0, 1, 2}))
                                   .splitting_rule_policy(SplittingRulePolicyType()));

    bench_summary.indexer_build_duration = timer.elapsed();

    auto dbscan = ffcl::DBSCAN<IndexerType>();

    dbscan.set_options(ffcl::DBSCAN<IndexerType>::Options().radius(radius).min_samples(min_samples));

    timer.reset();

    // /*
    const auto predictions = dbscan.predict(indexer, &IndexerType::radius_search_around_query_index, radius);
    // */
    /*
    const auto predictions =
        dbscan.predict(indexer,
                       &IndexerType::range_search_around_query_index,
                       HyperRangeType<SamplesIterator>({{-0.25, 0.25}, {-0.25, 0.25}, {-0.5, 0.5}}));
    */

    bench_summary.indexer_query_duration = timer.elapsed();

    bench_summary.total_duration = bench_summary.indexer_build_duration + bench_summary.indexer_query_duration;

    if (predictions_filepath.has_value()) {
        // bench::io::txt::write_data<std::size_t>(predictions, 1, predictions_filepath.value());
        bench::io::bin::encode(predictions, predictions_filepath.value());
    }
    return bench_summary;
}

void run_pointclouds_benchmarks() {
    // the path to the files from the inputs_folder
    const auto relative_path = fs::path("pointclouds_sequences/0000");
    const auto filenames     = bench::io::get_files_names_at_path(inputs_folder / relative_path);

    // Conversion factor for nanoseconds to seconds
    long double to_seconds = 1e-9;

    // the sequence object that will be used to compute the variance
    std::vector<utils::DurationsSummary> bench_summary_vector;
    bench_summary_vector.reserve(filenames.size());
    // the object that will be used to compute the mean
    utils::DurationsSummary bench_summary_mean;

    for (std::size_t file_index = 0; file_index < filenames.size(); ++file_index) {
        const auto& filename = filenames[file_index];

        auto bench_summary =
            run_dbscan(inputs_folder / relative_path / filename, predictions_folder / relative_path / filename);

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

    printf("DBSCAN (FFCL) computation time average ± variance (n_samples/queries: %.2Lf ± %.2Lf | n_features: %ld):"
           "\n\tbuild: %.12Lf ± %.12Lf"
           "\n\tpredictions: %.12Lf ± %.12Lf"
           "\n\ttotal: %.12Lf ± %.12Lf\n",
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

}  // namespace dbscan::benchmark