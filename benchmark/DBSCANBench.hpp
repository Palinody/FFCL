#pragma once

#include "IO.hpp"

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
#include <vector>

#include <cmath>

namespace dbscan::benchmark {

static constexpr float       radius      = 0.4;
static constexpr std::size_t min_samples = 10;

template <typename SamplesIterator>
void inplace_cartesian_to_polar(const SamplesIterator& samples_first,
                                const SamplesIterator& samples_last,
                                std::size_t            n_features) {
    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        const auto x = samples_first[sample_index * n_features];
        const auto y = samples_first[sample_index * n_features + 1];
        // to radius
        samples_first[sample_index * n_features] = std::sqrt(x * x + y * y);
        // to angle
        samples_first[sample_index * n_features + 1] = std::atan2(y, x);
    }
}

std::vector<std::size_t> generate_indices(std::size_t n_samples) {
    std::vector<std::size_t> elements(n_samples);
    std::iota(elements.begin(), elements.end(), static_cast<std::size_t>(0));
    return elements;
}

void run_dbscan(const fs::path& filepath, const std::optional<fs::path>& predictions_filepath = {}) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    auto              data       = load_data<dType>(filepath, ' ');
    const std::size_t n_features = get_num_features_in_file(filepath);
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

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
                                   .axis_selection_policy(AxisSelectionPolicyType())
                                   .splitting_rule_policy(SplittingRulePolicyType()));

    timer.print_elapsed_seconds(9);

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
    timer.print_elapsed_seconds(9);

    if (predictions_filepath.has_value()) {
        write_data<ssize_t>(predictions, 1, predictions_filepath.value());
    }
}

void run_pointclouds_benchmarks() {
    // the path to the files from the inputs_folder
    const auto relative_path = fs::path("pointclouds_sequences/1");
    const auto filenames     = get_files_names_at_path(inputs_folder / relative_path);

    for (const auto& filename : filenames) {
        run_dbscan(inputs_folder / relative_path / filename, predictions_folder / relative_path / filename);
    }
}

}  // namespace dbscan::benchmark