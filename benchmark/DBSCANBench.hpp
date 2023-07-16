#pragma once

#include "IO.hpp"

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDTreeIndexed.hpp"
#include "ffcl/dbscan/DBSCAN.hpp"

#include <sys/types.h>  // std::ssize_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace benchmark::dbscan {

std::vector<std::size_t> generate_indices(std::size_t n_samples) {
    std::vector<std::size_t> elements(n_samples);
    std::iota(elements.begin(), elements.end(), static_cast<std::size_t>(0));
    return elements;
}

void noisy_circles_bench() {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "noisy_circles.txt";

    auto              data       = load_data<dType>(inputs_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator         = decltype(indices)::iterator;
    using SamplesIterator         = decltype(data)::iterator;
    using IndexerType             = ffcl::containers::KDTreeIndexed<IndicesIterator, SamplesIterator>;
    using OptionsType             = IndexerType::Options;
    using AxisSelectionPolicyType = kdtree::policy::IndexedMaximumSpreadBuild<IndicesIterator, SamplesIterator>;
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

    dbscan.set_options(ffcl::DBSCAN<IndexerType>::Options().radius(2).min_samples(5));

    timer.reset();

    const float radius = 2;
    // /*
    const auto predictions = dbscan.predict(indexer, &IndexerType::radius_search_around_query_index, radius);
    // */
    /*
    const float edge        = std::sqrt(2.0) * radius;
    const auto  predictions = dbscan.predict(indexer,
                                            &IndexerType::range_search_around_query_index,
                                            HyperRangeType<SamplesIterator>({{-edge, edge}, {-edge, edge}}));
    */
    timer.print_elapsed_seconds(9);

    write_data<ssize_t>(predictions, 1, predictions_folder / fs::path(filename));
}

}  // namespace benchmark::dbscan