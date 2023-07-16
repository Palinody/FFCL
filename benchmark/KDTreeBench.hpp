#pragma once

#include "IO.hpp"

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDTreeIndexed.hpp"

namespace benchmark::kdtree {

constexpr std::size_t n_neighbors = 5;
constexpr dType       radius      = 1;

std::vector<std::size_t> generate_indices(std::size_t n_samples) {
    std::vector<std::size_t> elements(n_samples);
    std::iota(elements.begin(), elements.end(), static_cast<std::size_t>(0));
    return elements;
}

void radius_search_around_query_index_varied_bench() {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    timer.reset();

    auto indices = generate_indices(n_samples);

    using IndicesIterator         = decltype(indices)::iterator;
    using SamplesIterator         = decltype(data)::iterator;
    using IndexerType             = ffcl::containers::KDTreeIndexed<IndicesIterator, SamplesIterator>;
    using OptionsType             = IndexerType::Options;
    using AxisSelectionPolicyType = kdtree::policy::IndexedMaximumSpreadBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType = kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>;

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

    std::vector<std::size_t> nn_histogram(n_samples);

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        auto nearest_neighbors_buffer = kdtree.radius_search_around_query_index(indices[sample_index_query], radius);

        nn_histogram[indices[sample_index_query]] = nearest_neighbors_buffer.size();
    }
    timer.print_elapsed_seconds(9);

    std::cout << "radius_search_around_query_index (histogram)\n";
    print_data(nn_histogram, n_samples);
}

}  // namespace benchmark::kdtree