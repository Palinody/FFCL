#pragma once

#include "IO.hpp"

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDTreeIndexed.hpp"
#include "ffcl/math/random/Sampling.hpp"

namespace kdtree::benchmark {

constexpr std::size_t n_neighbors = 5;
constexpr dType       radius      = 1;

std::vector<std::size_t> generate_indices(std::size_t n_samples) {
    std::vector<std::size_t> elements(n_samples);
    std::iota(elements.begin(), elements.end(), static_cast<std::size_t>(0));
    return elements;
}

template <typename Type>
void print_data(const std::vector<Type>& data, std::size_t n_features) {
    if (!n_features) {
        return;
    }
    const std::size_t n_samples = data.size() / n_features;

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            std::cout << data[sample_index * n_features + feature_index] << " ";
        }
        std::cout << "\n";
    }
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
        auto nearest_neighbors_buffer = indexer.radius_search_around_query_index(indices[sample_index_query], radius);

        nn_histogram[indices[sample_index_query]] = nearest_neighbors_buffer.size();
    }
    timer.print_elapsed_seconds(9);

    std::cout << "radius_search_around_query_index (histogram)\n";
    print_data(nn_histogram, n_samples);
}

void test() {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    constexpr auto filenames_list = std::array<fs::path, 7>{/**/ "noisy_circles",
                                                            /**/ "noisy_moons",
                                                            /**/ "varied",
                                                            /**/ "aniso",
                                                            /**/ "blobs",
                                                            /**/ "no_structure",
                                                            /**/ "unbalanced_blobs"};

    const auto n_samples_list = std::array<std::size_t, 2>{10, 20};

    const auto n_samples_max = *std::max_element(n_samples_list.begin(), n_samples_list.end());

    auto indices = generate_indices(n_samples);

    for (const auto& filename : filenames_list) {
        auto              data       = load_data<dType>(inputs_folder / filename, ' ');
        const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

        for (const auto& n_samples : n_samples_list) {
            auto sampled_indices = math::random::select_from_range(n_samples, {0, n_samples_max});
            //
        }
    }
}

}  // namespace kdtree::benchmark