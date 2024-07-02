#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/common/math/heuristics/Distances.hpp"

#include "ffcl/search/buffer/IndicesToBuffersMap.hpp"

#include "ffcl/datastruct/tree/kdtree/KDTree.hpp"

#include "ffcl/search/Search.hpp"

#include <iterator>
#include <tuple>
#include <unordered_map>

namespace ffcl::search::algorithms {

template <typename IndicesIterator,
          typename SamplesIterator,
          typename OtherIndicesIterator,
          typename OtherSamplesIterator>
auto simple_dual_set_shortest_edge(
    const IndicesIterator&                                                          indices_range_first,
    const IndicesIterator&                                                          indices_range_last,
    const SamplesIterator&                                                          samples_range_first,
    const SamplesIterator&                                                          samples_range_last,
    std::size_t                                                                     n_features,
    const OtherIndicesIterator&                                                     other_indices_range_first,
    const OtherIndicesIterator&                                                     other_indices_range_last,
    const OtherSamplesIterator&                                                     other_samples_range_first,
    const OtherSamplesIterator&                                                     other_samples_range_last,
    std::size_t                                                                     other_n_features,
    const buffer::Edge<typename std::iterator_traits<IndicesIterator>::value_type,
                       typename std::iterator_traits<SamplesIterator>::value_type>& initial_shortest_edge =
        buffer::make_edge(common::infinity<typename std::iterator_traits<IndicesIterator>::value_type>(),
                          common::infinity<typename std::iterator_traits<IndicesIterator>::value_type>(),
                          common::infinity<typename std::iterator_traits<SamplesIterator>::value_type>()))
    -> buffer::Edge<typename std::iterator_traits<IndicesIterator>::value_type,
                    typename std::iterator_traits<SamplesIterator>::value_type> {
    common::ignore_parameters(samples_range_last, other_samples_range_last);

    auto shortest_edge = initial_shortest_edge;

    for (auto index_it = indices_range_first; index_it != indices_range_last; ++index_it) {
        for (auto other_index_it = other_indices_range_first; other_index_it != other_indices_range_last;
             ++other_index_it) {
            const auto samples_distance = common::math::heuristics::auto_distance(
                samples_range_first + (*index_it) * n_features,
                samples_range_first + (*index_it) * n_features + n_features,
                other_samples_range_first + (*other_index_it) * other_n_features,
                other_samples_range_first + (*other_index_it) * other_n_features + other_n_features);

            if (samples_distance > 0 && samples_distance < std::get<2>(shortest_edge)) {
                shortest_edge = buffer::make_edge(*index_it, *other_index_it, samples_distance);
            }
        }
    }
    return shortest_edge;
}

template <typename IndicesIterator,
          typename SamplesIterator,
          typename OtherIndicesIterator,
          typename OtherSamplesIterator,
          typename... BufferArgs>
auto dual_set_shortest_edge(const IndicesIterator&      indices_range_first,
                            const IndicesIterator&      indices_range_last,
                            const SamplesIterator&      samples_range_first,
                            const SamplesIterator&      samples_range_last,
                            std::size_t                 n_features,
                            const OtherIndicesIterator& other_indices_range_first,
                            const OtherIndicesIterator& other_indices_range_last,
                            const OtherSamplesIterator& other_samples_range_first,
                            const OtherSamplesIterator& other_samples_range_last,
                            std::size_t                 other_n_features,
                            BufferArgs&&... buffer_args) {
    using IndexerType = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType = typename IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    auto query_indexer = IndexerType(indices_range_first,
                                     indices_range_last,
                                     samples_range_first,
                                     samples_range_last,
                                     n_features,
                                     OptionsType()
                                         .bucket_size(std::distance(indices_range_first, indices_range_last))
                                         .axis_selection_policy(AxisSelectionPolicyType{})
                                         .splitting_rule_policy(SplittingRulePolicyType{}));

    auto reference_indexer =
        IndexerType(other_indices_range_first,
                    other_indices_range_last,
                    other_samples_range_first,
                    other_samples_range_last,
                    other_n_features,
                    OptionsType()
                        .bucket_size(std::sqrt(std::distance(other_indices_range_first, other_indices_range_last)))
                        .axis_selection_policy(AxisSelectionPolicyType{})
                        .splitting_rule_policy(SplittingRulePolicyType{}));

    auto searcher = Searcher(std::move(reference_indexer));

    return searcher.dual_tree_shortest_edge(std::move(query_indexer), std::forward<BufferArgs>(buffer_args)...);
}

template <typename IndicesIterator,
          typename SamplesIterator,
          typename OtherIndicesIterator,
          typename OtherSamplesIterator,
          typename... BufferArgs>
auto dual_set_shortest_distance(const IndicesIterator&      indices_range_first,
                                const IndicesIterator&      indices_range_last,
                                const SamplesIterator&      samples_range_first,
                                const SamplesIterator&      samples_range_last,
                                std::size_t                 n_features,
                                const OtherIndicesIterator& other_indices_range_first,
                                const OtherIndicesIterator& other_indices_range_last,
                                const OtherSamplesIterator& other_samples_range_first,
                                const OtherSamplesIterator& other_samples_range_last,
                                std::size_t                 other_n_features,
                                BufferArgs&&... buffer_args) ->
    typename std::iterator_traits<SamplesIterator>::value_type {
    const auto brute_force_shortest_edge = dual_set_shortest_edge(indices_range_first,
                                                                  indices_range_last,
                                                                  samples_range_first,
                                                                  samples_range_last,
                                                                  n_features,
                                                                  other_indices_range_first,
                                                                  other_indices_range_last,
                                                                  other_samples_range_first,
                                                                  other_samples_range_last,
                                                                  other_n_features,
                                                                  std::forward<BufferArgs>(buffer_args)...);

    return std::get<2>(brute_force_shortest_edge);
}

}  // namespace ffcl::search::algorithms