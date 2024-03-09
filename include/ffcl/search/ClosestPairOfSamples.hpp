#pragma once

#include "ffcl/common/math/heuristics/Distances.hpp"

#include "ffcl/common/Utils.hpp"

#include <iterator>
#include <tuple>

namespace ffcl::search::algorithms {

template <typename Index, typename Distance>
using Edge = std::tuple<Index, Index, Distance>;

template <typename Index, typename Distance>
constexpr auto make_edge(const Index& index_1, const Index& index_2, const Distance& distance) {
    return std::make_tuple(index_1, index_2, distance);
}

template <typename IndicesIterator,
          typename SamplesIterator,
          typename OtherIndicesIterator,
          typename OtherSamplesIterator>
auto dual_set_shortest_edge(
    const IndicesIterator&                                                  indices_range_first,
    const IndicesIterator&                                                  indices_range_last,
    const SamplesIterator&                                                  samples_range_first,
    const SamplesIterator&                                                  samples_range_last,
    std::size_t                                                             n_features,
    const OtherIndicesIterator&                                             other_indices_range_first,
    const OtherIndicesIterator&                                             other_indices_range_last,
    const OtherSamplesIterator&                                             other_samples_range_first,
    const OtherSamplesIterator&                                             other_samples_range_last,
    std::size_t                                                             other_n_features,
    const Edge<typename std::iterator_traits<IndicesIterator>::value_type,
               typename std::iterator_traits<SamplesIterator>::value_type>& initial_shortest_edge =
        make_edge(common::infinity<typename std::iterator_traits<IndicesIterator>::value_type>(),
                  common::infinity<typename std::iterator_traits<IndicesIterator>::value_type>(),
                  common::infinity<typename std::iterator_traits<SamplesIterator>::value_type>()))
    -> Edge<typename std::iterator_traits<IndicesIterator>::value_type,
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
                shortest_edge = make_edge(*index_it, *other_index_it, samples_distance);
            }
        }
    }
    return shortest_edge;
}

}  // namespace ffcl::search::algorithms
