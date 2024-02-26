#pragma once

#include "ffcl/common/math/heuristics/Distances.hpp"

#include "ffcl/common/Utils.hpp"

#include <iterator>
#include <tuple>

namespace ffcl::common::algorithms {

template <typename Index, typename Distance>
using Edge = std::tuple<Index, Index, Distance>;

template <typename Index, typename Distance>
constexpr auto make_edge(const Index& index_1, const Index& index_2, const Distance& distance) {
    return std::make_tuple(index_1, index_2, distance);
}

template <typename IndicesIterator,
          typename SamplesIterator,
          typename OtherIndicesIterator,
          typename OtherSamplesIterator,
          typename EdgeType>
void dual_set_closest_edge(const IndicesIterator&      indices_range_first,
                           const IndicesIterator&      indices_range_last,
                           const SamplesIterator&      samples_range_first,
                           const SamplesIterator&      samples_range_last,
                           std::size_t                 n_features,
                           const OtherIndicesIterator& other_indices_range_first,
                           const OtherIndicesIterator& other_indices_range_last,
                           const OtherSamplesIterator& other_samples_range_first,
                           const OtherSamplesIterator& other_samples_range_last,
                           std::size_t                 other_n_features,
                           EdgeType&                   shortest_edge) {
    common::ignore_parameters(samples_range_last, other_samples_range_last);

    for (auto index_it = indices_range_first; index_it != indices_range_last; ++index_it) {
        for (auto other_index_it = other_indices_range_first; other_index_it != other_indices_range_last;
             ++other_index_it) {
            const auto samples_distance = common::math::heuristics::auto_distance(
                samples_range_first + *index_it * n_features,
                samples_range_first + *index_it * n_features + n_features,
                other_samples_range_first + *other_index_it * other_n_features,
                other_samples_range_first + *other_index_it * other_n_features + other_n_features);

            if (samples_distance < std::get<2>(shortest_edge)) {
                shortest_edge = make_edge(*index_it, *other_index_it, samples_distance);
            }
        }
    }
}

template <typename IndicesIterator,
          typename SamplesIterator,
          typename OtherIndicesIterator,
          typename OtherSamplesIterator>
auto dual_set_closest_edge(const IndicesIterator&      indices_range_first,
                           const IndicesIterator&      indices_range_last,
                           const SamplesIterator&      samples_range_first,
                           const SamplesIterator&      samples_range_last,
                           std::size_t                 n_features,
                           const OtherIndicesIterator& other_indices_range_first,
                           const OtherIndicesIterator& other_indices_range_last,
                           const OtherSamplesIterator& other_samples_range_first,
                           const OtherSamplesIterator& other_samples_range_last,
                           std::size_t                 other_n_features) {
    common::ignore_parameters(samples_range_last, other_samples_range_last);

    using IndexType    = typename std::iterator_traits<IndicesIterator>::value_type;
    using DistanceType = typename std::iterator_traits<SamplesIterator>::value_type;

    auto shortest_edge = make_edge(IndexType{}, IndexType{}, common::infinity<DistanceType>());

    dual_set_closest_edge(indices_range_first,
                          indices_range_last,
                          samples_range_first,
                          samples_range_last,
                          n_features,
                          other_indices_range_first,
                          other_indices_range_last,
                          other_samples_range_first,
                          other_samples_range_last,
                          other_n_features,
                          shortest_edge);
    return shortest_edge;
}

template <typename NodePtr,
          typename SamplesIterator,
          typename OtherNodePtr,
          typename OtherSamplesIterator,
          typename EdgeType>
void dual_set_closest_edge(NodePtr                     node,
                           const SamplesIterator&      samples_range_first,
                           const SamplesIterator&      samples_range_last,
                           std::size_t                 n_features,
                           OtherNodePtr                other_node,
                           const OtherSamplesIterator& other_samples_range_first,
                           const OtherSamplesIterator& other_samples_range_last,
                           std::size_t                 other_n_features,
                           EdgeType&                   shortest_edge) {
    auto clostest_edge = dual_set_closest_edge(node->indices_range_.first,
                                               node->indices_range_.second,
                                               samples_range_first,
                                               samples_range_last,
                                               n_features,
                                               other_node->indices_range_.first,
                                               other_node->indices_range_.second,
                                               other_samples_range_first,
                                               other_samples_range_last,
                                               other_n_features,
                                               shortest_edge);

    const bool is_node_leaf       = node->is_leaf();
    const bool is_other_node_leaf = other_node->is_leaf();

    if (is_node_leaf && !is_other_node_leaf) {
        dual_set_closest_edge(node,
                              samples_range_first,
                              samples_range_last,
                              n_features,
                              other_node->right_,
                              other_samples_range_first,
                              other_samples_range_last,
                              other_n_features,
                              shortest_edge);

        dual_set_closest_edge(node,
                              samples_range_first,
                              samples_range_last,
                              n_features,
                              other_node->right_,
                              other_samples_range_first,
                              other_samples_range_last,
                              other_n_features,
                              shortest_edge);

    } else if (!is_node_leaf && is_other_node_leaf) {
        dual_set_closest_edge(node->left_,
                              samples_range_first,
                              samples_range_last,
                              n_features,
                              other_node,
                              other_samples_range_first,
                              other_samples_range_last,
                              other_n_features,
                              shortest_edge);

        dual_set_closest_edge(node->right_,
                              samples_range_first,
                              samples_range_last,
                              n_features,
                              other_node,
                              other_samples_range_first,
                              other_samples_range_last,
                              other_n_features,
                              shortest_edge);

    } else if (!is_node_leaf && !is_other_node_leaf) {
        dual_set_closest_edge(node->left_,
                              samples_range_first,
                              samples_range_last,
                              n_features,
                              other_node->left_,
                              other_samples_range_first,
                              other_samples_range_last,
                              other_n_features,
                              shortest_edge);

        dual_set_closest_edge(node->right_,
                              samples_range_first,
                              samples_range_last,
                              n_features,
                              other_node->left_,
                              other_samples_range_first,
                              other_samples_range_last,
                              other_n_features,
                              shortest_edge);

        dual_set_closest_edge(node->left_,
                              samples_range_first,
                              samples_range_last,
                              n_features,
                              other_node->right_,
                              other_samples_range_first,
                              other_samples_range_last,
                              other_n_features,
                              shortest_edge);

        dual_set_closest_edge(node->right_,
                              samples_range_first,
                              samples_range_last,
                              n_features,
                              other_node->right_,
                              other_samples_range_first,
                              other_samples_range_last,
                              other_n_features,
                              shortest_edge);
    }
}

template <typename NodePtr, typename SamplesIterator, typename OtherNodePtr, typename OtherSamplesIterator>
auto dual_set_closest_edge(NodePtr                     node,
                           const SamplesIterator&      samples_range_first,
                           const SamplesIterator&      samples_range_last,
                           std::size_t                 n_features,
                           OtherNodePtr                other_node,
                           const OtherSamplesIterator& other_samples_range_first,
                           const OtherSamplesIterator& other_samples_range_last,
                           std::size_t                 other_n_features) {
    using IndexType    = typename NodePtr::value_type;
    using DistanceType = typename std::iterator_traits<SamplesIterator>::value_type;

    auto shortest_edge = make_edge(IndexType{}, IndexType{}, common::infinity<DistanceType>());

    dual_set_closest_edge(node,
                          samples_range_first,
                          samples_range_last,
                          n_features,
                          other_node,
                          other_samples_range_first,
                          other_samples_range_last,
                          other_n_features,
                          shortest_edge);
    return shortest_edge;
}

}  // namespace ffcl::common::algorithms
