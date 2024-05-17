#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/StaticBuffer.hpp"

#include "ffcl/datastruct/bounds/StaticBoundWithCentroid.hpp"

#include <algorithm>
#include <cstddef>  // std::ptrdiff_t
#include <memory>

#include "rapidjson/writer.h"

namespace ffcl::datastruct {

template <typename IndicesIterator, typename Bound>
struct KDNodeView {
    using IndicesIteratorType = IndicesIterator;

    static_assert(common::is_iterator<IndicesIteratorType>::value, "IndicesIteratorType is not an iterator");

    using IndexType = typename std::iterator_traits<IndicesIterator>::value_type;
    using DataType  = typename Bound::ValueType;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    using NodeType = KDNodeView<IndicesIterator, Bound>;
    using NodePtr  = std::shared_ptr<NodeType>;

    using IteratorPairType = std::pair<IndicesIterator, IndicesIterator>;

    using BoundType = Bound;

    static_assert(common::is_crtp_of<BoundType, bounds::StaticBoundWithCentroid>::value,
                  "Provided a BoundType that does not inherit from StaticBoundWithCentroid<Derived>");

    KDNodeView(const IteratorPairType& indices_range, const BoundType& bound);

    KDNodeView(const IteratorPairType& indices_range, std::ptrdiff_t cut_axis_feature_index, const BoundType& bound);

    KDNodeView(const IteratorPairType&& indices_range, const BoundType& bound);

    KDNodeView(const IteratorPairType&& indices_range, std::ptrdiff_t cut_axis_feature_index, const BoundType& bound);

    KDNodeView(const KDNodeView&) = delete;

    constexpr auto begin() const;

    constexpr auto end() const;

    bool is_empty() const;

    std::size_t n_samples() const;

    bool is_leaf() const;

    bool has_parent() const;

    bool has_children() const;

    template <typename ReferenceSamplesIterator, typename QueryBuffer>
    auto select_sibling_node(const ReferenceSamplesIterator& reference_samples_range_first,
                             const ReferenceSamplesIterator& reference_samples_range_last,
                             std::size_t                     reference_n_features,
                             const QueryBuffer&              query_buffer) const
        -> std::enable_if_t<common::is_crtp_of<QueryBuffer, search::buffer::StaticBuffer>::value, NodePtr>;

    template <typename ReferenceSamplesIterator, typename QuerySamplesIterator>
    auto select_closest_child(const ReferenceSamplesIterator& reference_samples_range_first,
                              const ReferenceSamplesIterator& reference_samples_range_last,
                              std::size_t                     reference_n_features,
                              const QuerySamplesIterator&     query_features_range_first,
                              const QuerySamplesIterator&     query_features_range_last) const -> NodePtr;

    template <typename ReferenceSamplesIterator>
    void serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer,
                   const ReferenceSamplesIterator&             reference_samples_range_first,
                   const ReferenceSamplesIterator&             reference_samples_range_last,
                   std::size_t                                 reference_n_features) const;

    // A pair of iterators representing a window in the index array, referring to samples in the dataset.
    // This window can represent various ranges: empty, a 1 value range for pivot, or 1+ values range for a leaf node.
    IteratorPairType indices_range_;
    // The index of the feature dimension selected for cutting the dataset at this node. -1 means no cut (leaf node)
    std::ptrdiff_t cut_axis_feature_index_;
    // A 1D bounding box window that stores the actual dataset values referred to by the indices_range_.
    // The first value in this range represents the minimum value, while the second value represents the maximum value
    // within the dataset along the cut dimension for this node.
    BoundType bound_;
    // A child node representing the left partition of the dataset concerning the chosen cut dimension.
    // This child node may be empty if no further partitioning occurs.
    NodePtr left_;
    // A child node representing the right partition of the dataset concerning the chosen cut dimension.
    // This child node may be empty if no further partitioning occurs.
    NodePtr right_;
    // A weak pointer to the unique parent node of this node. It allows traversal up the tree hierarchy.
    std::weak_ptr<NodeType> parent_;

  private:
    bool is_left_child() const;

    bool is_right_child() const;
};

template <typename IteratorPairType, typename Bound>
KDNodeView(const IteratorPairType&, const Bound&)
    -> KDNodeView<typename std::iterator_traits<typename IteratorPairType::first_type>::value_type, Bound>;

template <typename IteratorPairType, typename Bound>
KDNodeView(const IteratorPairType&, std::ptrdiff_t, const Bound&)
    -> KDNodeView<typename std::iterator_traits<typename IteratorPairType::first_type>::value_type, Bound>;

template <typename IteratorPairType, typename Bound>
KDNodeView(IteratorPairType&&, const Bound&)
    -> KDNodeView<typename std::iterator_traits<typename IteratorPairType::first_type>::value_type, Bound>;

template <typename IteratorPairType, typename Bound>
KDNodeView(IteratorPairType&&, std::ptrdiff_t, const Bound&)
    -> KDNodeView<typename std::iterator_traits<typename IteratorPairType::first_type>::value_type, Bound>;

template <typename IndicesIterator, typename Bound>
KDNodeView<IndicesIterator, Bound>::KDNodeView(const IteratorPairType& indices_range, const BoundType& bound)
  : indices_range_{indices_range}
  , cut_axis_feature_index_{-1}
  , bound_{bound} {}

template <typename IndicesIterator, typename Bound>
KDNodeView<IndicesIterator, Bound>::KDNodeView(const IteratorPairType& indices_range,
                                               std::ptrdiff_t          cut_axis_feature_index,
                                               const BoundType&        bound)
  : indices_range_{indices_range}
  , cut_axis_feature_index_{cut_axis_feature_index}
  , bound_{bound} {}

template <typename IndicesIterator, typename Bound>
KDNodeView<IndicesIterator, Bound>::KDNodeView(const IteratorPairType&& indices_range, const BoundType& bound)
  : indices_range_{std::move(indices_range)}
  , cut_axis_feature_index_{-1}
  , bound_{bound} {}

template <typename IndicesIterator, typename Bound>
KDNodeView<IndicesIterator, Bound>::KDNodeView(const IteratorPairType&& indices_range,
                                               std::ptrdiff_t           cut_axis_feature_index,
                                               const BoundType&         bound)
  : indices_range_{std::move(indices_range)}
  , cut_axis_feature_index_{cut_axis_feature_index}
  , bound_{bound} {}

template <typename IndicesIterator, typename Bound>
constexpr auto KDNodeView<IndicesIterator, Bound>::begin() const {
    return indices_range_.first;
}

template <typename IndicesIterator, typename Bound>
constexpr auto KDNodeView<IndicesIterator, Bound>::end() const {
    return indices_range_.second;
}

template <typename IndicesIterator, typename Bound>
bool KDNodeView<IndicesIterator, Bound>::is_empty() const {
    return n_samples() == static_cast<std::size_t>(0);
}

template <typename IndicesIterator, typename Bound>
std::size_t KDNodeView<IndicesIterator, Bound>::n_samples() const {
    return std::distance(indices_range_.first, indices_range_.second);
}

template <typename IndicesIterator, typename Bound>
bool KDNodeView<IndicesIterator, Bound>::is_leaf() const {
    return !left_ && !right_;
}

template <typename IndicesIterator, typename Bound>
bool KDNodeView<IndicesIterator, Bound>::is_left_child() const {
    assert(has_parent());

    return this == parent_.lock()->left_.get();
}

template <typename IndicesIterator, typename Bound>
bool KDNodeView<IndicesIterator, Bound>::is_right_child() const {
    assert(has_parent());

    return this == parent_.lock()->right_.get();
}

template <typename IndicesIterator, typename Bound>
bool KDNodeView<IndicesIterator, Bound>::has_parent() const {
    return parent_.lock() != nullptr;
}

template <typename IndicesIterator, typename Bound>
bool KDNodeView<IndicesIterator, Bound>::has_children() const {
    return left_ && right_;
}

template <typename IndicesIterator, typename Bound>
template <typename ReferenceSamplesIterator, typename QueryBuffer>
auto KDNodeView<IndicesIterator, Bound>::select_sibling_node(
    const ReferenceSamplesIterator& reference_samples_range_first,
    const ReferenceSamplesIterator& reference_samples_range_last,
    std::size_t                     reference_n_features,
    const QueryBuffer&              query_buffer) const
    -> std::enable_if_t<common::is_crtp_of<QueryBuffer, search::buffer::StaticBuffer>::value, NodePtr> {
    static_assert(common::is_iterator<ReferenceSamplesIterator>::value, "ReferenceSamplesIterator is not an iterator");

    assert(has_parent());
    common::ignore_parameters(reference_samples_range_last);

    auto parent_node = parent_.lock();

    if (parent_node) {
        // get the pivot sample index in the dataset
        const auto pivot_index = parent_node->indices_range_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value =
            reference_samples_range_first[pivot_index * reference_n_features + parent_node->cut_axis_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = query_buffer.centroid_begin()[parent_node->cut_axis_feature_index_];

        const bool is_left_child_ret = is_left_child();
        // if the axiswise distance is equal to the current furthest nearest neighbor distance, there could be a
        // nearest neighbor to the other side of the hyperrectangle since the values that are equal to the pivot are
        // put to the right
        auto visit_sibling = [&]() {
            return is_left_child_ret ? common::abs(pivot_split_value - query_split_value) <=
                                           query_buffer.furthest_distance(parent_node->cut_axis_feature_index_)
                                     : common::abs(pivot_split_value - query_split_value) <
                                           query_buffer.furthest_distance(parent_node->cut_axis_feature_index_);
        };
        return query_buffer.remaining_capacity() || visit_sibling()
                   ? (is_left_child_ret ? parent_node->right_ : parent_node->left_)
                   : nullptr;
    }
    return nullptr;
}

template <typename IndicesIterator, typename Bound>
template <typename ReferenceSamplesIterator, typename QuerySamplesIterator>
auto KDNodeView<IndicesIterator, Bound>::select_closest_child(
    const ReferenceSamplesIterator& reference_samples_range_first,
    const ReferenceSamplesIterator& reference_samples_range_last,
    std::size_t                     reference_n_features,
    const QuerySamplesIterator&     query_features_range_first,
    const QuerySamplesIterator&     query_features_range_last) const -> NodePtr {
    static_assert(common::is_iterator<ReferenceSamplesIterator>::value, "ReferenceSamplesIterator is not an iterator");
    static_assert(common::is_iterator<QuerySamplesIterator>::value, "QuerySamplesIterator is not an iterator");

    common::ignore_parameters(reference_samples_range_last, query_features_range_last);

    // get the pivot sample index in the dataset
    const auto pivot_index = indices_range_.first[0];
    // get the split value according to the current split dimension
    const auto pivot_split_value =
        reference_samples_range_first[pivot_index * reference_n_features + cut_axis_feature_index_];
    // get the value of the query according to the split dimension
    const auto query_split_value = query_features_range_first[cut_axis_feature_index_];

    // traverse either the left or right child node depending on where the target sample is located relatively
    // to the cut value
    return query_split_value < pivot_split_value ? left_ : right_;
}

template <typename IndicesIterator, typename Bound>
template <typename ReferenceSamplesIterator>
void KDNodeView<IndicesIterator, Bound>::serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer,
                                                   const ReferenceSamplesIterator& reference_samples_range_first,
                                                   const ReferenceSamplesIterator& reference_samples_range_last,
                                                   std::size_t                     reference_n_features) const {
    static_assert(common::is_iterator<ReferenceSamplesIterator>::value, "ReferenceSamplesIterator is not an iterator");

    const auto [indices_range_first, indices_range_last] = indices_range_;

    common::ignore_parameters(reference_samples_range_last);

    writer.StartArray();
    for (auto subrange_index_it = indices_range_first; subrange_index_it != indices_range_last; ++subrange_index_it) {
        writer.StartArray();
        for (std::size_t feature_index = 0; feature_index < reference_n_features; ++feature_index) {
            if constexpr (std::is_integral_v<DataType>) {
                writer.Int64(reference_samples_range_first[*subrange_index_it * reference_n_features + feature_index]);

            } else if constexpr (std::is_floating_point_v<DataType>) {
                writer.Double(reference_samples_range_first[*subrange_index_it * reference_n_features + feature_index]);
            }
        }
        writer.EndArray();
    }
    writer.EndArray();
}

}  // namespace ffcl::datastruct