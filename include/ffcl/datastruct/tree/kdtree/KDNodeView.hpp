#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/Interval.hpp"

#include "ffcl/search/buffer/StaticBase.hpp"
#include "ffcl/search/count/StaticBase.hpp"

#include "ffcl/datastruct/bounds/segment_representation/MinAndMax.hpp"

#include <sys/types.h>  // ssize_t
#include <algorithm>
#include <array>
#include <memory>

#include "rapidjson/writer.h"

namespace ffcl::datastruct {

template <typename IndicesIterator, typename Data>
struct KDNodeView {
    using IndicesIteratorType = IndicesIterator;

    static_assert(common::is_iterator<IndicesIteratorType>::value, "IndicesIteratorType is not an iterator");

    using IndexType = typename std::iterator_traits<IndicesIterator>::value_type;
    using DataType  = Data;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    using NodeType = KDNodeView<IndicesIterator, DataType>;
    using NodePtr  = std::shared_ptr<NodeType>;

    using IteratorPairType = std::pair<IndicesIterator, IndicesIterator>;
    using IntervalType     = Interval<DataType>;

    KDNodeView(const IteratorPairType& indices_range, const IntervalType& axis_interval);

    KDNodeView(const IteratorPairType& indices_range,
               ssize_t                 cut_axis_feature_index,
               const IntervalType&     axis_interval);

    KDNodeView(const IteratorPairType&& indices_range, const IntervalType& axis_interval);

    KDNodeView(const IteratorPairType&& indices_range,
               ssize_t                  cut_axis_feature_index,
               const IntervalType&      axis_interval);

    KDNodeView(const KDNodeView&) = delete;

    bool is_empty() const;

    std::size_t n_samples() const;

    bool is_leaf() const;

    bool has_parent() const;

    bool has_children() const;

    template <typename OtherSamplesIterator, typename Buffer>
    auto select_sibling_node(const OtherSamplesIterator& samples_range_first,
                             const OtherSamplesIterator& samples_range_last,
                             std::size_t                 n_features,
                             const Buffer&               buffer) const
        -> std::enable_if_t<common::is_crtp_of<Buffer, search::buffer::StaticBase>::value, NodePtr>;

    template <typename OtherSamplesIterator>
    auto step_down(const OtherSamplesIterator& samples_range_first,
                   const OtherSamplesIterator& samples_range_last,
                   std::size_t                 n_features,
                   const OtherSamplesIterator& query_features_range_first,
                   const OtherSamplesIterator& query_features_range_last) const -> NodePtr;

    template <typename OtherSamplesIterator>
    void serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer,
                   const OtherSamplesIterator&                 samples_range_first,
                   const OtherSamplesIterator&                 samples_range_last,
                   std::size_t                                 n_features) const;

    // A pair of iterators representing a window in the index array, referring to samples in the dataset.
    // This window can represent various ranges: empty, a 1 value range for pivot, or 1+ values range for a leaf node.
    IteratorPairType indices_range_;
    // The index of the feature dimension selected for cutting the dataset at this node. -1 means no cut (leaf node)
    ssize_t cut_axis_feature_index_;
    // A 1D bounding box window that stores the actual dataset values referred to by the indices_range_.
    // The first value in this range represents the minimum value, while the second value represents the maximum value
    // within the dataset along the cut dimension for this node.
    IntervalType axis_interval_;
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

    auto get_sibling_node() const -> NodePtr;
};

template <typename IteratorPairType, typename DataType>
KDNodeView(const IteratorPairType&, const bounds::segment_representation::MinAndMax<DataType>&)
    -> KDNodeView<typename std::iterator_traits<typename IteratorPairType::first_type>::value_type,
                  typename bounds::segment_representation::MinAndMax<DataType>::ValueType>;

template <typename IteratorPairType, typename DataType>
KDNodeView(const IteratorPairType&, ssize_t, const bounds::segment_representation::MinAndMax<DataType>&)
    -> KDNodeView<typename std::iterator_traits<typename IteratorPairType::first_type>::value_type,
                  typename bounds::segment_representation::MinAndMax<DataType>::ValueType>;

template <typename IteratorPairType, typename DataType>
KDNodeView(IteratorPairType&&, const bounds::segment_representation::MinAndMax<DataType>&)
    -> KDNodeView<typename std::iterator_traits<typename IteratorPairType::first_type>::value_type,
                  typename bounds::segment_representation::MinAndMax<DataType>::ValueType>;

template <typename IteratorPairType, typename DataType>
KDNodeView(IteratorPairType&&, ssize_t, const bounds::segment_representation::MinAndMax<DataType>&)
    -> KDNodeView<typename std::iterator_traits<typename IteratorPairType::first_type>::value_type,
                  typename bounds::segment_representation::MinAndMax<DataType>::ValueType>;

template <typename IndicesIterator, typename Data>
KDNodeView<IndicesIterator, Data>::KDNodeView(const IteratorPairType& indices_range, const IntervalType& axis_interval)
  : indices_range_{indices_range}
  , cut_axis_feature_index_{-1}
  , axis_interval_{axis_interval} {}

template <typename IndicesIterator, typename Data>
KDNodeView<IndicesIterator, Data>::KDNodeView(const IteratorPairType& indices_range,
                                              ssize_t                 cut_axis_feature_index,
                                              const IntervalType&     axis_interval)
  : indices_range_{indices_range}
  , cut_axis_feature_index_{cut_axis_feature_index}
  , axis_interval_{axis_interval} {}

template <typename IndicesIterator, typename Data>
KDNodeView<IndicesIterator, Data>::KDNodeView(const IteratorPairType&& indices_range, const IntervalType& axis_interval)
  : indices_range_{std::move(indices_range)}
  , cut_axis_feature_index_{-1}
  , axis_interval_{axis_interval} {}

template <typename IndicesIterator, typename Data>
KDNodeView<IndicesIterator, Data>::KDNodeView(const IteratorPairType&& indices_range,
                                              ssize_t                  cut_axis_feature_index,
                                              const IntervalType&      axis_interval)
  : indices_range_{std::move(indices_range)}
  , cut_axis_feature_index_{cut_axis_feature_index}
  , axis_interval_{axis_interval} {}

template <typename IndicesIterator, typename Data>
bool KDNodeView<IndicesIterator, Data>::is_empty() const {
    return n_samples() == static_cast<std::size_t>(0);
}

template <typename IndicesIterator, typename Data>
std::size_t KDNodeView<IndicesIterator, Data>::n_samples() const {
    return std::distance(indices_range_.first, indices_range_.second);
}

template <typename IndicesIterator, typename Data>
bool KDNodeView<IndicesIterator, Data>::is_leaf() const {
    return !left_ && !right_;
}

template <typename IndicesIterator, typename Data>
bool KDNodeView<IndicesIterator, Data>::is_left_child() const {
    assert(has_parent());

    return this == parent_.lock()->left_.get();
}

template <typename IndicesIterator, typename Data>
bool KDNodeView<IndicesIterator, Data>::is_right_child() const {
    assert(has_parent());

    return this == parent_.lock()->right_.get();
}

template <typename IndicesIterator, typename Data>
bool KDNodeView<IndicesIterator, Data>::has_parent() const {
    return parent_.lock() != nullptr;
}

template <typename IndicesIterator, typename Data>
bool KDNodeView<IndicesIterator, Data>::has_children() const {
    return left_ && right_;
}

template <typename IndicesIterator, typename Data>
template <typename OtherSamplesIterator, typename Buffer>
auto KDNodeView<IndicesIterator, Data>::select_sibling_node(const OtherSamplesIterator& samples_range_first,
                                                            const OtherSamplesIterator& samples_range_last,
                                                            std::size_t                 n_features,
                                                            const Buffer&               buffer) const
    -> std::enable_if_t<common::is_crtp_of<Buffer, search::buffer::StaticBase>::value, NodePtr> {
    static_assert(common::is_iterator<OtherSamplesIterator>::value, "OtherSamplesIterator is not an iterator");

    assert(has_parent());
    common::ignore_parameters(samples_range_last);

    auto parent_node = parent_.lock();

    if (parent_node) {
        // get the pivot sample index in the dataset
        const auto pivot_index = parent_node->indices_range_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value =
            samples_range_first[pivot_index * n_features + parent_node->cut_axis_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = buffer.centroid_begin()[parent_node->cut_axis_feature_index_];

        const bool is_left_child_ret = is_left_child();
        // if the axiswise distance is equal to the current furthest nearest neighbor distance, there could be a
        // nearest neighbor to the other side of the hyperrectangle since the values that are equal to the pivot are
        // put to the right
        const bool visit_sibling = is_left_child_ret ? common::abs(pivot_split_value - query_split_value) <=
                                                           buffer.upper_bound(parent_node->cut_axis_feature_index_)
                                                     : common::abs(pivot_split_value - query_split_value) <
                                                           buffer.upper_bound(parent_node->cut_axis_feature_index_);

        return buffer.n_free_slots() || visit_sibling ? (is_left_child_ret ? parent_node->right_ : parent_node->left_)
                                                      : nullptr;
    }
    return nullptr;
}

template <typename IndicesIterator, typename Data>
auto KDNodeView<IndicesIterator, Data>::get_sibling_node() const -> NodePtr {
    assert(has_parent());

    auto parent_shared_ptr = parent_.lock();

    if (this == parent_shared_ptr->left_.get()) {
        return parent_shared_ptr->right_;

    } else if (this == parent_shared_ptr->right_.get()) {
        return parent_shared_ptr->left_;
    }
    return nullptr;
}

template <typename IndicesIterator, typename Data>
template <typename OtherSamplesIterator>
auto KDNodeView<IndicesIterator, Data>::step_down(const OtherSamplesIterator& samples_range_first,
                                                  const OtherSamplesIterator& samples_range_last,
                                                  std::size_t                 n_features,
                                                  const OtherSamplesIterator& query_features_range_first,
                                                  const OtherSamplesIterator& query_features_range_last) const
    -> NodePtr {
    static_assert(common::is_iterator<OtherSamplesIterator>::value, "OtherSamplesIterator is not an iterator");

    common::ignore_parameters(samples_range_last, query_features_range_last);

    // get the pivot sample index in the dataset
    const auto pivot_index = indices_range_.first[0];
    // get the split value according to the current split dimension
    const auto pivot_split_value = samples_range_first[pivot_index * n_features + cut_axis_feature_index_];
    // get the value of the query according to the split dimension
    const auto query_split_value = query_features_range_first[cut_axis_feature_index_];

    // traverse either the left or right child node depending on where the target sample is located relatively
    // to the cut value
    return query_split_value < pivot_split_value ? left_ : right_;
}

template <typename IndicesIterator, typename Data>
template <typename OtherSamplesIterator>
void KDNodeView<IndicesIterator, Data>::serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer,
                                                  const OtherSamplesIterator&                 samples_range_first,
                                                  const OtherSamplesIterator&                 samples_range_last,
                                                  std::size_t                                 n_features) const {
    static_assert(common::is_iterator<OtherSamplesIterator>::value, "OtherSamplesIterator is not an iterator");

    writer.StartArray();
    const auto [indices_range_first, indices_range_last] = indices_range_;

    common::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // sample (feature vector) array
        writer.StartArray();
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            if constexpr (std::is_integral_v<DataType>) {
                writer.Int64(samples_range_first[indices_range_first[sample_index] * n_features + feature_index]);

            } else if constexpr (std::is_floating_point_v<DataType>) {
                writer.Double(samples_range_first[indices_range_first[sample_index] * n_features + feature_index]);
            }
        }
        writer.EndArray();
    }
    writer.EndArray();
}

}  // namespace ffcl::datastruct