#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/kdtree/KDTreeAlgorithms.hpp"

#include <sys/types.h>  // ssize_t
#include <algorithm>
#include <array>
#include <memory>

#include "rapidjson/writer.h"

namespace ffcl::datastruct {

template <typename IndicesIterator, typename SamplesIterator>
struct KDNodeView {
    static_assert(common::is_iterator<IndicesIterator>::value, "IndicesIterator is not an iterator");
    static_assert(common::is_iterator<SamplesIterator>::value, "SamplesIterator is not an iterator");

    using KDNodeViewType = KDNodeView<IndicesIterator, SamplesIterator>;
    using KDNodeViewPtr  = std::shared_ptr<KDNodeViewType>;

    KDNodeView(const bbox::IteratorPairType<IndicesIterator>& indices_range,
               const bbox::RangeType<SamplesIterator>&        kd_bounding_box);

    KDNodeView(const bbox::IteratorPairType<IndicesIterator>& indices_range,
               ssize_t                                        cut_feature_index,
               const bbox::RangeType<SamplesIterator>&        kd_bounding_box);

    KDNodeView(const KDNodeView&) = delete;

    bool is_empty() const;

    std::size_t n_samples() const;

    bool is_leaf() const;

    bool is_left_child() const;

    bool is_right_child() const;

    bool has_parent() const;

    bool has_children() const;

    KDNodeViewPtr get_sibling_node() const;

    void serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer,
                   const SamplesIterator&                      samples_range_first,
                   const SamplesIterator&                      samples_range_last,
                   std::size_t                                 n_features) const;

    // A pair of iterators representing a window in the index array, referring to samples in the dataset.
    // This window can represent various ranges: empty, a 1 value range for pivot, or 1+ values range for a leaf node.
    bbox::IteratorPairType<IndicesIterator> indices_range_;
    // The index of the feature dimension selected for cutting the dataset at this node. -1 means no cut (leaf node)
    ssize_t cut_feature_index_;
    // A 1D bounding box window that stores the actual dataset values referred to by the indices_range_.
    // The first value in this range represents the minimum value, while the second value represents the maximum value
    // within the dataset along the cut dimension for this node.
    bbox::RangeType<SamplesIterator> cut_feature_range_;
    // A child node representing the left partition of the dataset concerning the chosen cut dimension.
    // This child node may be empty if no further partitioning occurs.
    KDNodeViewPtr left_;
    // A child node representing the right partition of the dataset concerning the chosen cut dimension.
    // This child node may be empty if no further partitioning occurs.
    KDNodeViewPtr right_;
    // A weak pointer to the unique parent node of this node. It allows traversal up the tree hierarchy.
    std::weak_ptr<KDNodeViewType> parent_;
};

template <typename IndicesIterator, typename SamplesIterator>
KDNodeView<IndicesIterator, SamplesIterator>::KDNodeView(const bbox::IteratorPairType<IndicesIterator>& indices_range,
                                                         const bbox::RangeType<SamplesIterator>&        kd_bounding_box)
  : indices_range_{indices_range}
  , cut_feature_index_{-1}
  , cut_feature_range_{kd_bounding_box} {}

template <typename IndicesIterator, typename SamplesIterator>
KDNodeView<IndicesIterator, SamplesIterator>::KDNodeView(const bbox::IteratorPairType<IndicesIterator>& indices_range,
                                                         ssize_t                                 cut_feature_index,
                                                         const bbox::RangeType<SamplesIterator>& kd_bounding_box)
  : indices_range_{indices_range}
  , cut_feature_index_{cut_feature_index}
  , cut_feature_range_{kd_bounding_box} {}

template <typename IndicesIterator, typename SamplesIterator>
bool KDNodeView<IndicesIterator, SamplesIterator>::is_empty() const {
    return n_samples() == static_cast<std::size_t>(0);
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t KDNodeView<IndicesIterator, SamplesIterator>::n_samples() const {
    return std::distance(indices_range_.first, indices_range_.second);
}

template <typename IndicesIterator, typename SamplesIterator>
bool KDNodeView<IndicesIterator, SamplesIterator>::is_leaf() const {
    return left_ == nullptr && right_ == nullptr;
}

template <typename IndicesIterator, typename SamplesIterator>
bool KDNodeView<IndicesIterator, SamplesIterator>::is_left_child() const {
    if (has_parent()) {
        return this == parent_.lock()->left_.get();
    }
    return false;
}

template <typename IndicesIterator, typename SamplesIterator>
bool KDNodeView<IndicesIterator, SamplesIterator>::is_right_child() const {
    if (has_parent()) {
        return this == parent_.lock()->right_.get();
    }
    return false;
}

template <typename IndicesIterator, typename SamplesIterator>
bool KDNodeView<IndicesIterator, SamplesIterator>::has_parent() const {
    return parent_.lock() != nullptr;
}

template <typename IndicesIterator, typename SamplesIterator>
bool KDNodeView<IndicesIterator, SamplesIterator>::has_children() const {
    return left_ != nullptr && right_ != nullptr;
}

template <typename IndicesIterator, typename SamplesIterator>
typename KDNodeView<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDNodeView<IndicesIterator, SamplesIterator>::get_sibling_node() const {
    if (has_parent()) {
        auto parent_shared_ptr = parent_.lock();

        if (this == parent_shared_ptr->left_.get()) {
            return parent_shared_ptr->right_;

        } else if (this == parent_shared_ptr->right_.get()) {
            return parent_shared_ptr->left_;
        }
    }
    return nullptr;
}

template <typename IndicesIterator, typename SamplesIterator>
void KDNodeView<IndicesIterator, SamplesIterator>::serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer,
                                                             const SamplesIterator& samples_range_first,
                                                             const SamplesIterator& samples_range_last,
                                                             std::size_t            n_features) const {
    using DataType = bbox::DataType<SamplesIterator>;

    static_assert(std::is_floating_point_v<DataType> || std::is_integral_v<DataType>,
                  "Unsupported type during kdnode serialization");

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