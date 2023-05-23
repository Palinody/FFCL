#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDTreeAlgorithms.hpp"
#include "ffcl/math/random/Distributions.hpp"

#include <sys/types.h>  // ssize_t
#include <algorithm>
#include <array>
#include <memory>

#include "rapidjson/writer.h"

namespace ffcl::containers {

template <typename IndicesIterator, typename SamplesIterator>
struct KDNodeIndexView {
    KDNodeIndexView(IteratorPairType<IndicesIterator>         indices_iterator_pair,
                    IteratorPairType<SamplesIterator>         samples_iterator_pair,
                    std::size_t                               n_features,
                    const BoundingBoxKDType<SamplesIterator>& kd_bounding_box);

    KDNodeIndexView(IteratorPairType<IndicesIterator>         indices_iterator_pair,
                    IteratorPairType<SamplesIterator>         samples_iterator_pair,
                    std::size_t                               n_features,
                    ssize_t                                   cut_feature_index,
                    const BoundingBoxKDType<SamplesIterator>& kd_bounding_box);

    KDNodeIndexView(const KDNodeIndexView&) = delete;

    bool is_empty() const;

    std::size_t n_samples() const;

    bool is_leaf() const;

    bool is_left_child() const;

    bool is_right_child() const;

    bool has_parent() const;

    std::shared_ptr<KDNodeIndexView<IndicesIterator, SamplesIterator>> get_sibling_node() const;

    void serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    IteratorPairType<IndicesIterator> indices_iterator_pair_;
    // might contain [0, bucket_size] samples if the node is leaf, else only 1
    IteratorPairType<SamplesIterator> samples_iterator_pair_;
    std::size_t                       n_features_;
    ssize_t                           cut_feature_index_;
    // bounding box hyper rectangle (w.r.t. each dimension)
    BoundingBoxKDType<SamplesIterator>                                 kd_bounding_box_;
    std::shared_ptr<KDNodeIndexView<IndicesIterator, SamplesIterator>> left_;
    std::shared_ptr<KDNodeIndexView<IndicesIterator, SamplesIterator>> right_;
    std::weak_ptr<KDNodeIndexView<IndicesIterator, SamplesIterator>>   parent_;
};

template <typename IndicesIterator, typename SamplesIterator>
KDNodeIndexView<IndicesIterator, SamplesIterator>::KDNodeIndexView(
    IteratorPairType<IndicesIterator>         indices_iterator_pair,
    IteratorPairType<SamplesIterator>         samples_iterator_pair,
    std::size_t                               n_features,
    const BoundingBoxKDType<SamplesIterator>& kd_bounding_box)
  : indices_iterator_pair_{indices_iterator_pair}
  , samples_iterator_pair_{samples_iterator_pair}
  , n_features_{n_features}
  , cut_feature_index_{-1}
  , kd_bounding_box_{kd_bounding_box} {}

template <typename IndicesIterator, typename SamplesIterator>
KDNodeIndexView<IndicesIterator, SamplesIterator>::KDNodeIndexView(
    IteratorPairType<IndicesIterator>         indices_iterator_pair,
    IteratorPairType<SamplesIterator>         samples_iterator_pair,
    std::size_t                               n_features,
    ssize_t                                   cut_feature_index,
    const BoundingBoxKDType<SamplesIterator>& kd_bounding_box)
  : indices_iterator_pair_{indices_iterator_pair}
  , samples_iterator_pair_{samples_iterator_pair}
  , n_features_{n_features}
  , cut_feature_index_{cut_feature_index}
  , kd_bounding_box_{kd_bounding_box} {}

template <typename IndicesIterator, typename SamplesIterator>
bool KDNodeIndexView<IndicesIterator, SamplesIterator>::is_empty() const {
    return n_samples() == static_cast<std::ptrdiff_t>(0);
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t KDNodeIndexView<IndicesIterator, SamplesIterator>::n_samples() const {
    return std::distance(indices_iterator_pair_.first, indices_iterator_pair_.second);
}

template <typename IndicesIterator, typename SamplesIterator>
bool KDNodeIndexView<IndicesIterator, SamplesIterator>::is_leaf() const {
    return cut_feature_index_ == -1;
}

template <typename IndicesIterator, typename SamplesIterator>
bool KDNodeIndexView<IndicesIterator, SamplesIterator>::has_parent() const {
    return parent_.lock() != nullptr;
}

template <typename IndicesIterator, typename SamplesIterator>
bool KDNodeIndexView<IndicesIterator, SamplesIterator>::is_left_child() const {
    if (has_parent()) {
        return this == parent_.lock()->left_.get();
    }
    return false;
}

template <typename IndicesIterator, typename SamplesIterator>
bool KDNodeIndexView<IndicesIterator, SamplesIterator>::is_right_child() const {
    if (has_parent()) {
        return this == parent_.lock()->right_.get();
    }
    return false;
}

template <typename IndicesIterator, typename SamplesIterator>
std::shared_ptr<KDNodeIndexView<IndicesIterator, SamplesIterator>>
KDNodeIndexView<IndicesIterator, SamplesIterator>::get_sibling_node() const {
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
void KDNodeIndexView<IndicesIterator, SamplesIterator>::serialize(
    rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
    using DataType = DataType<SamplesIterator>;

    static_assert(std::is_floating_point_v<DataType> || std::is_integral_v<DataType>,
                  "Unsupported type during kdnode serialization");

    writer.StartArray();
    const auto [indices_range_first, indices_range_last] = indices_iterator_pair_;
    // upper-left and lower-right (with sentinel) iterators
    const auto [range_first, range_last] = samples_iterator_pair_;

    common::utils::ignore_parameters(range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // sample (feature vector) array
        writer.StartArray();
        for (std::size_t feature_index = 0; feature_index < n_features_; ++feature_index) {
            if constexpr (std::is_integral_v<DataType>) {
                writer.Int64(range_first[indices_range_first[sample_index] * n_features_ + feature_index]);

            } else if constexpr (std::is_floating_point_v<DataType>) {
                writer.Double(range_first[indices_range_first[sample_index] * n_features_ + feature_index]);
            }
        }
        writer.EndArray();
    }
    writer.EndArray();
}

}  // namespace ffcl::containers