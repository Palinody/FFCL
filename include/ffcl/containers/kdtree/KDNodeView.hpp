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

template <typename Iterator>
struct KDNodeView {
    KDNodeView(bbox::IteratorPairType<Iterator>      iterator_pair,
               std::size_t                           n_features,
               const bbox::HyperRangeType<Iterator>& kd_bounding_box);

    KDNodeView(bbox::IteratorPairType<Iterator>      iterator_pair,
               std::size_t                           n_features,
               ssize_t                               cut_feature_index,
               const bbox::HyperRangeType<Iterator>& kd_bounding_box);

    KDNodeView(const KDNodeView&) = delete;

    bool is_empty() const;

    std::size_t n_samples() const;

    bool is_leaf() const;

    bool is_left_child() const;

    bool is_right_child() const;

    bool has_parent() const;

    bool has_children() const;

    std::shared_ptr<KDNodeView<Iterator>> get_sibling_node() const;

    void serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    // might contain [0, bucket_size] samples if the node is leaf, else only 1
    bbox::IteratorPairType<Iterator> samples_iterator_pair_;
    std::size_t                      n_features_;
    ssize_t                          cut_feature_index_;
    // bounding box hyper rectangle (w.r.t. each dimension)
    bbox::HyperRangeType<Iterator>        kd_bounding_box_;
    std::shared_ptr<KDNodeView<Iterator>> left_;
    std::shared_ptr<KDNodeView<Iterator>> right_;
    std::weak_ptr<KDNodeView<Iterator>>   parent_;
};

template <typename Iterator>
KDNodeView<Iterator>::KDNodeView(bbox::IteratorPairType<Iterator>      iterator_pair,
                                 std::size_t                           n_features,
                                 const bbox::HyperRangeType<Iterator>& kd_bounding_box)
  : samples_iterator_pair_{iterator_pair}
  , n_features_{n_features}
  , cut_feature_index_{-1}
  , kd_bounding_box_{kd_bounding_box} {}

template <typename Iterator>
KDNodeView<Iterator>::KDNodeView(bbox::IteratorPairType<Iterator>      iterator_pair,
                                 std::size_t                           n_features,
                                 ssize_t                               cut_feature_index,
                                 const bbox::HyperRangeType<Iterator>& kd_bounding_box)
  : samples_iterator_pair_{iterator_pair}
  , n_features_{n_features}
  , cut_feature_index_{cut_feature_index}
  , kd_bounding_box_{kd_bounding_box} {}

template <typename Iterator>
bool KDNodeView<Iterator>::is_empty() const {
    return std::distance(samples_iterator_pair_.first, samples_iterator_pair_.second) == static_cast<std::ptrdiff_t>(0);
}

template <typename Iterator>
std::size_t KDNodeView<Iterator>::n_samples() const {
    return common::utils::get_n_samples(samples_iterator_pair_.first, samples_iterator_pair_.second, n_features_);
}

template <typename Iterator>
bool KDNodeView<Iterator>::is_leaf() const {
    return cut_feature_index_ == -1;
}

template <typename Iterator>
bool KDNodeView<Iterator>::is_left_child() const {
    if (has_parent()) {
        return this == parent_.lock()->left_.get();
    }
    return false;
}

template <typename Iterator>
bool KDNodeView<Iterator>::is_right_child() const {
    if (has_parent()) {
        return this == parent_.lock()->right_.get();
    }
    return false;
}

template <typename Iterator>
bool KDNodeView<Iterator>::has_parent() const {
    return parent_.lock() != nullptr;
}

template <typename SamplesIterator>
bool KDNodeView<SamplesIterator>::has_children() const {
    return left_ != nullptr && right_ != nullptr;
}

template <typename Iterator>
std::shared_ptr<KDNodeView<Iterator>> KDNodeView<Iterator>::get_sibling_node() const {
    if (has_parent()) {
        auto parent_shared_ptr = parent_.lock();

        if (this == parent_shared_ptr->left_.get()) {
            return parent_shared_ptr->right_;

        } else {
            return parent_shared_ptr->left_;
        }
    }
    return nullptr;
}

template <typename Iterator>
void KDNodeView<Iterator>::serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
    using DataType = DataType<Iterator>;

    static_assert(std::is_floating_point_v<DataType> || std::is_integral_v<DataType>,
                  "Unsupported type during kdnode serialization");

    writer.StartArray();
    // upper-left and lower-right (with sentinel) iterators
    const auto [samples_first, samples_last] = samples_iterator_pair_;

    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features_);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // sample (feature vector) array
        writer.StartArray();
        for (std::size_t feature_index = 0; feature_index < n_features_; ++feature_index) {
            if constexpr (std::is_integral_v<DataType>) {
                writer.Int64(samples_first[sample_index * n_features_ + feature_index]);

            } else if constexpr (std::is_floating_point_v<DataType>) {
                writer.Double(samples_first[sample_index * n_features_ + feature_index]);
            }
        }
        writer.EndArray();
    }
    writer.EndArray();
}

}  // namespace ffcl::containers