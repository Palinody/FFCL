#pragma once

#include "cpp_clustering/common/Utils.hpp"
#include "cpp_clustering/containers/kdtree/KDTreeUtils.hpp"
#include "cpp_clustering/math/random/Distributions.hpp"

#include <sys/types.h>  // ssize_t
#include <algorithm>
#include <array>
#include <memory>

#include "rapidjson/writer.h"

namespace cpp_clustering::containers {

template <typename Iterator>
using IteratorPairType = std::pair<Iterator, Iterator>;

template <typename Iterator>
using DataType = typename Iterator::value_type;

template <typename Iterator>
using BoundingBox1DType = std::pair<DataType<Iterator>, DataType<Iterator>>;

template <typename Iterator>
using BoundingBoxKDType = std::vector<BoundingBox1DType<Iterator>>;

template <typename Iterator>
struct KDNode {
    KDNode(Iterator                           samples_first,
           Iterator                           samples_last,
           std::size_t                        n_features,
           const BoundingBoxKDType<Iterator>& kd_bounding_box);

    KDNode(Iterator                           samples_first,
           Iterator                           samples_last,
           std::size_t                        n_features,
           ssize_t                            cut_feature_index,
           const BoundingBoxKDType<Iterator>& kd_bounding_box);

    KDNode(const KDNode&) = delete;

    bool is_empty() const;

    bool is_leaf() const;

    void serialize_kdnode(rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    // might contain [0, bucket_size] samples if the node is leaf, else only 1
    IteratorPairType<Iterator> samples_iterator_pair_;
    std::size_t                n_features_;
    ssize_t                    cut_feature_index_;
    // bounding box w.r.t. the chosen feature index. No bounding box for leaf nodes
    // BoundingBox1DType                 bounding_box_1d_;
    // bounding box hyper rectangle (w.r.t. each dimension)
    BoundingBoxKDType<Iterator>       kd_bounding_box_;
    std::shared_ptr<KDNode<Iterator>> left_;
    std::shared_ptr<KDNode<Iterator>> right_;
};

template <typename Iterator>
KDNode<Iterator>::KDNode(Iterator                           samples_first,
                         Iterator                           samples_last,
                         std::size_t                        n_features,
                         const BoundingBoxKDType<Iterator>& kd_bounding_box)
  : samples_iterator_pair_{std::make_pair(samples_first, samples_last)}
  , n_features_{n_features}
  , cut_feature_index_{-1}
  , kd_bounding_box_{kd_bounding_box} {}

template <typename Iterator>
KDNode<Iterator>::KDNode(Iterator                           samples_first,
                         Iterator                           samples_last,
                         std::size_t                        n_features,
                         ssize_t                            cut_feature_index,
                         const BoundingBoxKDType<Iterator>& kd_bounding_box)
  : samples_iterator_pair_{kdtree::utils::quickselect_median_range(samples_first,
                                                                   samples_last,
                                                                   n_features,
                                                                   cut_feature_index)}
  , n_features_{n_features}
  , cut_feature_index_{cut_feature_index}
  , kd_bounding_box_{kd_bounding_box} {}

template <typename Iterator>
bool KDNode<Iterator>::is_empty() const {
    return std::distance(samples_iterator_pair_.first, samples_iterator_pair_.second) == static_cast<std::ptrdiff_t>(0);
}

template <typename Iterator>
bool KDNode<Iterator>::is_leaf() const {
    return cut_feature_index_ == -1;
}

template <typename Iterator>
void KDNode<Iterator>::serialize_kdnode(rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
    assert(is_leaf());

    writer.StartArray();
    // upper-left and lower-right (with sentinel) iterators
    const auto [range_first, range_last] = samples_iterator_pair_;

    const std::size_t n_samples = common::utils::get_n_samples(range_first, range_last, n_features_);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // sample (feature vector) array
        writer.StartArray();
        for (std::size_t feature_index = 0; feature_index < n_features_; ++feature_index) {
            writer.Double(samples_iterator_pair_.first[sample_index * n_features_ + feature_index]);
        }
        writer.EndArray();
    }
    writer.EndArray();
}

}  // namespace cpp_clustering::containers