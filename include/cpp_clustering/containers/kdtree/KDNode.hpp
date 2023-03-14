#pragma once

#include "cpp_clustering/common/Utils.hpp"
#include "cpp_clustering/containers/kdtree/KDTreeUtils.hpp"
#include "cpp_clustering/math/random/Distributions.hpp"

#include <sys/types.h>  // ssize_t
#include <algorithm>
#include <array>
#include <memory>

namespace cpp_clustering::containers {

template <typename Iterator>
using SamplesRangeType = std::pair<Iterator, Iterator>;

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

    SamplesRangeType<Iterator> samples_;
    std::size_t                n_features_;
    ssize_t                    cut_feature_index_;
    // bounding box w.r.t. the chosen feature index. No bounding box for leaf nodes
    // BoundingBox1DType                 bounding_box_1d_;
    BoundingBoxKDType<Iterator>       kd_bounding_box_;
    std::shared_ptr<KDNode<Iterator>> left_;
    std::shared_ptr<KDNode<Iterator>> right_;
};

template <typename Iterator>
KDNode<Iterator>::KDNode(Iterator                           samples_first,
                         Iterator                           samples_last,
                         std::size_t                        n_features,
                         const BoundingBoxKDType<Iterator>& kd_bounding_box)
  : samples_{std::make_pair(samples_first, samples_last)}
  , n_features_{n_features}
  , cut_feature_index_{-1}
  , kd_bounding_box_{kd_bounding_box} {}

template <typename Iterator>
KDNode<Iterator>::KDNode(Iterator                           samples_first,
                         Iterator                           samples_last,
                         std::size_t                        n_features,
                         ssize_t                            cut_feature_index,
                         const BoundingBoxKDType<Iterator>& kd_bounding_box)
  : samples_{kdtree::utils::quickselect_median_range(samples_first, samples_last, n_features, cut_feature_index)}
  , n_features_{n_features}
  , cut_feature_index_{cut_feature_index}
  , kd_bounding_box_{kd_bounding_box} {}

template <typename Iterator>
bool KDNode<Iterator>::is_empty() const {
    return samples_.first == samples_.second;
}

}  // namespace cpp_clustering::containers