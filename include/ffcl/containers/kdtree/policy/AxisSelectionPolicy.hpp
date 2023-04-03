#pragma once

#include "ffcl/containers/kdtree/KDTreeUtils.hpp"

#include "ffcl/containers/kdtree/policy/Utils.hpp"

namespace kdtree::policy {

template <typename RandomAccessIterator>
struct AxisSelectionPolicy {
    AxisSelectionPolicy() = default;

    virtual std::size_t operator()(const IteratorPairType<RandomAccessIterator>& iterator_pair,
                                   std::size_t                                   n_features,
                                   ssize_t                                       depth,
                                   BoundingBoxKDType<RandomAccessIterator>&      kd_bounding_box) const = 0;

  private:
    AxisSelectionPolicy(const AxisSelectionPolicy&) = delete;

    AxisSelectionPolicy& operator=(const AxisSelectionPolicy&) = delete;
};

template <typename RandomAccessIterator>
struct CycleThroughAxesBuild : public AxisSelectionPolicy<RandomAccessIterator> {
    std::size_t operator()(const IteratorPairType<RandomAccessIterator>& iterator_pair,
                           std::size_t                                   n_features,
                           ssize_t                                       depth,
                           BoundingBoxKDType<RandomAccessIterator>&      kd_bounding_box) const override;
};

template <typename RandomAccessIterator>
struct HighestVarianceBuild : public AxisSelectionPolicy<RandomAccessIterator> {
    HighestVarianceBuild& sampling_proportion(double sampling_proportion) {
        sampling_proportion_ = sampling_proportion;
        return *this;
    }

    std::size_t operator()(const IteratorPairType<RandomAccessIterator>& iterator_pair,
                           std::size_t                                   n_features,
                           ssize_t                                       depth,
                           BoundingBoxKDType<RandomAccessIterator>&      kd_bounding_box) const override;

    double sampling_proportion_ = 0.1;
};

template <typename RandomAccessIterator>
struct MaximumSpreadBuild : public AxisSelectionPolicy<RandomAccessIterator> {
    std::size_t operator()(const IteratorPairType<RandomAccessIterator>& iterator_pair,
                           std::size_t                                   n_features,
                           ssize_t                                       depth,
                           BoundingBoxKDType<RandomAccessIterator>&      kd_bounding_box) const override;
};

}  // namespace kdtree::policy

namespace kdtree::policy {

template <typename RandomAccessIterator>
std::size_t CycleThroughAxesBuild<RandomAccessIterator>::operator()(
    const IteratorPairType<RandomAccessIterator>& iterator_pair,
    std::size_t                                   n_features,
    ssize_t                                       depth,
    BoundingBoxKDType<RandomAccessIterator>&      kd_bounding_box) const {
    kdtree::policy::ignore_parameters(iterator_pair, kd_bounding_box);
    // cycle through the cut_feature_index (dimension) according to the current depth & post-increment depth
    // select the cut_feature_index according to the one with the most variance
    return depth % n_features;
}

template <typename RandomAccessIterator>
std::size_t HighestVarianceBuild<RandomAccessIterator>::operator()(
    const IteratorPairType<RandomAccessIterator>& iterator_pair,
    std::size_t                                   n_features,
    ssize_t                                       depth,
    BoundingBoxKDType<RandomAccessIterator>&      kd_bounding_box) const {
    kdtree::policy::ignore_parameters(depth, kd_bounding_box);
    // select the cut_feature_index according to the one with the most variance
    return kdtree::utils::select_axis_with_largest_variance<RandomAccessIterator>(
        iterator_pair.first, iterator_pair.second, n_features, sampling_proportion_);
}

template <typename RandomAccessIterator>
std::size_t MaximumSpreadBuild<RandomAccessIterator>::operator()(
    const IteratorPairType<RandomAccessIterator>& iterator_pair,
    std::size_t                                   n_features,
    ssize_t                                       depth,
    BoundingBoxKDType<RandomAccessIterator>&      kd_bounding_box) const {
    kdtree::policy::ignore_parameters(iterator_pair, n_features, depth);
    // select the cut_feature_index according to the one with the most spread (min-max values)
    return kdtree::utils::select_axis_with_largest_bounding_box_difference<RandomAccessIterator>(kd_bounding_box);
}

}  // namespace kdtree::policy