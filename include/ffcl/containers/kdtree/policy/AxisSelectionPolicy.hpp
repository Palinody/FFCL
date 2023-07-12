#pragma once

#include "ffcl/containers/kdtree/KDTreeAlgorithms.hpp"

#include "ffcl/common/Utils.hpp"

namespace kdtree::policy {

template <typename RandomAccessIterator>
struct AxisSelectionPolicy {
    AxisSelectionPolicy() = default;

    inline virtual std::size_t operator()(RandomAccessIterator                  samples_first,
                                          RandomAccessIterator                  samples_last,
                                          std::size_t                           n_features,
                                          ssize_t                               depth,
                                          HyperRangeType<RandomAccessIterator>& kd_bounding_box) const = 0;
};

template <typename RandomAccessIterator>
struct CycleThroughAxesBuild : public AxisSelectionPolicy<RandomAccessIterator> {
    inline std::size_t operator()(RandomAccessIterator                  samples_first,
                                  RandomAccessIterator                  samples_last,
                                  std::size_t                           n_features,
                                  ssize_t                               depth,
                                  HyperRangeType<RandomAccessIterator>& kd_bounding_box) const;
};

template <typename RandomAccessIterator>
struct HighestVarianceBuild : public AxisSelectionPolicy<RandomAccessIterator> {
    HighestVarianceBuild& sampling_proportion(double sampling_proportion) {
        sampling_proportion_ = sampling_proportion;
        return *this;
    }

    inline std::size_t operator()(RandomAccessIterator                  samples_first,
                                  RandomAccessIterator                  samples_last,
                                  std::size_t                           n_features,
                                  ssize_t                               depth,
                                  HyperRangeType<RandomAccessIterator>& kd_bounding_box) const;
    // default sampling proportion value. Range: [0, 1]
    double sampling_proportion_ = 0.1;
};

template <typename RandomAccessIterator>
struct MaximumSpreadBuild : public AxisSelectionPolicy<RandomAccessIterator> {
    inline std::size_t operator()(RandomAccessIterator                  samples_first,
                                  RandomAccessIterator                  samples_last,
                                  std::size_t                           n_features,
                                  ssize_t                               depth,
                                  HyperRangeType<RandomAccessIterator>& kd_bounding_box) const;
};

}  // namespace kdtree::policy

namespace kdtree::policy {

template <typename RandomAccessIterator>
std::size_t CycleThroughAxesBuild<RandomAccessIterator>::operator()(
    RandomAccessIterator                  samples_first,
    RandomAccessIterator                  samples_last,
    std::size_t                           n_features,
    ssize_t                               depth,
    HyperRangeType<RandomAccessIterator>& kd_bounding_box) const {
    common::utils::ignore_parameters(samples_first, samples_last, kd_bounding_box);
    // cycle through the cut_feature_index (dimension) according to the current depth & post-increment depth
    // select the cut_feature_index according to the one with the most variance
    return depth % n_features;
}

template <typename RandomAccessIterator>
std::size_t HighestVarianceBuild<RandomAccessIterator>::operator()(
    RandomAccessIterator                  samples_first,
    RandomAccessIterator                  samples_last,
    std::size_t                           n_features,
    ssize_t                               depth,
    HyperRangeType<RandomAccessIterator>& kd_bounding_box) const {
    common::utils::ignore_parameters(depth, kd_bounding_box);
    // select the cut_feature_index according to the one with the most variance
    return kdtree::algorithms::select_axis_with_largest_variance<RandomAccessIterator>(
        samples_first, samples_last, n_features, sampling_proportion_);
}

template <typename RandomAccessIterator>
std::size_t MaximumSpreadBuild<RandomAccessIterator>::operator()(
    RandomAccessIterator                  samples_first,
    RandomAccessIterator                  samples_last,
    std::size_t                           n_features,
    ssize_t                               depth,
    HyperRangeType<RandomAccessIterator>& kd_bounding_box) const {
    common::utils::ignore_parameters(samples_first, samples_last, n_features, depth);
    // select the cut_feature_index according to the one with the most spread (min-max values)
    return kdtree::algorithms::select_axis_with_largest_bounding_box_difference<RandomAccessIterator>(kd_bounding_box);
}

}  // namespace kdtree::policy