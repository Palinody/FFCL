#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDTreeAlgorithms.hpp"

#include <array>

namespace kdtree::policy {

template <typename RandomAccessIntIterator, typename RandomAccessIterator, std::size_t FeatureMaskSize = 0>
class IndexedAxisSelectionPolicy {
  public:
    using DataType = typename RandomAccessIterator::value_type;

    IndexedAxisSelectionPolicy() = default;

    // IndexedAxisSelectionPolicy(const std::vector<DataType>& feature_mask)
    //   : feature_mask_{feature_mask} {}

    inline virtual std::size_t operator()(RandomAccessIntIterator                  index_first,
                                          RandomAccessIntIterator                  index_last,
                                          RandomAccessIterator                     samples_first,
                                          RandomAccessIterator                     samples_last,
                                          std::size_t                              n_features,
                                          ssize_t                                  depth,
                                          BoundingBoxKDType<RandomAccessIterator>& kd_bounding_box) const = 0;
};

template <typename RandomAccessIntIterator, typename RandomAccessIterator, std::size_t FeatureMaskSize = 0>
class IndexedCycleThroughAxesBuild
  : public IndexedAxisSelectionPolicy<RandomAccessIntIterator, RandomAccessIterator, FeatureMaskSize> {
  public:
    using DataType = typename RandomAccessIterator::value_type;

    inline std::size_t operator()(RandomAccessIntIterator                  index_first,
                                  RandomAccessIntIterator                  index_last,
                                  RandomAccessIterator                     samples_first,
                                  RandomAccessIterator                     samples_last,
                                  std::size_t                              n_features,
                                  ssize_t                                  depth,
                                  BoundingBoxKDType<RandomAccessIterator>& kd_bounding_box) const;

    constexpr IndexedCycleThroughAxesBuild& feature_mask(const std::vector<std::size_t>& feature_mask) {
        feature_mask_ = feature_mask;
        return *this;
    }

    constexpr IndexedCycleThroughAxesBuild& feature_mask(std::vector<std::size_t>&& feature_mask) {
        feature_mask_ = std::move(feature_mask);
        return *this;
    }

  private:
    std::vector<std::size_t> feature_mask_;
};

template <typename RandomAccessIntIterator, typename RandomAccessIterator, std::size_t FeatureMaskSize = 0>
class IndexedHighestVarianceBuild
  : public IndexedAxisSelectionPolicy<RandomAccessIntIterator, RandomAccessIterator, FeatureMaskSize> {
  public:
    using DataType = typename RandomAccessIterator::value_type;

    IndexedHighestVarianceBuild& sampling_proportion(double sampling_proportion) {
        sampling_proportion_ = sampling_proportion;
        return *this;
    }

    inline std::size_t operator()(RandomAccessIntIterator                  index_first,
                                  RandomAccessIntIterator                  index_last,
                                  RandomAccessIterator                     samples_first,
                                  RandomAccessIterator                     samples_last,
                                  std::size_t                              n_features,
                                  ssize_t                                  depth,
                                  BoundingBoxKDType<RandomAccessIterator>& kd_bounding_box) const;

    constexpr IndexedHighestVarianceBuild& feature_mask(const std::vector<std::size_t>& feature_mask) {
        feature_mask_ = feature_mask;
        return *this;
    }

    constexpr IndexedHighestVarianceBuild& feature_mask(std::vector<std::size_t>&& feature_mask) {
        feature_mask_ = std::move(feature_mask);
        return *this;
    }

  private:
    std::vector<std::size_t> feature_mask_;
    // default sampling proportion value. Range: [0, 1]
    double sampling_proportion_ = 0.1;
};

template <typename RandomAccessIntIterator, typename RandomAccessIterator, std::size_t FeatureMaskSize = 0>
class IndexedMaximumSpreadBuild
  : public IndexedAxisSelectionPolicy<RandomAccessIntIterator, RandomAccessIterator, FeatureMaskSize> {
  public:
    using DataType = typename RandomAccessIterator::value_type;

    inline std::size_t operator()(RandomAccessIntIterator                  index_first,
                                  RandomAccessIntIterator                  index_last,
                                  RandomAccessIterator                     samples_first,
                                  RandomAccessIterator                     samples_last,
                                  std::size_t                              n_features,
                                  ssize_t                                  depth,
                                  BoundingBoxKDType<RandomAccessIterator>& kd_bounding_box) const;

    constexpr IndexedMaximumSpreadBuild& feature_mask(const std::vector<std::size_t>& feature_mask) {
        feature_mask_ = feature_mask;
        return *this;
    }

    constexpr IndexedMaximumSpreadBuild& feature_mask(std::vector<std::size_t>&& feature_mask) {
        feature_mask_ = std::move(feature_mask);
        return *this;
    }

  private:
    std::vector<std::size_t> feature_mask_;
};

}  // namespace kdtree::policy

namespace kdtree::policy {

template <typename RandomAccessIntIterator, typename RandomAccessIterator, std::size_t FeatureMaskSize>
std::size_t IndexedCycleThroughAxesBuild<RandomAccessIntIterator, RandomAccessIterator, FeatureMaskSize>::operator()(
    RandomAccessIntIterator                  index_first,
    RandomAccessIntIterator                  index_last,
    RandomAccessIterator                     samples_first,
    RandomAccessIterator                     samples_last,
    std::size_t                              n_features,
    ssize_t                                  depth,
    BoundingBoxKDType<RandomAccessIterator>& kd_bounding_box) const {
    common::utils::ignore_parameters(index_first, index_last, samples_first, samples_last, kd_bounding_box);
    if (feature_mask_.empty()) {
        // cycle through the cut_feature_index (dimension) according to the current depth & post-increment depth
        // select the cut_feature_index according to the one with the most variance
        return depth % n_features;
    }
    return feature_mask_[depth % feature_mask_.size()];
}

template <typename RandomAccessIntIterator, typename RandomAccessIterator, std::size_t FeatureMaskSize>
std::size_t IndexedHighestVarianceBuild<RandomAccessIntIterator, RandomAccessIterator, FeatureMaskSize>::operator()(
    RandomAccessIntIterator                  index_first,
    RandomAccessIntIterator                  index_last,
    RandomAccessIterator                     samples_first,
    RandomAccessIterator                     samples_last,
    std::size_t                              n_features,
    ssize_t                                  depth,
    BoundingBoxKDType<RandomAccessIterator>& kd_bounding_box) const {
    common::utils::ignore_parameters(depth, kd_bounding_box);

    if (feature_mask_.empty()) {
        // select the cut_feature_index according to the one with the most variance
        return kdtree::algorithms::select_axis_with_largest_variance<RandomAccessIntIterator, RandomAccessIterator>(
            /**/ index_first,
            /**/ index_last,
            /**/ samples_first,
            /**/ samples_last,
            /**/ n_features,
            /**/ sampling_proportion_);
    }
    return kdtree::algorithms::select_axis_with_largest_variance<RandomAccessIntIterator, RandomAccessIterator>(
        /**/ index_first,
        /**/ index_last,
        /**/ samples_first,
        /**/ samples_last,
        /**/ n_features,
        /**/ sampling_proportion_,
        /**/ feature_mask_);
}

template <typename RandomAccessIntIterator, typename RandomAccessIterator, std::size_t FeatureMaskSize>
std::size_t IndexedMaximumSpreadBuild<RandomAccessIntIterator, RandomAccessIterator, FeatureMaskSize>::operator()(
    RandomAccessIntIterator                  index_first,
    RandomAccessIntIterator                  index_last,
    RandomAccessIterator                     samples_first,
    RandomAccessIterator                     samples_last,
    std::size_t                              n_features,
    ssize_t                                  depth,
    BoundingBoxKDType<RandomAccessIterator>& kd_bounding_box) const {
    common::utils::ignore_parameters(index_first, index_last, samples_first, samples_last, n_features, depth);

    if (feature_mask_.empty()) {
        // select the cut_feature_index according to the one with the most spread (min-max values)
        return kdtree::algorithms::select_axis_with_largest_bounding_box_difference<RandomAccessIterator>(
            kd_bounding_box);
    }
    return kdtree::algorithms::select_axis_with_largest_bounding_box_difference<RandomAccessIterator>(kd_bounding_box,
                                                                                                      feature_mask_);
}

}  // namespace kdtree::policy