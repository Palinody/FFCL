#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/FeatureMaskArray.hpp"
#include "ffcl/datastruct/kdtree/KDTreeAlgorithms.hpp"

#include <array>

namespace kdtree::policy {

template <typename IndicesIterator, typename SamplesIterator>
class AxisSelectionPolicy {
  public:
    virtual std::size_t operator()(IndicesIterator                              index_first,
                                   IndicesIterator                              index_last,
                                   SamplesIterator                              samples_first,
                                   SamplesIterator                              samples_last,
                                   std::size_t                                  n_features,
                                   ssize_t                                      depth,
                                   ffcl::bbox::HyperRangeType<SamplesIterator>& kd_bounding_box) const = 0;
};

template <typename IndicesIterator, typename SamplesIterator>
class CycleThroughAxesBuild : public AxisSelectionPolicy<IndicesIterator, SamplesIterator> {
  public:
    using DataType = typename SamplesIterator::value_type;

    constexpr CycleThroughAxesBuild& feature_mask(const std::vector<std::size_t>& feature_mask) {
        feature_mask_ = feature_mask;
        return *this;
    }

    constexpr CycleThroughAxesBuild& feature_mask(std::vector<std::size_t>&& feature_mask) {
        feature_mask_ = std::move(feature_mask);
        return *this;
    }

    std::size_t operator()(IndicesIterator                              index_first,
                           IndicesIterator                              index_last,
                           SamplesIterator                              samples_first,
                           SamplesIterator                              samples_last,
                           std::size_t                                  n_features,
                           ssize_t                                      depth,
                           ffcl::bbox::HyperRangeType<SamplesIterator>& kd_bounding_box) const;

  private:
    // contains the sequence of feature indices of interest
    std::vector<std::size_t> feature_mask_;
};

template <typename IndicesIterator, typename SamplesIterator>
class HighestVarianceBuild : public AxisSelectionPolicy<IndicesIterator, SamplesIterator> {
  public:
    using DataType = typename SamplesIterator::value_type;

    constexpr HighestVarianceBuild& sampling_proportion(double sampling_proportion) {
        sampling_proportion_ = sampling_proportion;
        return *this;
    }

    constexpr HighestVarianceBuild& feature_mask(const std::vector<std::size_t>& feature_mask) {
        feature_mask_ = feature_mask;
        return *this;
    }

    constexpr HighestVarianceBuild& feature_mask(std::vector<std::size_t>&& feature_mask) {
        feature_mask_ = std::move(feature_mask);
        return *this;
    }

    std::size_t operator()(IndicesIterator                              index_first,
                           IndicesIterator                              index_last,
                           SamplesIterator                              samples_first,
                           SamplesIterator                              samples_last,
                           std::size_t                                  n_features,
                           ssize_t                                      depth,
                           ffcl::bbox::HyperRangeType<SamplesIterator>& kd_bounding_box) const;

  private:
    // contains the sequence of feature indices of interest
    std::vector<std::size_t> feature_mask_;
    // default sampling proportion value. Range: [0, 1]
    double sampling_proportion_ = 0.1;
};

template <typename IndicesIterator, typename SamplesIterator>
class MaximumSpreadBuild : public AxisSelectionPolicy<IndicesIterator, SamplesIterator> {
  public:
    using DataType = typename SamplesIterator::value_type;

    constexpr MaximumSpreadBuild& feature_mask(const std::vector<std::size_t>& feature_mask) {
        feature_mask_ = feature_mask;
        return *this;
    }

    constexpr MaximumSpreadBuild& feature_mask(std::vector<std::size_t>&& feature_mask) {
        feature_mask_ = std::move(feature_mask);
        return *this;
    }

    std::size_t operator()(IndicesIterator                              index_first,
                           IndicesIterator                              index_last,
                           SamplesIterator                              samples_first,
                           SamplesIterator                              samples_last,
                           std::size_t                                  n_features,
                           ssize_t                                      depth,
                           ffcl::bbox::HyperRangeType<SamplesIterator>& kd_bounding_box) const;

  private:
    // contains the sequence of feature indices of interest
    std::vector<std::size_t> feature_mask_;
};

}  // namespace kdtree::policy

namespace kdtree::policy {

template <typename IndicesIterator, typename SamplesIterator>
std::size_t CycleThroughAxesBuild<IndicesIterator, SamplesIterator>::operator()(
    IndicesIterator                              index_first,
    IndicesIterator                              index_last,
    SamplesIterator                              samples_first,
    SamplesIterator                              samples_last,
    std::size_t                                  n_features,
    ssize_t                                      depth,
    ffcl::bbox::HyperRangeType<SamplesIterator>& kd_bounding_box) const {
    common::utils::ignore_parameters(index_first, index_last, samples_first, samples_last, kd_bounding_box);
    if (feature_mask_.empty()) {
        // cycle through the cut_feature_index (dimension) according to the current depth & post-increment depth
        // select the cut_feature_index according to the one with the most variance
        return depth % n_features;
    }
    // cycle through the feature mask possibilities provided by the feature indices sequence
    return feature_mask_[depth % feature_mask_.size()];
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t HighestVarianceBuild<IndicesIterator, SamplesIterator>::operator()(
    IndicesIterator                              index_first,
    IndicesIterator                              index_last,
    SamplesIterator                              samples_first,
    SamplesIterator                              samples_last,
    std::size_t                                  n_features,
    ssize_t                                      depth,
    ffcl::bbox::HyperRangeType<SamplesIterator>& kd_bounding_box) const {
    common::utils::ignore_parameters(depth, kd_bounding_box);

    if (feature_mask_.empty()) {
        // select the cut_feature_index according to the one with the most variance
        return kdtree::algorithms::select_axis_with_largest_variance<IndicesIterator, SamplesIterator>(
            /**/ index_first,
            /**/ index_last,
            /**/ samples_first,
            /**/ samples_last,
            /**/ n_features,
            /**/ sampling_proportion_);
    }
    return kdtree::algorithms::select_axis_with_largest_variance<IndicesIterator, SamplesIterator>(
        /**/ index_first,
        /**/ index_last,
        /**/ samples_first,
        /**/ samples_last,
        /**/ n_features,
        /**/ sampling_proportion_,
        /**/ feature_mask_);
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t MaximumSpreadBuild<IndicesIterator, SamplesIterator>::operator()(
    IndicesIterator                              index_first,
    IndicesIterator                              index_last,
    SamplesIterator                              samples_first,
    SamplesIterator                              samples_last,
    std::size_t                                  n_features,
    ssize_t                                      depth,
    ffcl::bbox::HyperRangeType<SamplesIterator>& kd_bounding_box) const {
    common::utils::ignore_parameters(index_first, index_last, samples_first, samples_last, n_features, depth);

    if (feature_mask_.empty()) {
        // select the cut_feature_index according to the one with the most spread (min-max values)
        return kdtree::algorithms::select_axis_with_largest_bounding_box_difference<SamplesIterator>(kd_bounding_box);
    }
    return kdtree::algorithms::select_axis_with_largest_bounding_box_difference<SamplesIterator>(kd_bounding_box,
                                                                                                 feature_mask_);
}

}  // namespace kdtree::policy