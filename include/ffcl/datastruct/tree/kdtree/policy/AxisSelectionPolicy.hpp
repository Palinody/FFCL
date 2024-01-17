#pragma once

#include <cstddef>  // std::size_t
#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/FeatureMaskArray.hpp"
#include "ffcl/datastruct/tree/kdtree/KDTreeAlgorithms.hpp"

#include <array>

namespace ffcl::datastruct::kdtree::policy {

template <typename IndicesIterator, typename SamplesIterator>
class AxisSelectionPolicy {
  private:
    static_assert(common::is_iterator<IndicesIterator>::value, "IndicesIterator is not an iterator");
    static_assert(common::is_iterator<SamplesIterator>::value, "SamplesIterator is not an iterator");

    using IndexType = typename std::iterator_traits<IndicesIterator>::value_type;
    using DataType  = typename std::iterator_traits<SamplesIterator>::value_type;

    static_assert(std::is_integral_v<IndexType>, "IndexType must be integer.");
    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

  public:
    AxisSelectionPolicy() = default;

    virtual ~AxisSelectionPolicy() = default;

    virtual std::size_t operator()(IndicesIterator                 indices_range_first,
                                   IndicesIterator                 indices_range_last,
                                   SamplesIterator                 samples_range_first,
                                   SamplesIterator                 samples_range_last,
                                   std::size_t                     n_features,
                                   std::size_t                     depth,
                                   HyperInterval<SamplesIterator>& hyper_interval) const = 0;
};

template <typename IndicesIterator, typename SamplesIterator>
class CycleThroughAxesBuild : public AxisSelectionPolicy<IndicesIterator, SamplesIterator> {
  public:
    CycleThroughAxesBuild() = default;

    constexpr CycleThroughAxesBuild& feature_mask(const std::vector<std::size_t>& feature_mask) {
        feature_mask_ = feature_mask;
        return *this;
    }

    constexpr CycleThroughAxesBuild& feature_mask(std::vector<std::size_t>&& feature_mask) {
        feature_mask_ = std::move(feature_mask);
        return *this;
    }

    std::size_t operator()(IndicesIterator                 indices_range_first,
                           IndicesIterator                 indices_range_last,
                           SamplesIterator                 samples_range_first,
                           SamplesIterator                 samples_range_last,
                           std::size_t                     n_features,
                           std::size_t                     depth,
                           HyperInterval<SamplesIterator>& hyper_interval) const;

  private:
    // contains the sequence of feature indices of interest
    std::vector<std::size_t> feature_mask_;
};

template <typename IndicesIterator, typename SamplesIterator>
class HighestVarianceBuild : public AxisSelectionPolicy<IndicesIterator, SamplesIterator> {
  public:
    HighestVarianceBuild() = default;

    constexpr HighestVarianceBuild& sampling_rate(double sampling_rate) {
        sampling_rate_ = sampling_rate;
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

    std::size_t operator()(IndicesIterator                 indices_range_first,
                           IndicesIterator                 indices_range_last,
                           SamplesIterator                 samples_range_first,
                           SamplesIterator                 samples_range_last,
                           std::size_t                     n_features,
                           std::size_t                     depth,
                           HyperInterval<SamplesIterator>& hyper_interval) const;

  private:
    // contains the sequence of feature indices of interest
    std::vector<std::size_t> feature_mask_;
    // default sampling proportion value. Range: [0, 1]
    double sampling_rate_ = 0.1;
};

template <typename IndicesIterator, typename SamplesIterator>
class MaximumSpreadBuild : public AxisSelectionPolicy<IndicesIterator, SamplesIterator> {
  public:
    MaximumSpreadBuild() = default;

    constexpr MaximumSpreadBuild& feature_mask(const std::vector<std::size_t>& feature_mask) {
        feature_mask_ = feature_mask;
        return *this;
    }

    constexpr MaximumSpreadBuild& feature_mask(std::vector<std::size_t>&& feature_mask) {
        feature_mask_ = std::move(feature_mask);
        return *this;
    }

    std::size_t operator()(IndicesIterator                 indices_range_first,
                           IndicesIterator                 indices_range_last,
                           SamplesIterator                 samples_range_first,
                           SamplesIterator                 samples_range_last,
                           std::size_t                     n_features,
                           std::size_t                     depth,
                           HyperInterval<SamplesIterator>& hyper_interval) const;

  private:
    // contains the sequence of feature indices of interest
    std::vector<std::size_t> feature_mask_;
};

template <typename IndicesIterator, typename SamplesIterator>
std::size_t CycleThroughAxesBuild<IndicesIterator, SamplesIterator>::operator()(
    IndicesIterator                 indices_range_first,
    IndicesIterator                 indices_range_last,
    SamplesIterator                 samples_range_first,
    SamplesIterator                 samples_range_last,
    std::size_t                     n_features,
    std::size_t                     depth,
    HyperInterval<SamplesIterator>& hyper_interval) const {
    ffcl::common::ignore_parameters(
        indices_range_first, indices_range_last, samples_range_first, samples_range_last, hyper_interval);
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
    IndicesIterator                 indices_range_first,
    IndicesIterator                 indices_range_last,
    SamplesIterator                 samples_range_first,
    SamplesIterator                 samples_range_last,
    std::size_t                     n_features,
    std::size_t                     depth,
    HyperInterval<SamplesIterator>& hyper_interval) const {
    ffcl::common::ignore_parameters(depth, hyper_interval);

    if (feature_mask_.empty()) {
        // select the cut_feature_index according to the one with the most variance
        return kdtree::algorithms::select_axis_with_largest_variance<IndicesIterator, SamplesIterator>(
            /**/ indices_range_first,
            /**/ indices_range_last,
            /**/ samples_range_first,
            /**/ samples_range_last,
            /**/ n_features,
            /**/ sampling_rate_);
    }
    return kdtree::algorithms::select_axis_with_largest_variance<IndicesIterator, SamplesIterator>(
        /**/ indices_range_first,
        /**/ indices_range_last,
        /**/ samples_range_first,
        /**/ samples_range_last,
        /**/ n_features,
        /**/ sampling_rate_,
        /**/ feature_mask_);
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t MaximumSpreadBuild<IndicesIterator, SamplesIterator>::operator()(
    IndicesIterator                 indices_range_first,
    IndicesIterator                 indices_range_last,
    SamplesIterator                 samples_range_first,
    SamplesIterator                 samples_range_last,
    std::size_t                     n_features,
    std::size_t                     depth,
    HyperInterval<SamplesIterator>& hyper_interval) const {
    ffcl::common::ignore_parameters(
        indices_range_first, indices_range_last, samples_range_first, samples_range_last, n_features, depth);

    if (feature_mask_.empty()) {
        // select the cut_feature_index according to the one with the most spread (min-max values)
        return kdtree::algorithms::select_axis_with_largest_bounding_box_difference<SamplesIterator>(hyper_interval);
    }
    return kdtree::algorithms::select_axis_with_largest_bounding_box_difference<SamplesIterator>(hyper_interval,
                                                                                                 feature_mask_);
}

}  // namespace ffcl::datastruct::kdtree::policy