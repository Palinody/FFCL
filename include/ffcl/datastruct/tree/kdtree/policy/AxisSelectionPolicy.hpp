#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/tree/kdtree/KDTreeAlgorithms.hpp"

#include "ffcl/datastruct/bounds/AABBWithCentroid.hpp"
#include "ffcl/datastruct/bounds/segment/LowerBoundAndUpperBound.hpp"

#include <cstddef>  // std::size_t

namespace ffcl::datastruct::kdtree::policy {

template <typename IndicesIterator,
          typename SamplesIterator,
          typename Bound = bounds::AABBWithCentroid<
              bounds::segment::LowerBoundAndUpperBound<typename std::iterator_traits<SamplesIterator>::value_type>>>
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

    virtual std::size_t operator()(const IndicesIterator& indices_range_first,
                                   const IndicesIterator& indices_range_last,
                                   const SamplesIterator& samples_range_first,
                                   const SamplesIterator& samples_range_last,
                                   std::size_t            n_features,
                                   std::size_t            depth,
                                   const Bound&           bound) const = 0;
};

template <typename IndicesIterator,
          typename SamplesIterator,
          typename Bound = bounds::AABBWithCentroid<
              bounds::segment::LowerBoundAndUpperBound<typename std::iterator_traits<SamplesIterator>::value_type>>>
class CycleThroughAxesBuild : public AxisSelectionPolicy<IndicesIterator, SamplesIterator, Bound> {
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

    std::size_t operator()(const IndicesIterator& indices_range_first,
                           const IndicesIterator& indices_range_last,
                           const SamplesIterator& samples_range_first,
                           const SamplesIterator& samples_range_last,
                           std::size_t            n_features,
                           std::size_t            depth,
                           const Bound&           bound) const;

  private:
    // contains the sequence of feature indices of interest
    std::vector<std::size_t> feature_mask_;
};

template <typename IndicesIterator,
          typename SamplesIterator,
          typename Bound = bounds::AABBWithCentroid<
              bounds::segment::LowerBoundAndUpperBound<typename std::iterator_traits<SamplesIterator>::value_type>>>
class HighestVarianceBuild : public AxisSelectionPolicy<IndicesIterator, SamplesIterator, Bound> {
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

    std::size_t operator()(const IndicesIterator& indices_range_first,
                           const IndicesIterator& indices_range_last,
                           const SamplesIterator& samples_range_first,
                           const SamplesIterator& samples_range_last,
                           std::size_t            n_features,
                           std::size_t            depth,
                           const Bound&           bound) const;

  private:
    // contains the sequence of feature indices of interest
    std::vector<std::size_t> feature_mask_;
    // default sampling proportion value. Range: [0, 1]
    double sampling_rate_ = 0.1;
};

template <typename IndicesIterator,
          typename SamplesIterator,
          typename Bound = bounds::AABBWithCentroid<
              bounds::segment::LowerBoundAndUpperBound<typename std::iterator_traits<SamplesIterator>::value_type>>>
class MaximumSpreadBuild : public AxisSelectionPolicy<IndicesIterator, SamplesIterator, Bound> {
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

    std::size_t operator()(const IndicesIterator& indices_range_first,
                           const IndicesIterator& indices_range_last,
                           const SamplesIterator& samples_range_first,
                           const SamplesIterator& samples_range_last,
                           std::size_t            n_features,
                           std::size_t            depth,
                           const Bound&           bound) const;

  private:
    // contains the sequence of feature indices of interest
    std::vector<std::size_t> feature_mask_;
};

template <typename IndicesIterator, typename SamplesIterator, typename Bound>
std::size_t CycleThroughAxesBuild<IndicesIterator, SamplesIterator, Bound>::operator()(
    const IndicesIterator& indices_range_first,
    const IndicesIterator& indices_range_last,
    const SamplesIterator& samples_range_first,
    const SamplesIterator& samples_range_last,
    std::size_t            n_features,
    std::size_t            depth,
    const Bound&           bound) const {
    ffcl::common::ignore_parameters(
        indices_range_first, indices_range_last, samples_range_first, samples_range_last, bound);
    if (feature_mask_.empty()) {
        // cycle through the cut_feature_index (dimension) according to the current depth & post-increment depth
        // select the cut_feature_index according to the one with the most variance
        return depth % n_features;
    }
    // cycle through the feature mask possibilities provided by the feature indices sequence
    return feature_mask_[depth % feature_mask_.size()];
}

template <typename IndicesIterator, typename SamplesIterator, typename Bound>
std::size_t HighestVarianceBuild<IndicesIterator, SamplesIterator, Bound>::operator()(
    const IndicesIterator& indices_range_first,
    const IndicesIterator& indices_range_last,
    const SamplesIterator& samples_range_first,
    const SamplesIterator& samples_range_last,
    std::size_t            n_features,
    std::size_t            depth,
    const Bound&           bound) const {
    ffcl::common::ignore_parameters(depth, bound);

    if (feature_mask_.empty()) {
        // select the cut_feature_index according to the one with the most variance
        return kdtree::algorithms::select_largest_variance_axis<IndicesIterator, SamplesIterator, Bound>(
            /**/ indices_range_first,
            /**/ indices_range_last,
            /**/ samples_range_first,
            /**/ samples_range_last,
            /**/ n_features,
            /**/ sampling_rate_);
    }
    return kdtree::algorithms::select_largest_variance_axis<IndicesIterator, SamplesIterator, Bound>(
        /**/ indices_range_first,
        /**/ indices_range_last,
        /**/ samples_range_first,
        /**/ samples_range_last,
        /**/ n_features,
        /**/ sampling_rate_,
        /**/ feature_mask_);
}

template <typename IndicesIterator, typename SamplesIterator, typename Bound>
std::size_t MaximumSpreadBuild<IndicesIterator, SamplesIterator, Bound>::operator()(
    const IndicesIterator& indices_range_first,
    const IndicesIterator& indices_range_last,
    const SamplesIterator& samples_range_first,
    const SamplesIterator& samples_range_last,
    std::size_t            n_features,
    std::size_t            depth,
    const Bound&           bound) const {
    ffcl::common::ignore_parameters(
        indices_range_first, indices_range_last, samples_range_first, samples_range_last, n_features, depth);

    if (feature_mask_.empty()) {
        // select the cut_feature_index according to the one with the most spread (min-max values)
        return kdtree::algorithms::select_longest_axis(bound);
    }
    return kdtree::algorithms::select_longest_axis(bound, feature_mask_);
}

}  // namespace ffcl::datastruct::kdtree::policy