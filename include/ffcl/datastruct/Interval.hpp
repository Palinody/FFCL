#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/bounds/AABBWithCentroid.hpp"
#include "ffcl/datastruct/bounds/segment/LowerBoundAndUpperBound.hpp"

#include <cmath>
#include <cstddef>  // std::size_t
#include <limits>
#include <vector>

namespace ffcl::datastruct {

template <typename SamplesIterator,
          typename Segment =
              bounds::segment::LowerBoundAndUpperBound<typename std::iterator_traits<SamplesIterator>::value_type>>
auto make_tight_segment(const SamplesIterator& samples_range_first,
                        const SamplesIterator& samples_range_last,
                        std::size_t            n_features,
                        std::size_t            feature_index) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const std::size_t n_samples = common::get_n_samples(samples_range_first, samples_range_last, n_features);

    auto segment = Segment{std::numeric_limits<DataType>::max(), std::numeric_limits<DataType>::lowest()};

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // a candidate for being a min, max or min-max compared to the current min and max according to the current
        // feature_index
        const auto min_max_feature_candidate = samples_range_first[sample_index * n_features + feature_index];

        if (min_max_feature_candidate < segment.lower_bound()) {
            segment.update_lower_bound(min_max_feature_candidate);
        }
        if (min_max_feature_candidate > segment.upper_bound()) {
            segment.update_upper_bound(min_max_feature_candidate);
        }
    }
    return segment;
}

template <typename IndicesIterator,
          typename SamplesIterator,
          typename Segment =
              bounds::segment::LowerBoundAndUpperBound<typename std::iterator_traits<SamplesIterator>::value_type>>
auto make_tight_segment(const IndicesIterator& indices_range_first,
                        const IndicesIterator& indices_range_last,
                        const SamplesIterator& samples_range_first,
                        const SamplesIterator& samples_range_last,
                        std::size_t            n_features,
                        std::size_t            feature_index) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    common::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    auto segment = Segment{std::numeric_limits<DataType>::max(), std::numeric_limits<DataType>::lowest()};

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // a candidate for being a min, max or min-max compared to the current min and max according to the current
        // feature_index
        const auto min_max_feature_candidate =
            samples_range_first[indices_range_first[sample_index] * n_features + feature_index];

        if (min_max_feature_candidate < segment.lower_bound()) {
            segment.update_lower_bound(min_max_feature_candidate);
        }
        if (min_max_feature_candidate > segment.upper_bound()) {
            segment.update_upper_bound(min_max_feature_candidate);
        }
    }
    return segment;
}

template <typename SamplesIterator,
          typename Bound = bounds::AABBWithCentroid<
              bounds::segment::LowerBoundAndUpperBound<typename std::iterator_traits<SamplesIterator>::value_type>>>
auto make_tight_bound(const SamplesIterator& samples_range_first,
                      const SamplesIterator& samples_range_last,
                      std::size_t            n_features) {
    using DataType    = typename std::iterator_traits<SamplesIterator>::value_type;
    using SegmentType = typename Bound::SegmentType;

    const std::size_t n_samples = common::get_n_samples(samples_range_first, samples_range_last, n_features);

    // min max elements per feature vector
    auto segments = std::vector<SegmentType>(
        n_features, SegmentType{std::numeric_limits<DataType>::max(), std::numeric_limits<DataType>::lowest()});

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            // a candidate for being a min, max or min-max compared to the current min and max according to the current
            // feature_index
            const auto min_max_feature_candidate = samples_range_first[sample_index * n_features + feature_index];

            if (min_max_feature_candidate < segments[feature_index].lower_bound()) {
                segments[feature_index].update_lower_bound(min_max_feature_candidate);
            }
            if (min_max_feature_candidate > segments[feature_index].upper_bound()) {
                segments[feature_index].update_upper_bound(min_max_feature_candidate);
            }
        }
    }
    return Bound{std::move(segments)};
}

template <typename IndicesIterator,
          typename SamplesIterator,
          typename Bound = bounds::AABBWithCentroid<
              bounds::segment::LowerBoundAndUpperBound<typename std::iterator_traits<SamplesIterator>::value_type>>>
auto make_tight_bound(const IndicesIterator& indices_range_first,
                      const IndicesIterator& indices_range_last,
                      const SamplesIterator& samples_range_first,
                      const SamplesIterator& samples_range_last,
                      std::size_t            n_features) {
    using DataType    = typename std::iterator_traits<SamplesIterator>::value_type;
    using SegmentType = typename Bound::SegmentType;

    common::ignore_parameters(samples_range_last);

    // min max elements per feature vector
    auto segments = std::vector<SegmentType>(
        n_features, SegmentType{std::numeric_limits<DataType>::max(), std::numeric_limits<DataType>::lowest()});

    for (auto subrange_index_it = indices_range_first; subrange_index_it != indices_range_last; ++subrange_index_it) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            // a candidate for being a min, max or min-max compared to the current min and max according to the current
            // feature_index
            const auto min_max_feature_candidate = samples_range_first[*subrange_index_it * n_features + feature_index];

            if (min_max_feature_candidate < segments[feature_index].lower_bound()) {
                segments[feature_index].update_lower_bound(min_max_feature_candidate);
            }
            if (min_max_feature_candidate > segments[feature_index].upper_bound()) {
                segments[feature_index].update_upper_bound(min_max_feature_candidate);
            }
        }
    }
    return Bound{std::move(segments)};
}

}  // namespace ffcl::datastruct