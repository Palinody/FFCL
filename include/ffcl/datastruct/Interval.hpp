#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/bounds/segment_representation/MinAndMax.hpp"

#include <sys/types.h>  // ssize_t
#include <cmath>
#include <limits>
#include <vector>

namespace ffcl::datastruct {

// Primary template for non-iterator types.
template <typename T, bool IsIterator = common::is_iterator<T>::value>
struct GetTypeFromIteratorOrTrivialType {
    static_assert(std::is_trivial_v<T>, "Non-iterator type must be trivial");
    using type = T;
};

// Specialization for iterator types.
template <typename T>
struct GetTypeFromIteratorOrTrivialType<T, true> {
    using type = typename std::iterator_traits<T>::value_type;
};

template <typename T>
using Interval = bounds::segment_representation::MinAndMax<typename GetTypeFromIteratorOrTrivialType<T>::type>;

template <typename T>
using HyperInterval = std::vector<Interval<T>>;

template <typename SamplesIterator>
auto make_interval(const SamplesIterator& samples_range_first,
                   const SamplesIterator& samples_range_last,
                   std::size_t            n_features,
                   ssize_t                feature_index) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const std::size_t n_samples = common::get_n_samples(samples_range_first, samples_range_last, n_features);

    auto interval = Interval<DataType>(std::numeric_limits<DataType>::max(), std::numeric_limits<DataType>::lowest());

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // a candidate for being a min, max or min-max compared to the current min and max according to the current
        // feature_index
        const auto min_max_feature_candidate = samples_range_first[sample_index * n_features + feature_index];

        if (min_max_feature_candidate < interval.first()) {
            interval.first() = min_max_feature_candidate;
        }
        if (min_max_feature_candidate > interval.second()) {
            interval.second() = min_max_feature_candidate;
        }
    }
    return interval;
}

template <typename IndicesIterator, typename SamplesIterator>
auto make_interval(const IndicesIterator& indices_range_first,
                   const IndicesIterator& indices_range_last,
                   const SamplesIterator& samples_range_first,
                   const SamplesIterator& samples_range_last,
                   std::size_t            n_features,
                   ssize_t                feature_index) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    common::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    auto interval = Interval<DataType>(std::numeric_limits<DataType>::max(), std::numeric_limits<DataType>::lowest());

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // a candidate for being a min, max or min-max compared to the current min and max according to the current
        // feature_index
        const auto min_max_feature_candidate =
            samples_range_first[indices_range_first[sample_index] * n_features + feature_index];

        if (min_max_feature_candidate < interval.first()) {
            interval.first() = min_max_feature_candidate;
        }
        if (min_max_feature_candidate > interval.second()) {
            interval.second() = min_max_feature_candidate;
        }
    }
    return interval;
}

template <typename SamplesIterator>
auto make_hyper_interval(const SamplesIterator& samples_range_first,
                         const SamplesIterator& samples_range_last,
                         std::size_t            n_features) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const std::size_t n_samples = common::get_n_samples(samples_range_first, samples_range_last, n_features);

    // min max elements per feature vector
    auto hyper_interval = HyperInterval<SamplesIterator>(
        n_features, Interval<DataType>(std::numeric_limits<DataType>::max(), std::numeric_limits<DataType>::lowest()));

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            // a candidate for being a min, max or min-max compared to the current min and max according to the current
            // feature_index
            const auto min_max_feature_candidate = samples_range_first[sample_index * n_features + feature_index];

            if (min_max_feature_candidate < hyper_interval[feature_index].first()) {
                hyper_interval[feature_index].first() = min_max_feature_candidate;
            }
            if (min_max_feature_candidate > hyper_interval[feature_index].second()) {
                hyper_interval[feature_index].second() = min_max_feature_candidate;
            }
        }
    }
    return hyper_interval;
}

template <typename IndicesIterator, typename SamplesIterator>
auto make_hyper_interval(const IndicesIterator& indices_range_first,
                         const IndicesIterator& indices_range_last,
                         const SamplesIterator& samples_range_first,
                         const SamplesIterator& samples_range_last,
                         std::size_t            n_features) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    common::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    // min max elements per feature vector
    auto hyper_interval = HyperInterval<SamplesIterator>(
        n_features, Interval<DataType>(std::numeric_limits<DataType>::max(), std::numeric_limits<DataType>::lowest()));

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            // a candidate for being a min, max or min-max compared to the current min and max according to the current
            // feature_index
            const auto min_max_feature_candidate =
                samples_range_first[indices_range_first[sample_index] * n_features + feature_index];

            if (min_max_feature_candidate < hyper_interval[feature_index].first()) {
                hyper_interval[feature_index].first() = min_max_feature_candidate;
            }
            if (min_max_feature_candidate > hyper_interval[feature_index].second()) {
                hyper_interval[feature_index].second() = min_max_feature_candidate;
            }
        }
    }
    return hyper_interval;
}

template <typename SamplesIterator>
bool is_sample_in_hyper_interval(const SamplesIterator&                feature_first,
                                 const SamplesIterator&                feature_last,
                                 const HyperInterval<SamplesIterator>& hyper_interval) {
    const std::size_t n_features = std::distance(feature_first, feature_last);

    for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
        // A sample is inside the bounding box if p is in [lo, hi]
        if (feature_first[feature_index] < hyper_interval[feature_index].first() ||
            feature_first[feature_index] > hyper_interval[feature_index].second()) {
            return false;
        }
    }
    return true;
}

template <typename SamplesIterator>
auto relative_to_absolute_hyper_interval_coordinates(const SamplesIterator&           feature_first,
                                                     const SamplesIterator&           feature_last,
                                                     HyperInterval<SamplesIterator>&& relative_coordinates_sequence) {
    const std::size_t n_features = std::distance(feature_first, feature_last);

    // make a copy that will be the translated version
    auto range_bounding_box = std::forward<HyperInterval<SamplesIterator>>(relative_coordinates_sequence);

    for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
        // get the 1D bounding box (or range) w.r.t. the current dimension
        auto& range = range_bounding_box[feature_index];
        // shift it by the amount specified at the right dimension
        range.first += feature_first[feature_index];
        range.second += feature_first[feature_index];
    }
    return range_bounding_box;
}

}  // namespace ffcl::datastruct