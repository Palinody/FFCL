#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/bounds/segment_representation/MinAndMax.hpp"

#include <cmath>
#include <cstddef>  // std::size_t
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
                   std::size_t            feature_index) {
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
                   std::size_t            feature_index) {
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

template <typename FeaturesIterator>
bool is_sample_in_hyper_interval(const FeaturesIterator&                features_range_first,
                                 const FeaturesIterator&                features_range_last,
                                 const HyperInterval<FeaturesIterator>& hyper_interval) {
    const std::size_t n_features = std::distance(features_range_first, features_range_last);

    for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
        // A sample is inside the bounding box if p is in [lo, hi]
        if (features_range_first[feature_index] < hyper_interval[feature_index].first() ||
            features_range_first[feature_index] > hyper_interval[feature_index].second()) {
            return false;
        }
    }
    return true;
}

template <typename FeaturesIterator>
auto relative_to_absolute_hyper_interval_coordinates(const FeaturesIterator&           features_range_first,
                                                     const FeaturesIterator&           features_range_last,
                                                     HyperInterval<FeaturesIterator>&& relative_coordinates_sequence) {
    const std::size_t n_features = std::distance(features_range_first, features_range_last);

    // make a copy that will be the translated version or use move semantics on the original data
    auto hyper_interval = std::forward<HyperInterval<FeaturesIterator>>(relative_coordinates_sequence);

    for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
        // get the 1D bounding box (or range) w.r.t. the current dimension
        auto& range = hyper_interval[feature_index];
        // shift it by the amount specified at the right dimension
        range.first += features_range_first[feature_index];
        range.second += features_range_first[feature_index];
    }
    return hyper_interval;
}

}  // namespace ffcl::datastruct