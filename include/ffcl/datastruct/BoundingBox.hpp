#pragma once

#include "ffcl/common/Utils.hpp"

#include <sys/types.h>  // ssize_t
#include <cmath>
#include <limits>
#include <vector>

namespace ffcl::datastruct::bbox {

template <typename SamplesIterator>
using IteratorPairType = std::pair<SamplesIterator, SamplesIterator>;

template <typename SamplesIterator>
using DataType = typename SamplesIterator::value_type;

template <typename SamplesIterator>
using RangeType = std::pair<DataType<SamplesIterator>, DataType<SamplesIterator>>;

template <typename SamplesIterator>
using HyperRangeType = std::vector<RangeType<SamplesIterator>>;

template <typename SamplesIterator>
auto make_1d_bounding_box(const SamplesIterator& samples_range_first,
                          const SamplesIterator& samples_range_last,
                          std::size_t            n_features,
                          ssize_t                feature_index) {
    using DataType = DataType<SamplesIterator>;

    const std::size_t n_samples = common::get_n_samples(samples_range_first, samples_range_last, n_features);

    auto bounding_box_1d =
        RangeType<SamplesIterator>(std::numeric_limits<DataType>::max(), std::numeric_limits<DataType>::lowest());

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // a candidate for being a min, max or min-max compared to the current min and max according to the current
        // feature_index
        const auto min_max_feature_candidate = samples_range_first[sample_index * n_features + feature_index];

        if (min_max_feature_candidate < bounding_box_1d.first) {
            bounding_box_1d.first = min_max_feature_candidate;
        }
        if (min_max_feature_candidate > bounding_box_1d.second) {
            bounding_box_1d.second = min_max_feature_candidate;
        }
    }
    return bounding_box_1d;
}

template <typename IndicesIterator, typename SamplesIterator>
auto make_1d_bounding_box(const IndicesIterator& indices_range_first,
                          const IndicesIterator& indices_range_last,
                          const SamplesIterator& samples_range_first,
                          const SamplesIterator& samples_range_last,
                          std::size_t            n_features,
                          ssize_t                feature_index) {
    using DataType = DataType<SamplesIterator>;

    common::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    auto bounding_box_1d =
        RangeType<SamplesIterator>(std::numeric_limits<DataType>::max(), std::numeric_limits<DataType>::lowest());

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // a candidate for being a min, max or min-max compared to the current min and max according to the current
        // feature_index
        const auto min_max_feature_candidate =
            samples_range_first[indices_range_first[sample_index] * n_features + feature_index];

        if (min_max_feature_candidate < bounding_box_1d.first) {
            bounding_box_1d.first = min_max_feature_candidate;
        }
        if (min_max_feature_candidate > bounding_box_1d.second) {
            bounding_box_1d.second = min_max_feature_candidate;
        }
    }
    return bounding_box_1d;
}

template <typename SamplesIterator>
auto make_kd_bounding_box(const SamplesIterator& samples_range_first,
                          const SamplesIterator& samples_range_last,
                          std::size_t            n_features) {
    using DataType = DataType<SamplesIterator>;

    const std::size_t n_samples = common::get_n_samples(samples_range_first, samples_range_last, n_features);

    // min max elements per feature vector
    auto kd_bounding_box = HyperRangeType<SamplesIterator>(
        n_features,
        RangeType<SamplesIterator>(std::numeric_limits<DataType>::max(), std::numeric_limits<DataType>::lowest()));

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            // a candidate for being a min, max or min-max compared to the current min and max according to the current
            // feature_index
            const auto min_max_feature_candidate = samples_range_first[sample_index * n_features + feature_index];

            if (min_max_feature_candidate < kd_bounding_box[feature_index].first) {
                kd_bounding_box[feature_index].first = min_max_feature_candidate;
            }
            if (min_max_feature_candidate > kd_bounding_box[feature_index].second) {
                kd_bounding_box[feature_index].second = min_max_feature_candidate;
            }
        }
    }
    return kd_bounding_box;
}

template <typename IndicesIterator, typename SamplesIterator>
auto make_kd_bounding_box(const IndicesIterator& indices_range_first,
                          const IndicesIterator& indices_range_last,
                          const SamplesIterator& samples_range_first,
                          const SamplesIterator& samples_range_last,
                          std::size_t            n_features) {
    using DataType = DataType<SamplesIterator>;

    common::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    // min max elements per feature vector
    auto kd_bounding_box = HyperRangeType<SamplesIterator>(
        n_features,
        RangeType<SamplesIterator>(std::numeric_limits<DataType>::max(), std::numeric_limits<DataType>::lowest()));

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            // a candidate for being a min, max or min-max compared to the current min and max according to the current
            // feature_index
            const auto min_max_feature_candidate =
                samples_range_first[indices_range_first[sample_index] * n_features + feature_index];

            if (min_max_feature_candidate < kd_bounding_box[feature_index].first) {
                kd_bounding_box[feature_index].first = min_max_feature_candidate;
            }
            if (min_max_feature_candidate > kd_bounding_box[feature_index].second) {
                kd_bounding_box[feature_index].second = min_max_feature_candidate;
            }
        }
    }
    return kd_bounding_box;
}

template <typename SamplesIterator>
bool is_sample_in_kd_bounding_box(const SamplesIterator&                 feature_first,
                                  const SamplesIterator&                 feature_last,
                                  const HyperRangeType<SamplesIterator>& kd_bounding_box) {
    const std::size_t n_features = std::distance(feature_first, feature_last);

    for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
        // A sample is inside the bounding box if p is in [lo, hi]
        if (feature_first[feature_index] < kd_bounding_box[feature_index].first ||
            feature_first[feature_index] > kd_bounding_box[feature_index].second) {
            return false;
        }
    }
    return true;
}

template <typename SamplesIterator>
auto relative_to_absolute_coordinates(const SamplesIterator&                 feature_first,
                                      const SamplesIterator&                 feature_last,
                                      const HyperRangeType<SamplesIterator>& relative_coordinates_sequence) {
    const std::size_t n_features = std::distance(feature_first, feature_last);

    // make a copy that will be the translated version
    auto range_bounding_box = relative_coordinates_sequence;

    for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
        // get the 1D bounding box (or range) w.r.t. the current dimension
        auto& range = range_bounding_box[feature_index];
        // shift it by the amount specified at the right dimension
        range.first += feature_first[feature_index];
        range.second += feature_first[feature_index];
    }
    return range_bounding_box;
}

}  // namespace ffcl::datastruct::bbox