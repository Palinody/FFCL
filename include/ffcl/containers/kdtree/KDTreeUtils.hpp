#pragma once

#include "ffcl/algorithms/Sorting.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/math/random/Sampling.hpp"
#include "ffcl/math/statistics/Statistics.hpp"

#include <sys/types.h>  // ssize_t
#include <cmath>
#include <limits>
#include <vector>

template <typename Iterator>
using IteratorPairType = std::pair<Iterator, Iterator>;

template <typename Iterator>
using DataType = typename Iterator::value_type;

template <typename Iterator>
using BoundingBox1DType = std::pair<DataType<Iterator>, DataType<Iterator>>;

template <typename Iterator>
using BoundingBoxKDType = std::vector<BoundingBox1DType<Iterator>>;

namespace kdtree::utils {

template <typename Iterator>
BoundingBoxKDType<Iterator> make_1d_bounding_box(const Iterator& samples_first,
                                                 const Iterator& samples_last,
                                                 std::size_t     n_features,
                                                 ssize_t         axis) {
    using DataType = DataType<Iterator>;

    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

    auto bounding_box_1d =
        BoundingBox1DType<Iterator>({std::numeric_limits<DataType>::max(), std::numeric_limits<DataType>::lowest()});

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        const auto min_max_feature_candidate = samples_first[sample_index * n_features + axis];

        if (min_max_feature_candidate < bounding_box_1d.first) {
            bounding_box_1d.first = min_max_feature_candidate;
        }
        if (min_max_feature_candidate > bounding_box_1d.second) {
            bounding_box_1d.second = min_max_feature_candidate;
        }
    }
    return bounding_box_1d;
}

template <typename Iterator>
BoundingBoxKDType<Iterator> make_kd_bounding_box(const Iterator& samples_first,
                                                 const Iterator& samples_last,
                                                 std::size_t     n_features) {
    using DataType = DataType<Iterator>;

    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

    auto kd_bounding_box = BoundingBoxKDType<Iterator>(
        n_features,
        BoundingBox1DType<Iterator>({std::numeric_limits<DataType>::max(), std::numeric_limits<DataType>::lowest()}));

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            const auto min_max_feature_candidate = samples_first[sample_index * n_features + feature_index];

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

template <typename Iterator>
ssize_t select_axis_with_largest_bounding_box_difference(const BoundingBoxKDType<Iterator>& kd_bounding_box) {
    const auto cmp = [](const auto& lhs, const auto& rhs) {
        return std::abs(lhs.first - lhs.second) < std::abs(rhs.first - rhs.second);
    };
    auto it = std::max_element(kd_bounding_box.begin(), kd_bounding_box.end(), cmp);
    return std::distance(kd_bounding_box.begin(), it);
}

template <typename Iterator>
ssize_t select_axis_with_largest_variance(const Iterator& samples_first,
                                          const Iterator& samples_last,
                                          std::size_t     n_features,
                                          double          n_samples_fraction) {
    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);
    // set the number of samples as a fraction of the input
    const std::size_t n_choices = n_samples_fraction * n_samples;
    // select the axis based on variance only if the number of selected samples is greater than 2
    if (n_choices > 2) {
        const auto random_samples =
            math::random::select_n_random_samples(samples_first, samples_last, n_features, n_choices);

        // return the feature index with the maximum variance
        return math::statistics::argmax_variance_per_feature(random_samples.begin(), random_samples.end(), n_features);
    }
    // otherwise apply the bounding box method
    const auto kd_bounding_box = make_kd_bounding_box(samples_first, samples_last, n_features);

    return select_axis_with_largest_bounding_box_difference<Iterator>(kd_bounding_box);
}

template <typename RandomAccessIterator>
std::tuple<std::size_t,
           IteratorPairType<RandomAccessIterator>,
           IteratorPairType<RandomAccessIterator>,
           IteratorPairType<RandomAccessIterator>>
quickselect_median_range(RandomAccessIterator samples_first,
                         RandomAccessIterator samples_last,
                         std::size_t          n_features,
                         std::size_t          feature_index) {
    assert(feature_index < n_features);

    const auto median_index = common::utils::get_n_samples(samples_first, samples_last, n_features) / 2;

    const auto median_range =
        ffcl::algorithms::quickselect_range(samples_first, samples_last, n_features, median_index, feature_index);
    // all the points at the left of the pivot point
    const auto left_range = std::make_pair(samples_first, samples_first + median_index * n_features);

    // all the points at the right of the pivot point
    const auto right_range = std::make_pair(samples_first + median_index * n_features + n_features, samples_last);

    return {median_index, left_range, median_range, right_range};
}

}  // namespace kdtree::utils