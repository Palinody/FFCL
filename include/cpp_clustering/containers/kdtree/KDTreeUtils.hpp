#pragma once

#include "cpp_clustering/common/Utils.hpp"

#include <sys/types.h>  // ssize_t
#include <cmath>
#include <limits>
#include <vector>

namespace kdtree::utils {

template <typename Iterator>
using DataType = typename Iterator::value_type;

template <typename Iterator>
using BoundingBox1DType = std::pair<DataType<Iterator>, DataType<Iterator>>;

template <typename Iterator>
using BoundingBoxKDType = std::vector<BoundingBox1DType<Iterator>>;

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
        const auto min_max_feature_candidate = *(samples_first + sample_index * n_features + axis);

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
            const auto min_max_feature_candidate = *(samples_first + sample_index * n_features + feature_index);

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
    using DataType = DataType<Iterator>;

    const std::size_t n_features = kd_bounding_box.size();

    DataType absolute_difference = std::abs(kd_bounding_box[0].first - kd_bounding_box[0].second);
    ssize_t  axis                = 0;

    for (std::size_t feature_index = 1; feature_index < n_features; ++feature_index) {
        const auto absolute_difference_candidate =
            std::abs(kd_bounding_box[feature_index].first - kd_bounding_box[feature_index].second);

        if (absolute_difference_candidate > absolute_difference) {
            absolute_difference = absolute_difference_candidate;
            axis                = feature_index;
        }
    }
    return axis;
}

template <typename Iterator>
ssize_t select_axis_with_largest_variance(const Iterator& samples_first,
                                          const Iterator& samples_last,
                                          std::size_t     n_features,
                                          double          n_samples_fraction) {
    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);
    // compute the variance if n_samples >= min_samples and make at most n_choices random selections. If n_samples
    // becomes less than n_choices then n_samples choices will be used.
    const std::size_t n_choices = n_samples_fraction * n_samples;
    // select the axis based on variance only if the number the number of selected samples will be more than 2
    if (n_choices > 2) {
        const auto random_samples =
            common::utils::select_n_random_samples(samples_first, samples_last, n_features, n_choices);

        const auto variance_per_feature =
            common::utils::compute_variance_per_feature(random_samples.begin(), random_samples.end(), n_features);
        // return the feature index with the maximum variance
        return common::utils::argmax(variance_per_feature.begin(), variance_per_feature.end());
    }
    // otherwise apply the bounding box method
    const auto kd_bounding_box = make_kd_bounding_box(samples_first, samples_last, n_features);

    return select_axis_with_largest_bounding_box_difference<Iterator>(kd_bounding_box);
}

template <typename RandomAccessIterator>
std::pair<RandomAccessIterator, RandomAccessIterator> quickselect_median_range(RandomAccessIterator samples_first,
                                                                               RandomAccessIterator samples_last,
                                                                               std::size_t          n_features,
                                                                               std::size_t comparison_feature_index) {
    assert(comparison_feature_index < n_features);

    const auto median_index = common::utils::get_n_samples(samples_first, samples_last, n_features) / 2;

    return common::utils::quickselect_range(
        samples_first,
        samples_last,
        median_index,
        n_features,
        [comparison_feature_index](const auto& range1_first, const auto& range2_first) {
            // assumes that:
            //   * both ranges have length: n_features
            //   * comparison_feature_index in range [0, n_features)
            return *(range1_first + comparison_feature_index) < *(range2_first + comparison_feature_index);
        });
}

}  // namespace kdtree::utils