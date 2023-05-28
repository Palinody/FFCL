#pragma once

#include "ffcl/algorithms/Sorting.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/math/random/Distributions.hpp"
#include "ffcl/math/random/Sampling.hpp"
#include "ffcl/math/statistics/Statistics.hpp"

#include <sys/types.h>  // ssize_t
#include <cmath>
#include <limits>
#include <vector>

template <typename SamplesIterator>
using IteratorPairType = std::pair<SamplesIterator, SamplesIterator>;

template <typename SamplesIterator>
using DataType = typename SamplesIterator::value_type;

template <typename SamplesIterator>
using BoundingBox1DType = std::pair<DataType<SamplesIterator>, DataType<SamplesIterator>>;

template <typename SamplesIterator>
using BoundingBoxKDType = std::vector<BoundingBox1DType<SamplesIterator>>;

namespace kdtree::algorithms {

template <typename SamplesIterator>
auto make_1d_bounding_box(const SamplesIterator& samples_first,
                          const SamplesIterator& samples_last,
                          std::size_t            n_features,
                          ssize_t                feature_index) {
    using DataType = DataType<SamplesIterator>;

    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

    auto bounding_box_1d = BoundingBox1DType<SamplesIterator>(std::numeric_limits<DataType>::max(),
                                                              std::numeric_limits<DataType>::lowest());

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // a candidate for being a min, max or min-max compared to the current min and max according to the current
        // feature_index
        const auto min_max_feature_candidate = samples_first[sample_index * n_features + feature_index];

        if (min_max_feature_candidate < bounding_box_1d.first) {
            bounding_box_1d.first = min_max_feature_candidate;
        }
        if (min_max_feature_candidate > bounding_box_1d.second) {
            bounding_box_1d.second = min_max_feature_candidate;
        }
    }
    return bounding_box_1d;
}

template <typename RandomAccessIntIterator, typename SamplesIterator>
auto make_1d_bounding_box(const RandomAccessIntIterator& index_first,
                          const RandomAccessIntIterator& index_last,
                          const SamplesIterator&         samples_first,
                          const SamplesIterator&         samples_last,
                          std::size_t                    n_features,
                          ssize_t                        feature_index) {
    using DataType = DataType<SamplesIterator>;

    common::utils::ignore_parameters(samples_last);

    const std::size_t n_samples = std::distance(index_first, index_last);

    auto bounding_box_1d = BoundingBox1DType<SamplesIterator>(std::numeric_limits<DataType>::max(),
                                                              std::numeric_limits<DataType>::lowest());

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // a candidate for being a min, max or min-max compared to the current min and max according to the current
        // feature_index
        const auto min_max_feature_candidate = samples_first[index_first[sample_index] * n_features + feature_index];

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
BoundingBoxKDType<SamplesIterator> make_kd_bounding_box(const SamplesIterator& samples_first,
                                                        const SamplesIterator& samples_last,
                                                        std::size_t            n_features) {
    using DataType = DataType<SamplesIterator>;

    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

    // min max elements per feature vector
    auto kd_bounding_box =
        BoundingBoxKDType<SamplesIterator>(n_features,
                                           BoundingBox1DType<SamplesIterator>(std::numeric_limits<DataType>::max(),
                                                                              std::numeric_limits<DataType>::lowest()));

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            // a candidate for being a min, max or min-max compared to the current min and max according to the current
            // feature_index
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

template <typename RandomAccessIntIterator, typename SamplesIterator>
BoundingBoxKDType<SamplesIterator> make_kd_bounding_box(const RandomAccessIntIterator& index_first,
                                                        const RandomAccessIntIterator& index_last,
                                                        const SamplesIterator&         samples_first,
                                                        const SamplesIterator&         samples_last,
                                                        std::size_t                    n_features) {
    using DataType = DataType<SamplesIterator>;

    common::utils::ignore_parameters(samples_last);

    const std::size_t n_samples = std::distance(index_first, index_last);

    // min max elements per feature vector
    auto kd_bounding_box =
        BoundingBoxKDType<SamplesIterator>(n_features,
                                           BoundingBox1DType<SamplesIterator>(std::numeric_limits<DataType>::max(),
                                                                              std::numeric_limits<DataType>::lowest()));

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            // a candidate for being a min, max or min-max compared to the current min and max according to the current
            // feature_index
            const auto min_max_feature_candidate =
                samples_first[index_first[sample_index] * n_features + feature_index];

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
ssize_t select_axis_with_largest_bounding_box_difference(const BoundingBoxKDType<SamplesIterator>& kd_bounding_box) {
    const auto cmp = [](const auto& lhs, const auto& rhs) {
        return common::utils::abs(lhs.first - lhs.second) < common::utils::abs(rhs.first - rhs.second);
    };
    const auto it = std::max_element(kd_bounding_box.begin(), kd_bounding_box.end(), cmp);
    return std::distance(kd_bounding_box.begin(), it);
}

template <typename SamplesIterator>
ssize_t select_axis_with_largest_variance(const SamplesIterator& samples_first,
                                          const SamplesIterator& samples_last,
                                          std::size_t            n_features,
                                          double                 n_samples_fraction) {
    assert(n_samples_fraction >= 0 && n_samples_fraction <= 1);

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
    // else if the number of samples is greater than or equal to the minimum number of samples to compute the variance,
    // compute the variance with the minimum number of samples, which is 3
    if (n_samples > 2) {
        const auto random_samples = math::random::select_n_random_samples(samples_first, samples_last, n_features, 3);
        // return the feature index with the maximum variance
        return math::statistics::argmax_variance_per_feature(random_samples.begin(), random_samples.end(), n_features);
    }
    // select the axis according to the dimension with the most spread between 2 points
    if (n_samples == 2) {
        // otherwise apply the bounding box method
        const auto kd_bounding_box = make_kd_bounding_box(samples_first, samples_last, n_features);
        return select_axis_with_largest_bounding_box_difference<SamplesIterator>(kd_bounding_box);
    }
    // return a random axis if theres only one sample left (should never be called if leaves cannot be empty)
    return math::random::uniform_distribution<ssize_t>(0, n_features - 1)();
}

template <typename RandomAccessIntIterator, typename SamplesIterator>
ssize_t select_axis_with_largest_variance(const RandomAccessIntIterator& index_first,
                                          const RandomAccessIntIterator& index_last,
                                          const SamplesIterator&         samples_first,
                                          const SamplesIterator&         samples_last,
                                          std::size_t                    n_features,
                                          double                         n_samples_fraction) {
    assert(n_samples_fraction >= 0 && n_samples_fraction <= 1);

    const std::size_t n_samples = std::distance(index_first, index_last);
    // set the number of samples as a fraction of the input
    const std::size_t n_choices = n_samples_fraction * n_samples;
    // select the axis based on variance only if the number of selected samples is greater than 2
    if (n_choices > 2) {
        const auto random_samples = math::random::select_n_random_samples_from_indices(
            index_first, index_last, samples_first, samples_last, n_features, n_choices);
        // return the feature index with the maximum variance
        return math::statistics::argmax_variance_per_feature(random_samples.begin(), random_samples.end(), n_features);
    }
    // else if the number of samples is greater than or equal to the minimum number of samples to compute the variance,
    // compute the variance with the minimum number of samples, which is 3
    if (n_samples > 2) {
        const auto random_samples = math::random::select_n_random_samples_from_indices(
            index_first, index_last, samples_first, samples_last, n_features, 3);
        // return the feature index with the maximum variance
        return math::statistics::argmax_variance_per_feature(random_samples.begin(), random_samples.end(), n_features);
    }
    // select the axis according to the dimension with the most spread between 2 points
    if (n_samples == 2) {
        // otherwise apply the bounding box method
        const auto kd_bounding_box = make_kd_bounding_box(samples_first, samples_last, n_features);
        return select_axis_with_largest_bounding_box_difference<SamplesIterator>(kd_bounding_box);
    }
    // return a random axis if theres only one sample left (should never be called if leaves cannot be empty)
    return math::random::uniform_distribution<ssize_t>(0, n_features - 1)();
}

template <typename SamplesIterator>
std::tuple<std::size_t,
           IteratorPairType<SamplesIterator>,
           IteratorPairType<SamplesIterator>,
           IteratorPairType<SamplesIterator>>
quickselect_median_range(SamplesIterator samples_first,
                         SamplesIterator samples_last,
                         std::size_t     n_features,
                         std::size_t     feature_index) {
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

template <typename RandomAccessIntIterator, typename SamplesIterator>
std::tuple<std::size_t,
           IteratorPairType<RandomAccessIntIterator>,
           IteratorPairType<RandomAccessIntIterator>,
           IteratorPairType<RandomAccessIntIterator>>
shift_median_to_leftmost_equal_value(std::size_t&                              median_index,
                                     IteratorPairType<RandomAccessIntIterator> left_indices_range,
                                     IteratorPairType<RandomAccessIntIterator> median_indices_range,
                                     IteratorPairType<RandomAccessIntIterator> right_indices_range,
                                     SamplesIterator                           samples_first,
                                     SamplesIterator                           samples_last,
                                     std::size_t                               n_features,
                                     std::size_t                               feature_index) {
    common::utils::ignore_parameters(samples_last);

    const auto left_range_length = std::distance(left_indices_range.first, left_indices_range.second);
    // return if the left range is empty because no left shift is possible
    if (!left_range_length) {
        return {median_index, left_indices_range, median_indices_range, right_indices_range};
    }
    // the target value of the median
    const auto cut_value = samples_first[median_indices_range.first[0] * n_features + feature_index];
    // the left range iterator at the value the current pointer will be compared to at each iteration
    auto left_neighbor_value_it = left_indices_range.second - 1;

    // decrement the iterators while the left range isnt empty and the neighbor value at the left of the median is still
    // equal to the cut value at the corresponding feature index
    while (std::distance(left_indices_range.first, left_neighbor_value_it) >= 0 &&
           common::utils::equality(samples_first[left_neighbor_value_it[0] * n_features + feature_index], cut_value)) {
        --left_neighbor_value_it;
        --median_indices_range.first;
    }
    // update the ranges accordingly
    left_indices_range.second   = median_indices_range.first;
    median_indices_range.second = median_indices_range.first + 1;
    right_indices_range.first   = median_indices_range.second;
    median_index                = std::distance(left_indices_range.first, median_indices_range.first);

    return {median_index, left_indices_range, median_indices_range, right_indices_range};
}

template <typename RandomAccessIntIterator, typename SamplesIterator>
std::tuple<std::size_t,
           IteratorPairType<RandomAccessIntIterator>,
           IteratorPairType<RandomAccessIntIterator>,
           IteratorPairType<RandomAccessIntIterator>>
quickselect_median_indexed_range(RandomAccessIntIterator index_first,
                                 RandomAccessIntIterator index_last,
                                 SamplesIterator         samples_first,
                                 SamplesIterator         samples_last,
                                 std::size_t             n_features,
                                 std::size_t             feature_index) {
    assert(feature_index < n_features);

    std::size_t median_index = std::distance(index_first, index_last) / 2;

    const auto median_indices_range = ffcl::algorithms::quickselect_indexed_range(
        index_first, index_last, samples_first, samples_last, n_features, median_index, feature_index);

    // all the points at the left of the pivot point
    const auto left_indices_range = std::make_pair(index_first, index_first + median_index);

    // all the points at the right of the pivot point
    const auto right_indices_range = std::make_pair(index_first + median_index + 1, index_last);

    return shift_median_to_leftmost_equal_value(median_index,
                                                left_indices_range,
                                                median_indices_range,
                                                right_indices_range,
                                                samples_first,
                                                samples_last,
                                                n_features,
                                                feature_index);
}

}  // namespace kdtree::algorithms