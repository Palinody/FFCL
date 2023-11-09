#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/algorithms/Sorting.hpp"
#include "ffcl/common/math/random/Distributions.hpp"
#include "ffcl/common/math/random/Sampling.hpp"
#include "ffcl/common/math/statistics/Statistics.hpp"
#include "ffcl/datastruct/BoundingBox.hpp"

#include <sys/types.h>  // ssize_t
#include <cmath>
#include <limits>
#include <vector>

namespace kdtree::algorithms {

template <typename SamplesIterator>
ssize_t select_axis_with_largest_bounding_box_difference(
    const ffcl::bbox::HyperRangeType<SamplesIterator>& kd_bounding_box) {
    const auto comparison = [](const auto& lhs, const auto& rhs) {
        return ffcl::common::utils::abs(lhs.first - lhs.second) < ffcl::common::utils::abs(rhs.first - rhs.second);
    };
    const auto it = std::max_element(kd_bounding_box.begin(), kd_bounding_box.end(), comparison);
    return std::distance(kd_bounding_box.begin(), it);
}

template <typename SamplesIterator>
ssize_t select_axis_with_largest_bounding_box_difference(
    const ffcl::bbox::HyperRangeType<SamplesIterator>& kd_bounding_box,
    const std::vector<std::size_t>&                    feature_mask) {
    using DataType = typename SamplesIterator::value_type;

    // feature index of the greatest range dimension in the kd bounding box at the indices of the feature_mask
    std::size_t current_max_range_feature_index = 0;
    DataType    current_max_range               = 0;

    // "feature_index"s are only the ones specified in the feature_mask
    for (std::size_t feature_index : feature_mask) {
        const auto max_range_candidate =
            ffcl::common::utils::abs(kd_bounding_box[feature_index].first - kd_bounding_box[feature_index].second);

        if (max_range_candidate > current_max_range) {
            current_max_range_feature_index = feature_index;
            current_max_range               = max_range_candidate;
        }
    }
    return current_max_range_feature_index;
}

template <typename IndicesIterator, typename SamplesIterator>
ssize_t select_axis_with_largest_variance(const IndicesIterator& indices_range_first,
                                          const IndicesIterator& indices_range_last,
                                          const SamplesIterator& samples_range_first,
                                          const SamplesIterator& samples_range_last,
                                          std::size_t            n_features,
                                          double                 n_samples_rate) {
    assert(n_samples_rate >= 0 && n_samples_rate <= 1);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    // set the number of samples as a fraction of the input
    const std::size_t n_choices = n_samples_rate * n_samples;

    // select the axis based on variance only if the number of selected samples is greater than 2
    if (n_choices > 2) {
        const auto random_indices = ffcl::common::math::random::select_n_random_indices_from_indices(
            indices_range_first, indices_range_last, n_choices);
        // return the feature index with the maximum variance
        return ffcl::common::math::statistics::argmax_variance_per_feature(
            random_indices.begin(), random_indices.end(), samples_range_first, samples_range_last, n_features);
    }
    // else if the number of samples is greater than or equal to the minimum number of samples to compute the variance,
    // compute the variance with the minimum number of samples, which is 3
    if (n_samples > 2) {
        const auto random_indices = ffcl::common::math::random::select_n_random_indices_from_indices(
            indices_range_first, indices_range_last, 3);
        // return the feature index with the maximum variance
        return ffcl::common::math::statistics::argmax_variance_per_feature(
            random_indices.begin(), random_indices.end(), samples_range_first, samples_range_last, n_features);
    }
    // select the axis according to the dimension with the most spread between 2 points
    if (n_samples == 2) {
        // otherwise apply the bounding box method
        const auto kd_bounding_box =
            ffcl::bbox::make_kd_bounding_box(samples_range_first, samples_range_last, n_features);
        return select_axis_with_largest_bounding_box_difference<SamplesIterator>(kd_bounding_box);
    }
    // return a random axis if theres only one sample left
    return ffcl::common::math::random::uniform_distribution<ssize_t>(0, n_features - 1)();
}

template <typename IndicesIterator, typename SamplesIterator>
ssize_t select_axis_with_largest_variance(const IndicesIterator&          indices_range_first,
                                          const IndicesIterator&          indices_range_last,
                                          const SamplesIterator&          samples_range_first,
                                          const SamplesIterator&          samples_range_last,
                                          std::size_t                     n_features,
                                          double                          n_samples_rate,
                                          const std::vector<std::size_t>& feature_mask) {
    assert(0 <= n_samples_rate && n_samples_rate <= 1);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    // set the number of samples as a fraction of the input
    const std::size_t n_choices = n_samples_rate * n_samples;

    // select the axis based on variance only if the number of selected samples is greater than 2
    if (n_choices > 2) {
        const auto random_indices = ffcl::common::math::random::select_n_random_indices_from_indices(
            indices_range_first, indices_range_last, n_choices);
        // return the feature index with the maximum variance
        return ffcl::common::math::statistics::argmax_variance_per_feature(random_indices.begin(),
                                                                           random_indices.end(),
                                                                           samples_range_first,
                                                                           samples_range_last,
                                                                           n_features,
                                                                           feature_mask);
    }
    // else if the number of samples is greater than or equal to the minimum number of samples to compute the variance,
    // compute the variance with the minimum number of samples, which is 3
    if (n_samples > 2) {
        const auto random_indices = ffcl::common::math::random::select_n_random_indices_from_indices(
            indices_range_first, indices_range_last, 3);
        // return the feature index with the maximum variance
        return ffcl::common::math::statistics::argmax_variance_per_feature(random_indices.begin(),
                                                                           random_indices.end(),
                                                                           samples_range_first,
                                                                           samples_range_last,
                                                                           n_features,
                                                                           feature_mask);
    }
    // select the axis according to the dimension with the most spread between 2 points
    if (n_samples == 2) {
        // otherwise apply the bounding box method
        const auto kd_bounding_box =
            ffcl::bbox::make_kd_bounding_box(samples_range_first, samples_range_last, n_features);
        return select_axis_with_largest_bounding_box_difference<SamplesIterator>(kd_bounding_box, feature_mask);
    }
    // return a random axis if theres only one sample left
    return ffcl::common::math::random::uniform_distribution<ssize_t>(0, n_features - 1)();
}

template <typename IndicesIterator, typename SamplesIterator>
std::tuple<std::size_t,
           ffcl::bbox::IteratorPairType<IndicesIterator>,
           ffcl::bbox::IteratorPairType<IndicesIterator>,
           ffcl::bbox::IteratorPairType<IndicesIterator>>
shift_median_to_leftmost_equal_value(std::size_t                                   median_index,
                                     ffcl::bbox::IteratorPairType<IndicesIterator> left_indices_range,
                                     ffcl::bbox::IteratorPairType<IndicesIterator> median_indices_range,
                                     ffcl::bbox::IteratorPairType<IndicesIterator> right_indices_range,
                                     const SamplesIterator&                        samples_range_first,
                                     const SamplesIterator&                        samples_range_last,
                                     std::size_t                                   n_features,
                                     std::size_t                                   feature_index) {
    ffcl::common::utils::ignore_parameters(samples_range_last);

    const auto left_range_length = std::distance(left_indices_range.first, left_indices_range.second);

    // return if the left range is empty because no left shift is possible
    if (!left_range_length) {
        return std::make_tuple(median_index, left_indices_range, median_indices_range, right_indices_range);
    }
    // the target value of the median
    const auto cut_value = samples_range_first[median_indices_range.first[0] * n_features + feature_index];

    // the left range iterator at the value the current pointer will be compared to at each iteration
    auto left_neighbor_value_it = left_indices_range.second - 1;

    // decrement the iterators while the left range isnt empty and the neighbor value at the left of the median is still
    // equal to the cut value at the corresponding feature index
    while (std::distance(left_indices_range.first, left_neighbor_value_it) >= 0 &&
           ffcl::common::utils::equality(samples_range_first[left_neighbor_value_it[0] * n_features + feature_index],
                                         cut_value)) {
        --left_neighbor_value_it;
        --median_indices_range.first;
    }
    // update the ranges accordingly
    left_indices_range.second   = median_indices_range.first;
    median_indices_range.second = median_indices_range.first + 1;
    right_indices_range.first   = median_indices_range.second;
    median_index                = std::distance(left_indices_range.first, median_indices_range.first);

    return std::make_tuple(median_index, left_indices_range, median_indices_range, right_indices_range);
}

template <typename IndicesIterator, typename SamplesIterator>
std::tuple<std::size_t,
           ffcl::bbox::IteratorPairType<IndicesIterator>,
           ffcl::bbox::IteratorPairType<IndicesIterator>,
           ffcl::bbox::IteratorPairType<IndicesIterator>>
quickselect_median(const IndicesIterator& indices_range_first,
                   const IndicesIterator& indices_range_last,
                   const SamplesIterator& samples_range_first,
                   const SamplesIterator& samples_range_last,
                   std::size_t            n_features,
                   std::size_t            feature_index) {
    assert(feature_index < n_features);

    std::size_t median_index = std::distance(indices_range_first, indices_range_last) / 2;

    const auto median_indices_range = ffcl::common::algorithms::quickselect(indices_range_first,
                                                                            indices_range_last,
                                                                            samples_range_first,
                                                                            samples_range_last,
                                                                            n_features,
                                                                            median_index,
                                                                            feature_index);

    // all the points at the left of the pivot point
    const auto left_indices_range = std::make_pair(indices_range_first, indices_range_first + median_index);

    // all the points at the right of the pivot point
    const auto right_indices_range = std::make_pair(indices_range_first + median_index + 1, indices_range_last);

    return shift_median_to_leftmost_equal_value(median_index,
                                                left_indices_range,
                                                median_indices_range,
                                                right_indices_range,
                                                samples_range_first,
                                                samples_range_last,
                                                n_features,
                                                feature_index);
}

}  // namespace kdtree::algorithms