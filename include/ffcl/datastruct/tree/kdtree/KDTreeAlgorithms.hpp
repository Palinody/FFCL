#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/algorithms/Sorting.hpp"
#include "ffcl/common/math/random/Distributions.hpp"
#include "ffcl/common/math/random/Sampling.hpp"
#include "ffcl/common/math/statistics/Statistics.hpp"
#include "ffcl/datastruct/Interval.hpp"

#include <cmath>
#include <cstddef>  // std::size_t
#include <limits>
#include <vector>

namespace ffcl::datastruct::kdtree::algorithms {

template <typename SamplesIterator>
std::size_t select_axis_with_largest_bounding_box_difference(const HyperInterval<SamplesIterator>& hyper_interval) {
    const auto comparison = [](const auto& lhs, const auto& rhs) {
        return common::abs(lhs.first() - lhs.second()) < common::abs(rhs.first() - rhs.second());
    };
    const auto it = std::max_element(hyper_interval.begin(), hyper_interval.end(), comparison);
    return std::distance(hyper_interval.begin(), it);
}

template <typename SamplesIterator>
std::size_t select_axis_with_largest_bounding_box_difference(const HyperInterval<SamplesIterator>& hyper_interval,
                                                             const std::vector<std::size_t>&       feature_mask) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    // feature index of the greatest range dimension in the kd bounding box at the indices of the feature_mask
    std::size_t current_max_range_feature_index = 0;
    DataType    current_max_range               = 0;

    // "feature_index"s are only the ones specified in the feature_mask
    for (std::size_t feature_index : feature_mask) {
        const auto max_range_candidate =
            common::abs(hyper_interval[feature_index].second() - hyper_interval[feature_index].first());

        if (max_range_candidate > current_max_range) {
            current_max_range_feature_index = feature_index;
            current_max_range               = max_range_candidate;
        }
    }
    return current_max_range_feature_index;
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t select_axis_with_largest_variance(const IndicesIterator& indices_range_first,
                                              const IndicesIterator& indices_range_last,
                                              const SamplesIterator& samples_range_first,
                                              const SamplesIterator& samples_range_last,
                                              std::size_t            n_features,
                                              double                 n_samples_rate) {
    assert(n_samples_rate >= 0 && n_samples_rate <= 1);

    static_assert(common::is_iterator<IndicesIterator>::value, "IndicesIterator is not an iterator");
    static_assert(common::is_iterator<SamplesIterator>::value, "SamplesIterator is not an iterator");

    using IndexType = typename std::iterator_traits<IndicesIterator>::value_type;
    using DataType  = typename std::iterator_traits<SamplesIterator>::value_type;

    static_assert(std::is_integral_v<IndexType>, "IndexType must be integer.");
    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    // set the number of samples as a fraction of the input
    const std::size_t n_choices = n_samples_rate * n_samples;

    // select the axis based on variance only if the number of selected samples is greater than 2
    if (n_choices > 2) {
        const auto random_indices =
            common::math::random::select_n_elements(indices_range_first, indices_range_last, n_choices);
        // return the feature index with the maximum variance
        return common::math::statistics::argmax_variance_per_feature(
            random_indices.begin(), random_indices.end(), samples_range_first, samples_range_last, n_features);
    }
    // else if the number of samples is greater than or equal to the minimum number of samples to compute the variance,
    // compute the variance with the minimum number of samples, which is 3
    if (n_samples > 2) {
        const auto random_indices = common::math::random::select_n_elements(indices_range_first, indices_range_last, 3);
        // return the feature index with the maximum variance
        return common::math::statistics::argmax_variance_per_feature(
            random_indices.begin(), random_indices.end(), samples_range_first, samples_range_last, n_features);
    }
    // select the axis according to the dimension with the most spread between 2 points
    if (n_samples == 2) {
        // otherwise apply the bounding box method
        const auto hyper_interval = make_hyper_interval(samples_range_first, samples_range_last, n_features);
        return select_axis_with_largest_bounding_box_difference<SamplesIterator>(hyper_interval);
    }
    // return a random axis if theres only one sample left
    return common::math::random::uniform_distribution<std::size_t>(0, n_features - 1)();
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t select_axis_with_largest_variance(const IndicesIterator&          indices_range_first,
                                              const IndicesIterator&          indices_range_last,
                                              const SamplesIterator&          samples_range_first,
                                              const SamplesIterator&          samples_range_last,
                                              std::size_t                     n_features,
                                              double                          n_samples_rate,
                                              const std::vector<std::size_t>& feature_mask) {
    assert(0 <= n_samples_rate && n_samples_rate <= 1);

    static_assert(common::is_iterator<IndicesIterator>::value, "IndicesIterator is not an iterator");
    static_assert(common::is_iterator<SamplesIterator>::value, "SamplesIterator is not an iterator");

    using IndexType = typename std::iterator_traits<IndicesIterator>::value_type;
    using DataType  = typename std::iterator_traits<SamplesIterator>::value_type;

    static_assert(std::is_integral_v<IndexType>, "IndexType must be integer.");
    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    // set the number of samples as a fraction of the input
    const std::size_t n_choices = n_samples_rate * n_samples;

    // select the axis based on variance only if the number of selected samples is greater than 2
    if (n_choices > 2) {
        const auto random_indices =
            common::math::random::select_n_elements(indices_range_first, indices_range_last, n_choices);
        // return the feature index with the maximum variance
        return common::math::statistics::argmax_variance_per_feature(random_indices.begin(),
                                                                     random_indices.end(),
                                                                     samples_range_first,
                                                                     samples_range_last,
                                                                     n_features,
                                                                     feature_mask);
    }
    // else if the number of samples is greater than or equal to the minimum number of samples to compute the variance,
    // compute the variance with the minimum number of samples, which is 3
    if (n_samples > 2) {
        const auto random_indices = common::math::random::select_n_elements(indices_range_first, indices_range_last, 3);
        // return the feature index with the maximum variance
        return common::math::statistics::argmax_variance_per_feature(random_indices.begin(),
                                                                     random_indices.end(),
                                                                     samples_range_first,
                                                                     samples_range_last,
                                                                     n_features,
                                                                     feature_mask);
    }
    // select the axis according to the dimension with the most spread between 2 points
    if (n_samples == 2) {
        // otherwise apply the bounding box method
        const auto hyper_interval = make_hyper_interval(samples_range_first, samples_range_last, n_features);
        return select_axis_with_largest_bounding_box_difference<SamplesIterator>(hyper_interval, feature_mask);
    }
    // return a random axis if theres only one sample left
    return common::math::random::uniform_distribution<std::size_t>(0, n_features - 1)();
}
template <typename ForwardedIndicesIteratorPair, typename SamplesIterator>
auto shift_median_to_leftmost_equal_value(std::size_t                    median_index,
                                          ForwardedIndicesIteratorPair&& left_indices_range,
                                          ForwardedIndicesIteratorPair&& median_indices_range,
                                          ForwardedIndicesIteratorPair&& right_indices_range,
                                          const SamplesIterator&         samples_range_first,
                                          const SamplesIterator&         samples_range_last,
                                          std::size_t                    n_features,
                                          std::size_t                    feature_index) -> std::
    tuple<std::size_t, ForwardedIndicesIteratorPair, ForwardedIndicesIteratorPair, ForwardedIndicesIteratorPair> {
    common::ignore_parameters(samples_range_last);

    using FirstIndicesIteratorType  = typename ForwardedIndicesIteratorPair::first_type;
    using SecondIndicesIteratorType = typename ForwardedIndicesIteratorPair::second_type;

    static_assert(common::is_iterator<FirstIndicesIteratorType>::value, "FirstIndicesIteratorType is not an iterator");
    static_assert(common::is_iterator<SecondIndicesIteratorType>::value,
                  "SecondIndicesIteratorType is not an iterator");

    static_assert(
        std::is_same_v<ForwardedIndicesIteratorPair, std::pair<FirstIndicesIteratorType, SecondIndicesIteratorType>>,
        "ForwardedIndicesIteratorPair must be an std::pair");

    using FirstIndexType  = typename std::iterator_traits<FirstIndicesIteratorType>::value_type;
    using SecondIndexType = typename std::iterator_traits<SecondIndicesIteratorType>::value_type;

    static_assert(std::is_trivial_v<FirstIndexType>, "FirstIndexType must be trivial.");
    static_assert(std::is_trivial_v<SecondIndexType>, "SecondIndexType must be trivial.");

    static_assert(common::is_iterator<FirstIndicesIteratorType>::value && std::is_integral_v<FirstIndexType>,
                  "The first element of FirstIndicesIteratorType must be an iterator of integers");

    static_assert(common::is_iterator<SecondIndicesIteratorType>::value && std::is_integral_v<SecondIndexType>,
                  "The first element of SecondIndicesIteratorType must be an iterator of integers");

    static_assert(common::is_iterator<SamplesIterator>::value, "SamplesIterator is not an iterator");

    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    // no left shift is possible if the left_indices_range is empty
    if (left_indices_range.first != left_indices_range.second) {
        // The target value of the median
        const auto cut_value = samples_range_first[median_indices_range.first[0] * n_features + feature_index];

        // The left range iterator at the value the current pointer will be compared to at each iteration
        auto left_neighbor_it = std::prev(left_indices_range.second);

        // Decrement the iterators while the left range isn't empty and the neighbor value at the left of the median is
        // still equal to the cut value at the corresponding feature index
        while (left_neighbor_it >= left_indices_range.first &&
               common::equality(samples_range_first[left_neighbor_it[0] * n_features + feature_index], cut_value)) {
            --left_neighbor_it;
        }
        // Update the ranges accordingly
        median_indices_range.first  = std::next(left_neighbor_it);
        median_indices_range.second = median_indices_range.first + 1;
        right_indices_range.first   = median_indices_range.second;
        left_indices_range.second   = median_indices_range.first;
        median_index                = std::distance(left_indices_range.first, median_indices_range.first);
    }
    return std::make_tuple(median_index,
                           std::forward<ForwardedIndicesIteratorPair>(left_indices_range),
                           std::forward<ForwardedIndicesIteratorPair>(median_indices_range),
                           std::forward<ForwardedIndicesIteratorPair>(right_indices_range));
}

template <typename IndicesIterator, typename SamplesIterator>
std::tuple<std::size_t,
           std::pair<IndicesIterator, IndicesIterator>,
           std::pair<IndicesIterator, IndicesIterator>,
           std::pair<IndicesIterator, IndicesIterator>>
quickselect_median(const IndicesIterator& indices_range_first,
                   const IndicesIterator& indices_range_last,
                   const SamplesIterator& samples_range_first,
                   const SamplesIterator& samples_range_last,
                   std::size_t            n_features,
                   std::size_t            feature_index) {
    assert(feature_index < n_features);
    assert(std::distance(indices_range_first, indices_range_last) > 0);

    static_assert(common::is_iterator<IndicesIterator>::value, "IndicesIterator is not an iterator");
    static_assert(common::is_iterator<SamplesIterator>::value, "SamplesIterator is not an iterator");

    using IndexType = typename std::iterator_traits<IndicesIterator>::value_type;
    using DataType  = typename std::iterator_traits<SamplesIterator>::value_type;

    static_assert(std::is_integral_v<IndexType>, "IndexType must be integer.");
    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    const std::size_t indices_range_size = std::distance(indices_range_first, indices_range_last);
    // the median index uses left rounding for ranges of sizes that are even
    const std::size_t median_index = (indices_range_size - 1) / 2;

    auto median_indices_range = common::algorithms::quickselect(indices_range_first,
                                                                indices_range_last,
                                                                samples_range_first,
                                                                samples_range_last,
                                                                n_features,
                                                                median_index,
                                                                feature_index);

    // all the points at the left of the pivot point
    auto left_indices_range = std::make_pair(indices_range_first, indices_range_first + median_index);

    // all the points at the right of the pivot point
    auto right_indices_range = std::make_pair(indices_range_first + median_index + 1, indices_range_last);

    return shift_median_to_leftmost_equal_value(median_index,
                                                std::move(left_indices_range),
                                                std::move(median_indices_range),
                                                std::move(right_indices_range),
                                                samples_range_first,
                                                samples_range_last,
                                                n_features,
                                                feature_index);
}

}  // namespace ffcl::datastruct::kdtree::algorithms