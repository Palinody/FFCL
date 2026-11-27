#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/algorithms/Sorting.hpp"
#include "ffcl/common/math/random/Distributions.hpp"
#include "ffcl/common/math/random/Sampling.hpp"
#include "ffcl/common/math/statistics/Statistics.hpp"
#include "ffcl/datastruct/Interval.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>  // std::size_t
#include <limits>
#include <vector>

namespace ffcl::datastruct::kdtree::algorithms {

template <typename Bound>
std::size_t select_longest_axis(const Bound& bound) {
    const auto comparison = [](const auto& lhs, const auto& rhs) {
        return common::abs(lhs.lower_bound() - lhs.upper_bound()) < common::abs(rhs.lower_bound() - rhs.upper_bound());
    };
    const auto it = std::max_element(bound.segments_begin(), bound.segments_end(), comparison);
    return std::distance(bound.segments_begin(), it);
}

template <typename Bound>
std::size_t select_longest_axis(const Bound& bound, const std::vector<std::size_t>& feature_mask) {
    using DataType = typename Bound::ValueType;

    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    // feature index of the greatest range dimension in the kd bounding box at the indices of the feature_mask
    std::size_t current_max_range_feature_index = 0;
    DataType    current_max_range               = 0;

    // "feature_index"s are only the ones specified in the feature_mask
    for (std::size_t feature_index : feature_mask) {
        const auto max_range_candidate =
            common::abs(bound.segment_at(feature_index).upper_bound() - bound.segment_at(feature_index).lower_bound());

        if (max_range_candidate > current_max_range) {
            current_max_range_feature_index = feature_index;
            current_max_range               = max_range_candidate;
        }
    }
    return current_max_range_feature_index;
}

template <typename IndicesIterator, typename SamplesIterator, typename Bound>
std::size_t select_largest_variance_axis(const IndicesIterator& indices_range_first,
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
        // Sample in place: draw positions uniformly from the node's own index range and accumulate the
        // per-feature sums directly. Copying the range first and letting the statistics helpers allocate
        // per sample they read was most of the build's allocation traffic.
        auto sum_per_feature         = std::vector<double>(n_features, 0.0);
        auto squared_sum_per_feature = std::vector<double>(n_features, 0.0);

        for (std::size_t choice_index = 0; choice_index < n_choices; ++choice_index) {
            const auto sample_index =
                indices_range_first[common::math::random::uniform_distribution<std::size_t>(0, n_samples - 1)()];

            const auto sample_first = samples_range_first + static_cast<std::ptrdiff_t>(sample_index * n_features);

            for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
                const auto value = static_cast<double>(sample_first[feature_index]);
                sum_per_feature[feature_index] += value;
                squared_sum_per_feature[feature_index] += value * value;
            }
        }
        // variance of a sample of size m is E[x^2] - E[x]^2; the m / (m - 1) correction is a common
        // positive factor for every feature, so it cannot change the argmax
        std::size_t argmax_feature_index = 0;
        auto        max_variance         = std::numeric_limits<double>::lowest();

        const auto n_choices_as_double = static_cast<double>(n_choices);

        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            const auto mean     = sum_per_feature[feature_index] / n_choices_as_double;
            const auto variance = squared_sum_per_feature[feature_index] / n_choices_as_double - mean * mean;

            if (variance > max_variance) {
                max_variance         = variance;
                argmax_feature_index = feature_index;
            }
        }
        return argmax_feature_index;
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
                                                                     n_features);
    }
    // select the axis according to the dimension with the most spread between 2 points
    if (n_samples == 2) {
        // otherwise apply the bounding box method
        auto bound = make_tight_bound<SamplesIterator, Bound>(samples_range_first, samples_range_last, n_features);

        return select_longest_axis(bound);
    }
    // return a random axis if theres only one sample left
    return common::math::random::uniform_distribution<std::size_t>(0, n_features - 1)();
}

template <typename IndicesIterator, typename SamplesIterator>
auto quickselect_median(const IndicesIterator& indices_range_first,
                        const IndicesIterator& indices_range_last,
                        const SamplesIterator& samples_range_first,
                        const SamplesIterator& samples_range_last,
                        std::size_t            n_features,
                        std::size_t            feature_index) -> std::tuple<std::size_t,
                                                                 std::pair<IndicesIterator, IndicesIterator>,
                                                                 std::pair<IndicesIterator, IndicesIterator>,
                                                                 std::pair<IndicesIterator, IndicesIterator>>;

// Split at the middle of the node's own extent along the axis: one pass to find the extent, one partition
// pass, no selection at all. This is why it builds faster than the median rule.
//
// The contract matches `quickselect_median`: the pivot is the last element of the left range, so it stays
// owned by the left child and the leaves still cover every sample exactly once. The pivot is the largest
// value on the left side, which keeps the query routing (`query < pivot ? left : right`) and the cell bounds
// valid, because every value on the right side is strictly greater than the pivot.
//
// Falls back to the median when the axis has no extent (all values equal) or when the partition would leave
// one side empty, so neither child can ever be empty.
template <typename IndicesIterator, typename SamplesIterator>
auto midpoint_range(const IndicesIterator& indices_range_first,
                    const IndicesIterator& indices_range_last,
                    const SamplesIterator& samples_range_first,
                    const SamplesIterator& samples_range_last,
                    std::size_t            n_features,
                    std::size_t            feature_index) -> std::tuple<std::size_t,
                                                             std::pair<IndicesIterator, IndicesIterator>,
                                                             std::pair<IndicesIterator, IndicesIterator>,
                                                             std::pair<IndicesIterator, IndicesIterator>> {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const auto n_samples = std::distance(indices_range_first, indices_range_last);

    if (n_samples < 3) {
        return quickselect_median(indices_range_first,
                                  indices_range_last,
                                  samples_range_first,
                                  samples_range_last,
                                  n_features,
                                  feature_index);
    }
    auto lower_bound = std::numeric_limits<DataType>::max();
    auto upper_bound = std::numeric_limits<DataType>::lowest();

    for (auto index_it = indices_range_first; index_it != indices_range_last; ++index_it) {
        const auto value = samples_range_first[*index_it * n_features + feature_index];

        lower_bound = std::min(lower_bound, value);
        upper_bound = std::max(upper_bound, value);
    }
    if (!(lower_bound < upper_bound)) {
        return quickselect_median(indices_range_first,
                                  indices_range_last,
                                  samples_range_first,
                                  samples_range_last,
                                  n_features,
                                  feature_index);
    }
    const auto cut_value = lower_bound + (upper_bound - lower_bound) / 2;

    const auto median_it = std::partition(indices_range_first, indices_range_last, [&](const auto& index) {
        return samples_range_first[index * n_features + feature_index] < cut_value;
    });

    // A midpoint split can be very unbalanced - one outlier on one side and everything else on the other -
    // and FFCL's traversal pays a function call, a bound test and a sibling test per node, so it only
    // amortises on reasonably full leaves. Fall back to the median when a side would hold less than a
    // quarter of the samples: the cheap rule is then used where it is cheap *and* balanced.
    const auto n_left  = std::distance(indices_range_first, median_it);
    const auto n_right = std::distance(median_it, indices_range_last);

    if (n_left < static_cast<std::ptrdiff_t>(n_samples / 4) || n_right < static_cast<std::ptrdiff_t>(n_samples / 4)) {
        return quickselect_median(indices_range_first,
                                  indices_range_last,
                                  samples_range_first,
                                  samples_range_last,
                                  n_features,
                                  feature_index);
    }
    if (median_it == indices_range_first || median_it == indices_range_last) {
        return quickselect_median(indices_range_first,
                                  indices_range_last,
                                  samples_range_first,
                                  samples_range_last,
                                  n_features,
                                  feature_index);
    }
    // `std::partition` does not sort, so the last element of the left range is not necessarily its largest.
    // The routing test compares against the pivot, so the pivot has to be the maximum of the left side:
    // move that maximum to the end of the left range and use it.
    const auto max_it   = std::max_element(indices_range_first, median_it, [&](const auto& lhs, const auto& rhs) {
        return samples_range_first[lhs * n_features + feature_index] <
               samples_range_first[rhs * n_features + feature_index];
    });
    const auto pivot_it = std::prev(median_it);

    std::iter_swap(max_it, pivot_it);

    const auto pivot_index = static_cast<std::size_t>(*pivot_it);

    return std::make_tuple(pivot_index,
                           std::make_pair(indices_range_first, median_it),
                           std::make_pair(pivot_it, median_it),
                           std::make_pair(median_it, indices_range_last));
}

template <typename IndicesIterator, typename SamplesIterator, typename Bound>
std::size_t select_largest_variance_axis(const IndicesIterator&          indices_range_first,
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
        auto bound = make_tight_bound<SamplesIterator, Bound>(samples_range_first, samples_range_last, n_features);

        return select_longest_axis(bound, feature_mask);
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
        median_index                = std::distance(left_indices_range.first, median_indices_range.first);
    }
    // The pivot sample is kept in the node's own `indices_range_` because it is needed to route the queries
    // (`select_closest_child` / `select_sibling_node` compare against its value). It is *also* kept as the last
    // element of the left range so that the leaves of the tree cover every sample exactly once. Without this
    // duplication the pivot samples only live in internal nodes, which forces a dual-tree traversal to visit
    // every node combination twice (query side and reference side) and makes it quadratic in the number of
    // nodes.
    left_indices_range.second = median_indices_range.second;
    right_indices_range.first = median_indices_range.second;
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