#pragma once

#include "ffcl/common/Utils.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace ffcl::algorithms {

template <typename RandomAccessIterator>
std::size_t median_index_of_three_ranges(RandomAccessIterator first,
                                         RandomAccessIterator last,
                                         std::size_t          n_features,
                                         std::size_t          feature_index) {
    const std::size_t n_samples    = common::utils::get_n_samples(first, last, n_features);
    std::size_t       middle_index = n_samples / 2;

    if (n_samples < 3) {
        return middle_index;
    }
    std::size_t left_index  = 0;
    std::size_t right_index = n_samples - 1;

    if (first[middle_index * n_features + feature_index] < first[left_index * n_features + feature_index]) {
        std::swap(middle_index, left_index);
    }
    if (first[right_index * n_features + feature_index] < first[left_index * n_features + feature_index]) {
        std::swap(right_index, left_index);
    }
    if (first[middle_index * n_features + feature_index] < first[right_index * n_features + feature_index]) {
        std::swap(middle_index, right_index);
    }
    return right_index;
}

template <typename RandomAccessIntIterator, typename RandomAccessIterator>
std::size_t median_index_of_three_indexed_ranges(RandomAccessIntIterator index_first,
                                                 RandomAccessIntIterator index_last,
                                                 RandomAccessIterator    first,
                                                 RandomAccessIterator    last,
                                                 std::size_t             n_features,
                                                 std::size_t             feature_index) {
    common::utils::ignore_parameters(last);

    const std::size_t n_samples    = std::distance(index_first, index_last);
    std::size_t       middle_index = n_samples / 2;

    if (n_samples < 3) {
        return middle_index;
    }
    std::size_t left_index  = 0;
    std::size_t right_index = n_samples - 1;

    if (first[index_first[middle_index] * n_features + feature_index] <
        first[index_first[left_index] * n_features + feature_index]) {
        std::swap(middle_index, left_index);
    }
    if (first[index_first[right_index] * n_features + feature_index] <
        first[index_first[left_index] * n_features + feature_index]) {
        std::swap(right_index, left_index);
    }
    if (first[index_first[middle_index] * n_features + feature_index] <
        first[index_first[right_index] * n_features + feature_index]) {
        std::swap(middle_index, right_index);
    }
    return right_index;
}

template <typename RandomAccessIterator>
std::pair<RandomAccessIterator, RandomAccessIterator> median_values_range_of_three_ranges(RandomAccessIterator first,
                                                                                          RandomAccessIterator last,
                                                                                          std::size_t n_features,
                                                                                          std::size_t feature_index) {
    const std::size_t median_index = median_index_of_three_ranges(first, last, n_features, feature_index);

    return {first + median_index * n_features, first + median_index * n_features + n_features};
}

template <typename RandomAccessIntIterator, typename RandomAccessIterator>
std::pair<RandomAccessIterator, RandomAccessIterator> median_values_range_of_three_indexed_ranges(
    RandomAccessIntIterator index_first,
    RandomAccessIntIterator index_last,
    RandomAccessIterator    first,
    RandomAccessIterator    last,
    std::size_t             n_features,
    std::size_t             feature_index) {
    const std::size_t median_index =
        median_index_of_three_indexed_ranges(index_first, index_last, first, last, n_features, feature_index);

    return {first + index_first[median_index] * n_features,
            first + index_first[median_index] * n_features + n_features};
}

template <typename RandomAccessIterator>
std::pair<std::size_t, std::pair<RandomAccessIterator, RandomAccessIterator>>
median_index_and_values_range_of_three_ranges(RandomAccessIterator first,
                                              RandomAccessIterator last,
                                              std::size_t          n_features,
                                              std::size_t          feature_index) {
    const std::size_t median_index = median_index_of_three_ranges(first, last, n_features, feature_index);

    return {median_index, {first + median_index * n_features, first + median_index * n_features + n_features}};
}

template <typename RandomAccessIntIterator, typename RandomAccessIterator>
std::pair<std::size_t, std::pair<RandomAccessIterator, RandomAccessIterator>>
median_index_and_values_range_of_three_indexed_ranges(RandomAccessIntIterator index_first,
                                                      RandomAccessIntIterator index_last,
                                                      RandomAccessIterator    first,
                                                      RandomAccessIterator    last,
                                                      std::size_t             n_features,
                                                      std::size_t             feature_index) {
    const std::size_t median_index =
        median_index_of_three_indexed_ranges(index_first, index_last, first, last, n_features, feature_index);

    return {
        median_index,
        {first + index_first[median_index] * n_features, first + index_first[median_index] * n_features + n_features}};
}

/**
 * @brief Hoare partition scheme algorithm.
 * This implementation will also swap ranges that are evaluated as equal.
 * However, it will ensure that the indices dont move out of range, without checks.
 */
template <typename RandomAccessIterator>
std::size_t partition_around_nth_range(RandomAccessIterator first,
                                       RandomAccessIterator last,
                                       std::size_t          n_features,
                                       std::size_t          pivot_index,
                                       std::size_t          feature_index) {
    // no op if the input contains only one feature vector
    if (std::distance(first, last) == static_cast<std::ptrdiff_t>(n_features)) {
        return pivot_index;
    }
    // Initialize the left and right indices to be out of bounds, so that they never go out of bounds when incremented
    // or decremented in the loops
    ssize_t left_index  = -1;
    ssize_t right_index = common::utils::get_n_samples(first, last, n_features);

    while (true) {
        do {
            ++left_index;
        } while (first[left_index * n_features + feature_index] < first[pivot_index * n_features + feature_index]);

        do {
            --right_index;
        } while (first[pivot_index * n_features + feature_index] < first[right_index * n_features + feature_index]);

        // the partitioning is done if the left and right indices cross
        if (left_index >= right_index) {
            break;
        }
        std::swap_ranges(first + left_index * n_features,
                         first + left_index * n_features + n_features,
                         first + right_index * n_features);

        // if the pivot index was swapped, update it to the index it was swapped with
        if (pivot_index == static_cast<std::size_t>(left_index)) {
            pivot_index = right_index;
            // shift the right index by one to avoid crossing over the left of the new pivot
            ++right_index;

        } else if (pivot_index == static_cast<std::size_t>(right_index)) {
            pivot_index = left_index;
        }
        // always shift the left_index back since the value at right_index might be the same as the one at the pivot
        --left_index;
    }
    return pivot_index;
}

template <typename RandomAccessIntIterator, typename RandomAccessIterator>
std::size_t partition_around_nth_indexed_range(RandomAccessIntIterator index_first,
                                               RandomAccessIntIterator index_last,
                                               RandomAccessIterator    first,
                                               RandomAccessIterator    last,
                                               std::size_t             n_features,
                                               std::size_t             pivot_index,
                                               std::size_t             feature_index) {
    static_assert(std::is_integral_v<typename RandomAccessIntIterator::value_type>, "Index input should be integral.");

    common::utils::ignore_parameters(last);

    // no op if the input contains only one feature vector
    if (std::distance(index_first, index_last) == static_cast<std::ptrdiff_t>(1)) {
        return pivot_index;
    }
    // Initialize the left and right indices to be out of bounds, so that they never go out of bounds when incremented
    // or decremented in the loops
    ssize_t left_index  = -1;
    ssize_t right_index = std::distance(index_first, index_last);

    while (true) {
        do {
            ++left_index;
        } while (first[index_first[left_index] * n_features + feature_index] <
                 first[index_first[pivot_index] * n_features + feature_index]);

        do {
            --right_index;
        } while (first[index_first[pivot_index] * n_features + feature_index] <
                 first[index_first[right_index] * n_features + feature_index]);

        // the partitioning is done if the left and right indices cross
        if (left_index >= right_index) {
            break;
        }
        std::iter_swap(index_first + left_index, index_first + right_index);

        // if the pivot index was swapped, update it to the index it was swapped with
        if (pivot_index == static_cast<std::size_t>(left_index)) {
            pivot_index = right_index;
            // shift the right index by one to avoid crossing over the left of the new pivot
            ++right_index;

        } else if (pivot_index == static_cast<std::size_t>(right_index)) {
            pivot_index = left_index;
        }
        // always shift the left_index back since the value at right_index might be the same as the one at the pivot
        --left_index;
    }
    return pivot_index;
}

/**
 * @brief A quickselect implementation https://en.wikipedia.org/wiki/Quickselect
 */
template <typename RandomAccessIterator>
std::pair<RandomAccessIterator, RandomAccessIterator> quickselect_range(RandomAccessIterator first,
                                                                        RandomAccessIterator last,
                                                                        std::size_t          n_features,
                                                                        std::size_t          kth_smallest,
                                                                        std::size_t          feature_index) {
    std::size_t left_index  = 0;
    std::size_t right_index = common::utils::get_n_samples(first, last, n_features) - 1;

    while (true) {
        if (left_index == right_index) {
            return {first + left_index * n_features, first + right_index * n_features + n_features};
        }
        std::size_t pivot_index = median_index_of_three_ranges(
            first + left_index * n_features, first + right_index * n_features + n_features, n_features, feature_index);

        // partition the range around the pivot, which has moved to its sorted index. The pivot index is relative to the
        // left_index, so to get its absolute index, we need to shift it by the same amount
        pivot_index = left_index + partition_around_nth_range(first + left_index * n_features,
                                                              first + right_index * n_features + n_features,
                                                              n_features,
                                                              pivot_index,
                                                              feature_index);

        if (kth_smallest == pivot_index) {
            return {first + pivot_index * n_features, first + pivot_index * n_features + n_features};

        } else if (kth_smallest < pivot_index) {
            right_index = pivot_index - 1;

        } else {
            left_index = pivot_index + 1;
        }
    }
}

template <typename RandomAccessIntIterator, typename RandomAccessIterator>
std::pair<RandomAccessIterator, RandomAccessIterator> quickselect_indexed_range(RandomAccessIntIterator index_first,
                                                                                RandomAccessIntIterator index_last,
                                                                                RandomAccessIterator    first,
                                                                                RandomAccessIterator    last,
                                                                                std::size_t             n_features,
                                                                                std::size_t             kth_smallest,
                                                                                std::size_t             feature_index) {
    std::size_t left_index  = 0;
    std::size_t right_index = std::distance(index_first, index_last) - 1;

    while (true) {
        if (left_index == right_index) {
            return {first + index_first[left_index] * n_features,
                    first + index_first[right_index] * n_features + n_features};
        }
        std::size_t pivot_index = median_index_of_three_indexed_ranges(
            index_first + left_index, index_first + right_index + 1, first, last, n_features, feature_index);

        // partition the range around the pivot, which has moved to its sorted index. The pivot index is relative to the
        // left_index, so to get its absolute index, we need to shift it by the same amount
        pivot_index = left_index + partition_around_nth_indexed_range(index_first + left_index,
                                                                      index_first + right_index + 1,
                                                                      first,
                                                                      last,
                                                                      n_features,
                                                                      pivot_index,
                                                                      feature_index);

        if (kth_smallest == pivot_index) {
            return {first + index_first[pivot_index] * n_features,
                    first + index_first[pivot_index] * n_features + n_features};

        } else if (kth_smallest < pivot_index) {
            right_index = pivot_index - 1;

        } else {
            left_index = pivot_index + 1;
        }
    }
}

/**
 * @brief A quicksort implementation https://en.wikipedia.org/wiki/Quicksort#Algorithm
 */
template <typename RandomAccessIterator>
std::size_t quicksort_range(RandomAccessIterator first,
                            RandomAccessIterator last,
                            std::size_t          n_features,
                            std::size_t          initial_pivot_index,
                            std::size_t          feature_index) {
    const std::size_t n_samples = common::utils::get_n_samples(first, last, n_features);

    // the pivot index is already correct if the number of samples is 0 or 1
    if (n_samples < 2) {
        return initial_pivot_index;
    }
    // partial sort ranges around new pivot index
    const std::size_t new_pivot_index =
        partition_around_nth_range(first, last, n_features, initial_pivot_index, feature_index);

    // compute the median of the subranges

    std::size_t pivot_index_subrange_left =
        median_index_of_three_ranges(first, first + new_pivot_index * n_features, n_features, feature_index);

    std::size_t pivot_index_subrange_right = median_index_of_three_ranges(
        first + new_pivot_index * n_features + n_features, last, n_features, feature_index);

    // the pivot range is included in the left subrange

    quicksort_range(/*first iterator, left subrange*/ first,
                    /*last iterator, left subrange*/ first + new_pivot_index * n_features,
                    n_features,
                    pivot_index_subrange_left,
                    feature_index);

    quicksort_range(/*first iterator, right subrange*/ first + new_pivot_index * n_features + n_features,
                    /*last iterator, right subrange*/ last,
                    n_features,
                    pivot_index_subrange_right,
                    feature_index);

    return new_pivot_index;
}

template <typename RandomAccessIntIterator, typename RandomAccessIterator>
std::size_t quicksort_indexed_range(RandomAccessIntIterator index_first,
                                    RandomAccessIntIterator index_last,
                                    RandomAccessIterator    first,
                                    RandomAccessIterator    last,
                                    std::size_t             n_features,
                                    std::size_t             initial_pivot_index,
                                    std::size_t             feature_index) {
    const std::size_t n_samples = std::distance(index_first, index_last);

    // the pivot index is already correct if the number of samples is 0 or 1
    if (n_samples < 2) {
        return initial_pivot_index;
    }
    // partial sort ranges around new pivot index
    const std::size_t new_pivot_index = partition_around_nth_indexed_range(
        index_first, index_last, first, last, n_features, initial_pivot_index, feature_index);

    // compute the median of the subranges

    std::size_t pivot_index_subrange_left = median_index_of_three_indexed_ranges(
        index_first, index_first + new_pivot_index, first, last, n_features, feature_index);

    std::size_t pivot_index_subrange_right = median_index_of_three_indexed_ranges(
        index_first + new_pivot_index + 1, index_last, first, last, n_features, feature_index);

    // the pivot range is included in the left subrange

    quicksort_indexed_range(index_first,
                            index_first + new_pivot_index,
                            /*first iterator, left subrange*/ first,
                            /*last iterator, left subrange*/ last,
                            n_features,
                            pivot_index_subrange_left,
                            feature_index);

    quicksort_indexed_range(index_first + new_pivot_index + 1,
                            index_last,
                            /*first iterator, right subrange*/ first,
                            /*last iterator, right subrange*/ last,
                            n_features,
                            pivot_index_subrange_right,
                            feature_index);

    return new_pivot_index;
}

}  // namespace ffcl::algorithms