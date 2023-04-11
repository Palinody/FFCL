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
 * @brief Wiki (Dutch national flag problem): https://en.wikipedia.org/wiki/Dutch_national_flag_problem
 */
template <typename RandomAccessIterator>
std::pair<std::size_t, std::size_t> three_way_partition_around_nth_range(RandomAccessIterator first,
                                                                         RandomAccessIterator last,
                                                                         std::size_t          n_features,
                                                                         std::size_t          pivot_index,
                                                                         std::size_t          feature_index) {
    // values less than pivot: [0, i)
    std::size_t i = 0;
    // values equal to pivot: [i, j)
    std::size_t j = 0;
    // values not yet sorted: [j, k]
    std::size_t k = common::utils::get_n_samples(first, last, n_features) - 1;
    // values greater than pivot: [k+1, n_samples)

    const auto pivot_value = first[pivot_index * n_features + feature_index];

    while (j <= k) {
        if (first[j * n_features + feature_index] < pivot_value) {
            std::swap_ranges(first + i * n_features, first + i * n_features + n_features, first + j * n_features);
            ++i;
            ++j;

        } else if (first[j * n_features + feature_index] > pivot_value) {
            std::swap_ranges(first + j * n_features, first + j * n_features + n_features, first + k * n_features);
            --k;

        } else {
            ++j;
        }
    }
    return {i, j};
}

/**
 * @brief Hoare partition scheme algorithm that also handles duplicated values.
 */
template <typename RandomAccessIterator>
std::size_t partition_around_nth_range(RandomAccessIterator first,
                                       RandomAccessIterator last,
                                       std::size_t          n_features,
                                       std::size_t          pivot_index,
                                       std::size_t          feature_index) {
    const std::size_t n_samples = common::utils::get_n_samples(first, last, n_features);
    // no op if the input contains only one feature vector
    if (n_samples == 1) {
        return pivot_index;
    }
    // Initialize the left and right indices to be out of bounds, so that they never go out of bounds when incremented
    // or decremented in the loops
    ssize_t left_index  = -1;
    ssize_t right_index = n_samples;

    const auto pivot_value = first[pivot_index * n_features + feature_index];

    while (true) {
        do {
            ++left_index;
        } while (first[left_index * n_features + feature_index] < pivot_value);

        do {
            --right_index;
        } while (pivot_value < first[right_index * n_features + feature_index]);

        // the partitioning is done if the left and right indices cross
        if (left_index >= right_index) {
            break;
        }
        // if the values at the pivot and at the right index are not equal
        if (common::utils::inequality(pivot_value, first[right_index * n_features + feature_index])) {
            // swap the ranges at the left index and right index
            std::swap_ranges(first + left_index * n_features,
                             first + left_index * n_features + n_features,
                             first + right_index * n_features);
            // if the pivot was swapped because left index was equal to pivot index, update it to the index it was
            // swapped with (right index in this case)
            if (pivot_index == static_cast<std::size_t>(left_index)) {
                // the pivot index has now the value of the index it was swapped with (right index here)
                pivot_index = right_index;
                // then shift the right index back by one to avoid crossing over the pivot
                ++right_index;
            }
        }
        // the values at the pivot and the right index are equal
        else {
            // if the value at the left index is equal to the value at the pivot index
            if (common::utils::equality(first[left_index * n_features + feature_index], pivot_value)) {
                // dont swap if the left range and pivot range are actually confounded
                if (static_cast<std::size_t>(left_index) != pivot_index) {
                    // swap the ranges so that the range of the left index is put at the right of the pivot
                    std::swap_ranges(first + left_index * n_features,
                                     first + left_index * n_features + n_features,
                                     first + pivot_index * n_features);

                    // the pivot index has now the value of the index it was swapped with (left index here)
                    pivot_index = left_index;
                }
            }
            // if the value at the left index is not equal to the value at the pivot index
            else {
                // swap the ranges so that the range of the left index is put at the right of the pivot
                std::swap_ranges(first + left_index * n_features,
                                 first + left_index * n_features + n_features,
                                 first + pivot_index * n_features);

                // the pivot index has now the value of the index it was swapped with (right index here)
                pivot_index = left_index;
            }
            // shift the left index back by one to avoid crossing over the pivot
            // this operation is needed in all the conditional branches at the current level because the left range is
            // either swapped with the pivot range or left_index is equal to pivot_index
            --left_index;
        }
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

    const std::size_t n_samples = std::distance(index_first, index_last);

    // no op if the input contains only one feature vector
    if (n_samples == 1) {
        return pivot_index;
    }
    // Initialize the left and right indices to be out of bounds, so that they never go out of bounds when incremented
    // or decremented in the loops
    ssize_t left_index  = -1;
    ssize_t right_index = n_samples;

    const auto pivot_value = first[index_first[pivot_index] * n_features + feature_index];

    while (true) {
        do {
            ++left_index;
        } while (first[index_first[left_index] * n_features + feature_index] < pivot_value);

        do {
            --right_index;
        } while (pivot_value < first[index_first[right_index] * n_features + feature_index]);

        // the partitioning is done if the left and right indices cross
        if (left_index >= right_index) {
            break;
        }
        // if the values at the pivot and at the right index are not equal
        if (common::utils::inequality(pivot_value, first[index_first[right_index] * n_features + feature_index])) {
            // swap the ranges at the left index and right index
            std::iter_swap(index_first + left_index, index_first + right_index);
            // if the pivot was swapped because left index was equal to pivot index, update it to the index it was
            // swapped with (right index in this case)
            if (pivot_index == static_cast<std::size_t>(left_index)) {
                // the pivot index has now the value of the index it was swapped with (right index here)
                pivot_index = right_index;
                // then shift the right index back by one to avoid crossing over the pivot
                ++right_index;
            }
        }
        // the values at the pivot and the right index are equal
        else {
            // if the value at the left index is equal to the value at the pivot index
            if (common::utils::equality(first[index_first[left_index] * n_features + feature_index], pivot_value)) {
                // dont swap if the left range and pivot range are actually confounded
                if (static_cast<std::size_t>(left_index) != pivot_index) {
                    // swap the ranges so that the range of the left index is put at the right of the pivot
                    std::iter_swap(index_first + left_index, index_first + pivot_index);
                    // the pivot index has now the value of the index it was swapped with (left index here)
                    pivot_index = left_index;
                }
            }
            // if the value at the left index is not equal to the value at the pivot index
            else {
                // swap the ranges so that the range of the left index is put at the right of the pivot
                std::iter_swap(index_first + left_index, index_first + pivot_index);
                // the pivot index has now the value of the index it was swapped with (left index here)
                pivot_index = left_index;
            }
            // shift the left index back by one to avoid crossing over the pivot
            // this operation is needed in all the conditional branches at the current level because the left range is
            // either swapped with the pivot range or left_index is equal to pivot_index
            --left_index;
        }
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