#pragma once

#include "ffcl/common/Utils.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace ffcl::algorithms {

template <typename IndicesIterator, typename SamplesIterator>
std::size_t median_index_of_three(const IndicesIterator& index_first,
                                  const IndicesIterator& index_last,
                                  const SamplesIterator& samples_first,
                                  const SamplesIterator& samples_last,
                                  std::size_t            n_features,
                                  std::size_t            feature_index) {
    common::utils::ignore_parameters(samples_last);

    const std::size_t n_samples    = std::distance(index_first, index_last);
    std::size_t       middle_index = n_samples / 2;

    if (n_samples < 3) {
        return middle_index;
    }
    std::size_t left_index  = 0;
    std::size_t right_index = n_samples - 1;

    if (samples_first[index_first[middle_index] * n_features + feature_index] <
        samples_first[index_first[left_index] * n_features + feature_index]) {
        std::swap(middle_index, left_index);
    }
    if (samples_first[index_first[right_index] * n_features + feature_index] <
        samples_first[index_first[left_index] * n_features + feature_index]) {
        std::swap(right_index, left_index);
    }
    if (samples_first[index_first[middle_index] * n_features + feature_index] <
        samples_first[index_first[right_index] * n_features + feature_index]) {
        std::swap(middle_index, right_index);
    }
    return right_index;
}

template <typename IndicesIterator, typename SamplesIterator>
std::pair<SamplesIterator, SamplesIterator> median_values_range_of_three(const IndicesIterator& index_first,
                                                                         const IndicesIterator& index_last,
                                                                         const SamplesIterator& samples_first,
                                                                         const SamplesIterator& samples_last,
                                                                         std::size_t            n_features,
                                                                         std::size_t            feature_index) {
    const std::size_t median_index =
        median_index_of_three(index_first, index_last, samples_first, samples_last, n_features, feature_index);

    return {samples_first + index_first[median_index] * n_features,
            samples_first + index_first[median_index] * n_features + n_features};
}

template <typename IndicesIterator, typename SamplesIterator>
std::pair<std::size_t, std::pair<SamplesIterator, SamplesIterator>> median_index_and_values_range_of_three(
    const IndicesIterator& index_first,
    const IndicesIterator& index_last,
    const SamplesIterator& samples_first,
    const SamplesIterator& samples_last,
    std::size_t            n_features,
    std::size_t            feature_index) {
    const std::size_t median_index =
        median_index_of_three(index_first, index_last, samples_first, samples_last, n_features, feature_index);

    return {median_index,
            {samples_first + index_first[median_index] * n_features,
             samples_first + index_first[median_index] * n_features + n_features}};
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t partition_around_nth_index(const IndicesIterator& index_first,
                                       const IndicesIterator& index_last,
                                       const SamplesIterator& samples_first,
                                       const SamplesIterator& samples_last,
                                       std::size_t            n_features,
                                       std::size_t            pivot_index,
                                       std::size_t            feature_index) {
    static_assert(std::is_integral_v<typename IndicesIterator::value_type>, "Index input should be integral.");

    common::utils::ignore_parameters(samples_last);

    const std::size_t n_samples = std::distance(index_first, index_last);

    // no op if the input contains only one feature vector
    if (n_samples == 1) {
        return pivot_index;
    }
    // Initialize the left and right indices to be out of bounds, so that they never go out of bounds when incremented
    // or decremented in the loops
    ssize_t left_index  = -1;
    ssize_t right_index = n_samples;

    const auto pivot_value = samples_first[index_first[pivot_index] * n_features + feature_index];

    while (true) {
        do {
            ++left_index;
        } while (samples_first[index_first[left_index] * n_features + feature_index] < pivot_value);

        do {
            --right_index;
        } while (pivot_value < samples_first[index_first[right_index] * n_features + feature_index]);

        // the partitioning is done if the left and right indices cross
        if (left_index >= right_index) {
            break;
        }
        /*
        std::iter_swap(index_first + left_index, index_first + right_index);
        // if the pivot has been swapped (because it was equal to one of the swapped ranges),
        // assign the pivot_index to the (left/right)_index its been swapped with
        if (pivot_index == static_cast<std::size_t>(left_index)) {
            pivot_index = right_index;
            // shift the right index by one so that it doesnt cross-over past the left of the now swapped pivot
            ++right_index;

        } else if (pivot_index == static_cast<std::size_t>(right_index)) {
            pivot_index = left_index;
            // shift the left index by one so that it doesnt cross-over past the right of the now swapped pivot
            --left_index;
        }
        */
        // /*
        // if the values at the pivot and at the right index are not equal
        if (common::utils::inequality(pivot_value,
                                      samples_first[index_first[right_index] * n_features + feature_index])) {
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
            if (common::utils::equality(samples_first[index_first[left_index] * n_features + feature_index],
                                        pivot_value)) {
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
        // */
    }
    return pivot_index;
}

template <typename IndicesIterator, typename SamplesIterator>
std::pair<IndicesIterator, IndicesIterator> quickselect(const IndicesIterator& index_first,
                                                        const IndicesIterator& index_last,
                                                        const SamplesIterator& samples_first,
                                                        const SamplesIterator& samples_last,
                                                        std::size_t            n_features,
                                                        std::size_t            kth_smallest,
                                                        std::size_t            feature_index) {
    std::size_t left_index  = 0;
    std::size_t right_index = std::distance(index_first, index_last) - 1;

    while (true) {
        if (left_index == right_index) {
            return {index_first + left_index, index_first + right_index + 1};
        }
        std::size_t pivot_index = median_index_of_three(index_first + left_index,
                                                        index_first + right_index + 1,
                                                        samples_first,
                                                        samples_last,
                                                        n_features,
                                                        feature_index);

        // partition the range around the pivot, which has moved to its sorted index. The pivot index is relative to the
        // left_index, so to get its absolute index, we need to shift it by the same amount
        pivot_index = left_index + partition_around_nth_index(index_first + left_index,
                                                              index_first + right_index + 1,
                                                              samples_first,
                                                              samples_last,
                                                              n_features,
                                                              pivot_index,
                                                              feature_index);

        if (kth_smallest == pivot_index) {
            return {index_first + pivot_index, index_first + pivot_index + 1};

        } else if (kth_smallest < pivot_index) {
            right_index = pivot_index - 1;

        } else {
            left_index = pivot_index + 1;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t quicksort(const IndicesIterator& index_first,
                      const IndicesIterator& index_last,
                      const SamplesIterator& samples_first,
                      const SamplesIterator& samples_last,
                      std::size_t            n_features,
                      std::size_t            initial_pivot_index,
                      std::size_t            feature_index) {
    const std::size_t n_samples = std::distance(index_first, index_last);

    // the pivot index is already correct if the number of samples is 0 or 1
    if (n_samples < 2) {
        return initial_pivot_index;
    }
    // partial sort ranges around new pivot index
    const std::size_t new_pivot_index = partition_around_nth_index(
        index_first, index_last, samples_first, samples_last, n_features, initial_pivot_index, feature_index);

    // compute the median of the subranges

    std::size_t pivot_index_subrange_left = median_index_of_three(
        index_first, index_first + new_pivot_index, samples_first, samples_last, n_features, feature_index);

    std::size_t pivot_index_subrange_right = median_index_of_three(
        index_first + new_pivot_index + 1, index_last, samples_first, samples_last, n_features, feature_index);

    // the pivot range is included in the left subrange

    quicksort(/*left subrange*/ index_first,
              /*left subrange*/ index_first + new_pivot_index,
              samples_first,
              samples_last,
              n_features,
              pivot_index_subrange_left,
              feature_index);

    quicksort(/*right subrange*/ index_first + new_pivot_index + 1,
              /*right subrange*/ index_last,
              samples_first,
              samples_last,
              n_features,
              pivot_index_subrange_right,
              feature_index);

    return new_pivot_index;
}

}  // namespace ffcl::algorithms