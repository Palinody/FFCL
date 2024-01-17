#pragma once

#include "ffcl/common/Utils.hpp"

#include <sys/types.h>  // ssize_t
#include <algorithm>
#include <cstddef>  // std::size_t
#include <memory>
#include <vector>

namespace ffcl::common::algorithms {

template <typename IndicesIterator, typename SamplesIterator>
std::size_t median_index_of_three(const IndicesIterator& indices_range_first,
                                  const IndicesIterator& indices_range_last,
                                  const SamplesIterator& samples_range_first,
                                  const SamplesIterator& samples_range_last,
                                  std::size_t            n_features,
                                  std::size_t            feature_index) {
    ignore_parameters(samples_range_last);

    const std::size_t indices_range_size = std::distance(indices_range_first, indices_range_last);
    // return 'median_index=0' if the range is empty or contains 1 or 2 elements (left rounding median index)
    if (indices_range_size < 3) {
        return 0;
    }
    std::size_t left_index  = 0;
    std::size_t right_index = indices_range_size - 1;
    // the median index uses left rounding for ranges of sizes that are even
    std::size_t median_index = right_index / 2;

    if (samples_range_first[indices_range_first[median_index] * n_features + feature_index] <
        samples_range_first[indices_range_first[left_index] * n_features + feature_index]) {
        std::swap(median_index, left_index);
    }
    if (samples_range_first[indices_range_first[right_index] * n_features + feature_index] <
        samples_range_first[indices_range_first[left_index] * n_features + feature_index]) {
        std::swap(right_index, left_index);
    }
    if (samples_range_first[indices_range_first[median_index] * n_features + feature_index] <
        samples_range_first[indices_range_first[right_index] * n_features + feature_index]) {
        std::swap(median_index, right_index);
    }
    return right_index;
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t partition_around_pivot_index(const IndicesIterator& indices_range_first,
                                         const IndicesIterator& indices_range_last,
                                         const SamplesIterator& samples_range_first,
                                         const SamplesIterator& samples_range_last,
                                         std::size_t            n_features,
                                         std::size_t            pivot_index,
                                         std::size_t            feature_index) {
    static_assert(std::is_integral_v<typename IndicesIterator::value_type>, "Index input should be integral.");

    ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    assert(n_samples);

    // no op if the input contains only one feature vector
    if (n_samples == 1) {
        return pivot_index;
    }
    // Initialize the left and right indices to be out of bounds, so that they never go out of bounds when incremented
    // or decremented in the loops
    ssize_t left_index  = -1;
    ssize_t right_index = n_samples;

    const auto pivot_value = samples_range_first[indices_range_first[pivot_index] * n_features + feature_index];

    while (true) {
        do {
            ++left_index;
        } while (samples_range_first[indices_range_first[left_index] * n_features + feature_index] < pivot_value);

        do {
            --right_index;
        } while (pivot_value < samples_range_first[indices_range_first[right_index] * n_features + feature_index]);

        // the partitioning is done if the left and right indices cross
        if (left_index >= right_index) {
            break;
        }
        /*
        std::iter_swap(indices_range_first + left_index, indices_range_first + right_index);
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

        // at this point:
        // (left_value <= pivot_value && pivot_value >= right_value)
        // &&
        // left_index (!= || ==) pivot_index && pivot_index (!= || ==) right_index

        // pivot_value > right_value
        if (inequality(pivot_value,
                       samples_range_first[indices_range_first[right_index] * n_features + feature_index])) {
            // swap left_value and right_value
            std::iter_swap(indices_range_first + left_index, indices_range_first + right_index);
            // if the pivot was swapped because left index was equal to pivot index, update it to the index it was
            // swapped with (right index in this case)
            if (pivot_index == static_cast<std::size_t>(left_index)) {
                // the pivot index has now the value of the index it was swapped with (right index here)
                pivot_index = right_index;
                // then shift the right index back by one to avoid crossing over the pivot
                ++right_index;
            }
        }
        // else if pivot_value == right_value
        else {
            // if left_value == pivot_value
            if (equality(samples_range_first[indices_range_first[left_index] * n_features + feature_index],
                         pivot_value)) {
                // dont swap if the left_index == pivot_index
                if (static_cast<std::size_t>(left_index) != pivot_index) {
                    // swap the ranges so that the range of the left index is put at the right of the pivot
                    std::iter_swap(indices_range_first + left_index, indices_range_first + pivot_index);
                    // the pivot index has now the value of the index it was swapped with (left index here)
                    pivot_index = left_index;
                }
            }
            // if the value at the left index is not equal to the value at the pivot index
            else {
                // swap the ranges so that the range of the left index is put at the right of the pivot
                std::iter_swap(indices_range_first + left_index, indices_range_first + pivot_index);
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
auto quickselect(const IndicesIterator& indices_range_first,
                 const IndicesIterator& indices_range_last,
                 const SamplesIterator& samples_range_first,
                 const SamplesIterator& samples_range_last,
                 std::size_t            n_features,
                 std::size_t            kth_smallest,
                 std::size_t            feature_index) {
    std::size_t left_index  = 0;
    std::size_t right_index = std::distance(indices_range_first, indices_range_last) - 1;

    while (true) {
        if (left_index == right_index) {
            return make_pair(indices_range_first + left_index, indices_range_first + right_index + 1);
        }
        std::size_t pivot_index = median_index_of_three(indices_range_first + left_index,
                                                        indices_range_first + right_index + 1,
                                                        samples_range_first,
                                                        samples_range_last,
                                                        n_features,
                                                        feature_index);

        // partition the range around the pivot, which has moved to its sorted index. The pivot index is relative to the
        // left_index, so to get its absolute index, we need to shift it by the same amount
        pivot_index = left_index + partition_around_pivot_index(indices_range_first + left_index,
                                                                indices_range_first + right_index + 1,
                                                                samples_range_first,
                                                                samples_range_last,
                                                                n_features,
                                                                pivot_index,
                                                                feature_index);

        if (kth_smallest == pivot_index) {
            return std::make_pair(indices_range_first + pivot_index, indices_range_first + pivot_index + 1);

        } else if (kth_smallest < pivot_index) {
            right_index = pivot_index - 1;

        } else {
            left_index = pivot_index + 1;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t quicksort(const IndicesIterator& indices_range_first,
                      const IndicesIterator& indices_range_last,
                      const SamplesIterator& samples_range_first,
                      const SamplesIterator& samples_range_last,
                      std::size_t            n_features,
                      std::size_t            initial_pivot_index,
                      std::size_t            feature_index) {
    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    // the pivot index is already correct if the number of samples is 0 or 1
    if (n_samples < 2) {
        return initial_pivot_index;
    }
    // partial sort ranges around new pivot index
    const std::size_t new_pivot_index = partition_around_pivot_index(indices_range_first,
                                                                     indices_range_last,
                                                                     samples_range_first,
                                                                     samples_range_last,
                                                                     n_features,
                                                                     initial_pivot_index,
                                                                     feature_index);

    // compute the median of the subranges

    const std::size_t pivot_index_of_left_subrange = median_index_of_three(indices_range_first,
                                                                           indices_range_first + new_pivot_index,
                                                                           samples_range_first,
                                                                           samples_range_last,
                                                                           n_features,
                                                                           feature_index);

    const std::size_t pivot_index_of_right_subrange = median_index_of_three(indices_range_first + new_pivot_index + 1,
                                                                            indices_range_last,
                                                                            samples_range_first,
                                                                            samples_range_last,
                                                                            n_features,
                                                                            feature_index);

    // the pivot range is included in the left subrange

    quicksort(indices_range_first,
              indices_range_first + new_pivot_index,
              samples_range_first,
              samples_range_last,
              n_features,
              pivot_index_of_left_subrange,
              feature_index);

    quicksort(indices_range_first + new_pivot_index + 1,
              indices_range_last,
              samples_range_first,
              samples_range_last,
              n_features,
              pivot_index_of_right_subrange,
              feature_index);

    return new_pivot_index;
}

}  // namespace ffcl::common::algorithms