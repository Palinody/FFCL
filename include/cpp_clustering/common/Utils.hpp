#pragma once

#include "cpp_clustering/math/random/Distributions.hpp"

#include <sys/types.h>  // ssize_t
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace common::utils {

template <typename T>
static constexpr T infinity() {
    return std::numeric_limits<T>::max();
}

inline void xor_swap(std::size_t* lhs, std::size_t* rhs) {
    // no swap if both variables share same memory
    if (lhs != rhs) {
        *lhs ^= *rhs;
        *rhs ^= *lhs;
        *lhs ^= *rhs;
    } else {
        throw std::invalid_argument("Cannot swap 2 variables that share memory.\n");
    }
}

template <typename IntType>
IntType factorial(const IntType& n) {
    static_assert(std::is_integral_v<IntType>, "The input type should be integral.");

    IntType result = 1;
    for (IntType i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

template <typename T>
T abs(const T& x) {
    if constexpr (std::is_integral_v<T>) {
        return x < static_cast<T>(0) ? -x : x;
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::abs(x);
    } else {
        throw std::invalid_argument("Data type not handled by any heuristic.");
    }
}

template <typename OutputContainer, typename Iterator>
OutputContainer abs_distances(const Iterator& data_first, const Iterator& data_last, const Iterator& other_first) {
    std::size_t n_elements = std::distance(data_first, data_last);

    auto abs_distances_values = OutputContainer(n_elements);

    for (std::size_t n = 0; n < n_elements; ++n) {
        abs_distances_values[n] = std::abs(*(data_first + n) - *(other_first + n));
    }
    return abs_distances_values;
}

template <typename TargetType, typename Iterator>
std::vector<TargetType> to_type(const Iterator& data_first, const Iterator& data_last) {
    using InputType = typename std::iterator_traits<Iterator>::value_type;

    if constexpr (std::is_same_v<InputType, TargetType>) {
        // If InputType is already TargetType, return a copy of the input range
        return std::vector<TargetType>(data_first, data_last);
    }
    // Input is not of type TargetType, convert to TargetType and return the result
    auto result = std::vector<TargetType>(std::distance(data_first, data_last));
    std::transform(data_first, data_last, result.begin(), [](const auto& x) { return static_cast<TargetType>(x); });
    return result;
}

template <typename Iterator>
bool are_containers_equal(Iterator first1, Iterator last1, Iterator first2, typename Iterator::value_type epsilon = 0) {
    while (first1 != last1) {
        if (*first1 != *first2 && abs(*first1 - *first2) > epsilon) {
            return false;
        }
        ++first1;
        ++first2;
    }
    return true;
}

template <typename Container>
bool are_containers_equal(const Container&                      first,
                          const Container&                      second,
                          const typename Container::value_type& epsilon = 0) {
    if (first.size() != second.size()) {
        return false;
    }
    return are_containers_equal(first.begin(), first.end(), second.begin(), epsilon);
}

template <typename Iterator1, typename Iterator2>
std::size_t count_matches(Iterator1 first1, Iterator1 last1, Iterator2 first2) {
    std::size_t count = 0;

    while (first1 != last1) {
        if (*(first1++) == *(first2++)) {
            ++count;
        }
    }
    return count;
}

template <typename Iterator1, typename Iterator2, typename T>
std::size_t count_matches_for_value(Iterator1 first1, Iterator1 last1, Iterator2 first2, const T& value) {
    std::size_t count = 0;

    while (first1 != last1) {
        if (*first2 == value && *first1 == *first2) {
            ++count;
        }
        ++first1;
        ++first2;
    }
    return count;
}

template <typename InputContainer, typename IndicesContainer>
InputContainer permutation_from_indices(const InputContainer& input, const IndicesContainer& indices) {
    if (input.size() != indices.size()) {
        throw std::invalid_argument("The number of elements in the input and indices containers must match.");
    }
    // Create a temporary container to store the elements
    auto swapped = InputContainer(input.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        swapped[i] = input[indices[i]];
    }
    return swapped;
}

template <typename InputContainer, typename IndicesContainer>
InputContainer range_permutation_from_indices(const InputContainer&   input,
                                              const IndicesContainer& indices,
                                              std::size_t             length) {
    if (input.size() / length != indices.size()) {
        throw std::invalid_argument(
            "The number of elements in the input and indices containers must match. The input size or the "
            "length you provided is wrong.");
    }
    // Create a copy container to store the elements (overlapping indices wont be copied)
    auto swapped = input;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        // indices must be non overlapping
        if (indices[i] != i) {
            std::copy(input.begin() + indices[i] * length,
                      input.begin() + indices[i] * length + length,
                      swapped.begin() + i * length);
        }
    }
    return swapped;
}

template <typename Iterator>
std::size_t get_n_samples(const Iterator& first, const Iterator& last, std::size_t n_features) {
    // get the total number of elements
    const std::size_t n_elements = std::distance(first, last);
    // divide by the number of features provided to get the number of samples (rows)
    assert(n_elements % n_features == 0 && "Input data missing values or wrong number of features specified.\n");

    return n_elements / n_features;
}

template <typename IteratorFloat>
std::vector<typename IteratorFloat::value_type> init_spatial_uniform(const IteratorFloat& data_first,
                                                                     const IteratorFloat& data_last,
                                                                     std::size_t          n_centroids,
                                                                     std::size_t          n_features) {
    using FloatType = typename IteratorFloat::value_type;

    const std::size_t n_samples = common::utils::get_n_samples(data_first, data_last, n_features);

    auto centroids = std::vector<FloatType>(n_centroids * n_features);

    // row vector buffers to keep track of min-max values
    auto min_buffer = std::vector<FloatType>(n_features, std::numeric_limits<FloatType>::max());
    auto max_buffer = std::vector<FloatType>(n_features, std::numeric_limits<FloatType>::min());
    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t j = 0; j < n_features; ++j) {
            const FloatType curr_elem = *(data_first + j + i * n_features);
            min_buffer[j]             = std::min(curr_elem, min_buffer[j]);
            max_buffer[j]             = std::max(curr_elem, max_buffer[j]);
        }
    }
    using uniform_distr_ptr = typename std::unique_ptr<math::random::uniform_distribution<FloatType>>;
    // initialize a uniform random generatore w.r.t. each feature
    auto random_buffer = std::vector<uniform_distr_ptr>(n_features);
    for (std::size_t f = 0; f < n_features; ++f) {
        random_buffer[f] =
            std::make_unique<math::random::uniform_distribution<FloatType>>(min_buffer[f], max_buffer[f]);
    }
    for (std::size_t k = 0; k < n_centroids; ++k) {
        for (std::size_t f = 0; f < n_features; ++f) {
            // generate a centroid position that lies in the [min, max] range
            centroids[f + k * n_features] = (*random_buffer[f])();
        }
    }
    return centroids;
}

/**
 * @brief might have overlaps when n_choices is close to the range size
 *
 * @param n_choices
 * @param indices_range
 * @return std::vector<std::size_t>
 */
inline std::vector<std::size_t> select_from_range(std::size_t                                n_choices,
                                                  const std::pair<std::size_t, std::size_t>& indices_range) {
    if (indices_range.second - indices_range.first < n_choices) {
        throw std::invalid_argument(
            "The number of random choice indices should be less or equal than the indices candidates.");
    }
    // the unique indices
    std::vector<std::size_t> random_distinct_indices;
    // keeps track of the indices that have already been generated as unique objects
    std::unordered_set<std::size_t> generated_indices;
    // range: [0, n_indices-1], upper bound included
    math::random::uniform_distribution<std::size_t> random_number_generator(indices_range.first,
                                                                            indices_range.second - 1);

    while (random_distinct_indices.size() < n_choices) {
        const auto index_candidate = random_number_generator();
        // check if the index candidate is already in the set and adds it to both containers if not
        if (generated_indices.find(index_candidate) == generated_indices.end()) {
            random_distinct_indices.emplace_back(index_candidate);
            generated_indices.insert(index_candidate);
        }
    }
    return random_distinct_indices;
}

/**
 * @brief not recommended for very large ranges since it will create a buffer for it that might take this amount of
 * memory but it wont have overlaps for n_choices ~= n_indices_candidates
 *
 * @param n_choices
 * @param indices_range
 * @return std::vector<std::size_t>
 */
inline std::vector<std::size_t> select_from_range_buffered(std::size_t                                n_choices,
                                                           const std::pair<std::size_t, std::size_t>& indices_range) {
    // indices_range upper bound excluded
    std::size_t n_indices_candidates = indices_range.second - indices_range.first;

    if (n_indices_candidates < n_choices) {
        throw std::invalid_argument(
            "The number of random choice indices should be less or equal than the indices candidates.");
    }
    // the unique indices
    std::vector<std::size_t> random_distinct_indices(n_choices);
    // generate the initial indices sequence which elements will be drawn from
    std::vector<std::size_t> initial_indices_candidates(n_indices_candidates);
    std::iota(initial_indices_candidates.begin(), initial_indices_candidates.end(), indices_range.first);

    for (auto& selected_index : random_distinct_indices) {
        // range: [0, N-1], upper bound is included
        math::random::uniform_distribution<std::size_t> random_number_generator(0,
                                                                                initial_indices_candidates.size() - 1);
        // generate the index of the indices vector
        const auto index_index = random_number_generator();
        // get the actual value
        const auto index_value = initial_indices_candidates[index_index];
        // save the index value
        selected_index = index_value;
        // and remove it from the candidates to make it unavalable for ther next iteration
        initial_indices_candidates.erase(initial_indices_candidates.begin() + index_index);
    }
    return random_distinct_indices;
}

template <typename IteratorFloat>
std::vector<typename IteratorFloat::value_type> init_uniform(const IteratorFloat& data_first,
                                                             const IteratorFloat& data_last,
                                                             std::size_t          n_centroids,
                                                             std::size_t          n_features) {
    using FloatType = typename IteratorFloat::value_type;

    const std::size_t n_samples = common::utils::get_n_samples(data_first, data_last, n_features);

    auto centroids = std::vector<FloatType>(n_centroids * n_features);

    const auto indices = select_from_range(n_centroids, {0, n_samples});

    for (std::size_t k = 0; k < n_centroids; ++k) {
        const auto idx = indices[k];
        for (std::size_t f = 0; f < n_features; ++f) {
            centroids[k * n_features + f] = *(data_first + idx * n_features + f);
        }
    }
    return centroids;
}

template <typename T = double, typename IteratorFloat>
std::vector<T> normalize_min_max(const IteratorFloat& data_first, const IteratorFloat& data_last) {
    const auto [min, max] = std::minmax_element(data_first, data_last);

    if (*min == *max) {
        const std::size_t n_elements = std::distance(data_first, data_last);
        // if all the values are the same distribute the weights equaly
        return std::vector<T>(n_elements, 1.0 / n_elements);
    }
    auto res = std::vector<T>(data_last - data_first);
    // closest objects get a higher score. Distance zero -> 1
    std::transform(data_first, data_last, res.begin(), [&min, &max](const auto& dist) {
        return static_cast<T>(1) - (dist - *min) / (*max - *min);
    });
    return res;
}

template <typename T = double, typename IteratorFloat>
std::vector<T> normalize(const IteratorFloat& data_first, const IteratorFloat& data_last) {
    auto normalized_vector = std::vector<T>(std::distance(data_first, data_last));

    // Calculate the sum of all elements in the vector
    const T sum = std::accumulate(data_first, data_last, static_cast<T>(0));
    // Divide each element by the sum to normalize the vector
    std::transform(data_first, data_last, normalized_vector.begin(), [sum](const T& val) { return val / sum; });

    return normalized_vector;
}

template <typename Iterator>
std::vector<typename Iterator::value_type> select_random_sample(const Iterator& data_first,
                                                                const Iterator& data_last,
                                                                std::size_t     n_features) {
    using DataType = typename Iterator::value_type;

    const auto n_samples = common::utils::get_n_samples(data_first, data_last, n_features);
    // selects an index w.r.t. an uniform random distribution [0, n_samples)
    auto index_select = math::random::uniform_distribution<std::size_t>(0, n_samples - 1);
    // pick the initial index that represents the first cluster
    std::size_t random_index = index_select();

    return std::vector<DataType>(data_first + random_index * n_features,
                                 data_first + random_index * n_features + n_features);
}

template <typename Iterator>
bool is_element_in(const Iterator&                      data_first,
                   const Iterator&                      data_last,
                   const typename Iterator::value_type& element) {
    // if std::find reaches the end of the container then nothing is found and it returns true
    // so we want is_element_in to return the opposite boolean value
    return !(std::find(data_first, data_last, element) == data_last);
}

template <typename Iterator>
bool is_element_not_in(const Iterator&                      data_first,
                       const Iterator&                      data_last,
                       const typename Iterator::value_type& element) {
    // if std::find reaches the end of the container then nothing is found and it returns true
    // so we want is_element_in to return the same boolean value
    return std::find(data_first, data_last, element) == data_last;
}

/**
 * @brief
 *
 * @tparam IteratorPair
 * @param data_first: beginning of iterator of std::container<std::pair<T, U>>
 * @param data_last: end of iterator of std::container<std::pair<T, U>>
 * @param element: some element of type T
 * @return true
 * @return false
 */
template <typename IteratorPair>
bool is_element_in_first(const IteratorPair&                                  data_first,
                         const IteratorPair&                                  data_last,
                         const typename IteratorPair::value_type::first_type& element) {
    // if std::find reaches the end of the container then nothing is found and it returns true
    // so we want is_element_in to return the opposite boolean value
    return !(std::find_if(data_first, data_last, [&element](const auto& pair) { return element == pair.first; }) ==
             data_last);
}

/**
 * @brief
 *
 * @tparam IteratorPair
 * @param data_first: beginning of iterator of std::container<std::pair<T, U>>
 * @param data_last: end of iterator of std::container<std::pair<T, U>>
 * @param element: some element of type T
 * @return true
 * @return false
 */
template <typename IteratorPair>
bool is_element_not_in_first(const IteratorPair&                                  data_first,
                             const IteratorPair&                                  data_last,
                             const typename IteratorPair::value_type::first_type& element) {
    // if std::find reaches the end of the container then nothing is found and it returns true
    // so we want is_element_in to return the same boolean value
    return std::find_if(data_first, data_last, [&element](const auto& pair) { return element == pair.first; }) ==
           data_last;
}

template <typename Container>
void min_container_inplace_left(Container& lhs, const Container& rhs) {
    // compute element-wise minimum and place the result in the left container
    std::transform(
        lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), [](const auto& x, const auto& y) { return x < y ? x : y; });
}

template <typename Iterator>
typename Iterator::value_type argmin(const Iterator& data_first, const Iterator& data_last) {
    auto min_it = std::min_element(data_first, data_last);
    return std::distance(data_first, min_it);
}

template <typename Iterator>
typename Iterator::value_type argmax(const Iterator& data_first, const Iterator& data_last) {
    auto max_it = std::max_element(data_first, data_last);
    return std::distance(data_first, max_it);
}

template <typename Iterator>
std::pair<std::size_t, typename Iterator::value_type> get_min_index_value_pair(const Iterator& data_first,
                                                                               const Iterator& data_last) {
    using Type = typename Iterator::value_type;

    const auto  min_element = std::min_element(data_first, data_last);
    std::size_t min_index   = std::distance(data_first, min_element);
    const Type  min_value   = *min_element;

    return {min_index, min_value};
}

template <typename Iterator>
std::pair<std::size_t, typename Iterator::value_type> get_max_index_value_pair(const Iterator& data_first,
                                                                               const Iterator& data_last) {
    using Type = typename Iterator::value_type;

    const auto  max_element = std::max_element(data_first, data_last);
    std::size_t max_index   = std::distance(data_first, max_element);
    const Type  max_value   = *max_element;

    return {max_index, max_value};
}

template <typename Iterator>
std::pair<std::size_t, typename Iterator::value_type> get_second_max_index_value_pair(const Iterator& data_first,
                                                                                      const Iterator& data_last) {
    using Type = typename Iterator::value_type;

    // Get the index of the maximum element
    auto        max_it    = std::max_element(data_first, data_last);
    std::size_t max_idx   = std::distance(data_first, max_it);
    const Type  max_value = *max_it;

    if (std::distance(data_first, data_last) < 2) {
        return {max_idx, max_value};
    }
    // Replace the maximum element with the lowest possible value
    *max_it = std::numeric_limits<Type>::lowest();

    // Get the index of the second maximum element
    const auto  second_max_it    = std::max_element(data_first, data_last);
    std::size_t second_max_idx   = std::distance(data_first, second_max_it);
    const Type  second_max_value = *second_max_it;

    // Restore the original maximum value
    *max_it = max_value;

    return {second_max_idx, second_max_value};
}

template <typename Iterator>
std::pair<std::size_t, typename Iterator::value_type> find_nth_smallest_index_and_element(const Iterator& data_first,
                                                                                          const Iterator& data_last,
                                                                                          std::size_t     n) {
    using Type = typename Iterator::value_type;

    if (n > static_cast<std::size_t>(data_last - data_first)) {
        throw std::invalid_argument("N-th smallest requested element shouldn't be greater that the container's size.");
    }
    std::vector<Type> data_sorted(data_last - data_first);
    std::copy(data_first, data_last, data_sorted.begin());
    std::sort(data_sorted.begin(), data_sorted.end());

    // get the n-th smallest element
    auto nth_smallest = data_sorted.at(n - 1);

    // find the index of the n-th smallest element in the original container
    auto it    = std::find(data_first, data_last, nth_smallest);
    auto index = std::distance(data_first, it);

    return {index, nth_smallest};
}

template <typename Iterator, typename IteratorInt>
void update_except_at(const Iterator&    target_first,
                      const Iterator&    target_last,
                      const Iterator&    source_first,
                      const IteratorInt& invalid_indices_first,
                      const IteratorInt& invalid_indices_last) {
    static_assert(std::is_integral<typename IteratorInt::value_type>::value, "Data should be integer type.");

    const std::size_t n_elements = target_last - target_first;

    for (std::size_t idx = 0; idx < n_elements; ++idx) {
        // enter condition only if the current index is not in the invalid indices
        if (common::utils::is_element_not_in(invalid_indices_first, invalid_indices_last, idx)) {
            // update the target vector if its the case
            *(target_first + idx) = *(source_first + idx);
        }
    }
}

template <typename RandomAccessIterator, typename RangeComparisonFunction>
std::size_t median_index_of_three_ranges(RandomAccessIterator           first,
                                         RandomAccessIterator           last,
                                         std::size_t                    n_features,
                                         const RangeComparisonFunction& compare_ranges) {
    const std::size_t n_samples    = common::utils::get_n_samples(first, last, n_features);
    std::size_t       middle_index = n_samples / 2;

    if (n_samples < 3) {
        return middle_index;
    }
    std::size_t left_index  = 0;
    std::size_t right_index = n_samples - 1;

    if (compare_ranges(first + middle_index * n_features, first + left_index * n_features)) {
        std::swap(middle_index, left_index);
    }
    if (compare_ranges(first + right_index * n_features, first + left_index * n_features)) {
        std::swap(right_index, left_index);
    }
    if (compare_ranges(first + middle_index * n_features, first + right_index * n_features)) {
        std::swap(middle_index, right_index);
    }
    return right_index;
}

template <typename RandomAccessIterator, typename RangeComparisonFunction>
std::pair<RandomAccessIterator, RandomAccessIterator> median_values_range_of_three_ranges(
    RandomAccessIterator           first,
    RandomAccessIterator           last,
    std::size_t                    n_features,
    const RangeComparisonFunction& compare_ranges) {
    const std::size_t median_index = median_index_of_three_ranges(first, last, n_features, compare_ranges);

    return {first + median_index * n_features, first + median_index * n_features + n_features};
}

template <typename RandomAccessIterator, typename RangeComparisonFunction>
std::pair<std::size_t, std::pair<RandomAccessIterator, RandomAccessIterator>> median_values_range_of_three_ranges(
    RandomAccessIterator           first,
    RandomAccessIterator           last,
    std::size_t                    n_features,
    const RangeComparisonFunction& compare_ranges) {
    const std::size_t median_index = median_index_of_three_ranges(first, last, n_features, compare_ranges);

    return {median_index, {first + median_index * n_features, first + median_index * n_features + n_features}};
}

/**
 * @brief Hoare partition scheme algorithm.
 * This implementation will also swap ranges that are evaluated as equal.
 * However, it will ensure that the indices dont move out of range, without checks.
 */
template <typename RandomAccessIterator, typename RangeComparisonFunction>
std::size_t partition_around_nth_range(RandomAccessIterator           first,
                                       RandomAccessIterator           last,
                                       std::size_t                    pivot_index,
                                       std::size_t                    n_features,
                                       const RangeComparisonFunction& compare_ranges) {
    // no op if the input contains only one feature vector
    if (std::distance(first, last) <= static_cast<std::ptrdiff_t>(n_features)) {
        return pivot_index;
    }
    // start out of bound so that the indices never go out of bounds when (in/de)cremented
    ssize_t left_index  = -1;
    ssize_t right_index = common::utils::get_n_samples(first, last, n_features);

    while (true) {
        do {
            ++left_index;
        } while (compare_ranges(first + left_index * n_features, first + pivot_index * n_features));

        do {
            --right_index;
        } while (compare_ranges(first + pivot_index * n_features, first + right_index * n_features));

        if (left_index >= right_index) {
            break;
        }
        std::swap_ranges(first + left_index * n_features,
                         first + left_index * n_features + n_features,
                         first + right_index * n_features);

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
    }
    return pivot_index;
}

/**
 * @brief A quickselect implementation https://en.wikipedia.org/wiki/Quickselect
 */
template <typename RandomAccessIterator, typename RangeComparisonFunction>
std::pair<RandomAccessIterator, RandomAccessIterator> quickselect_range(RandomAccessIterator           first,
                                                                        RandomAccessIterator           last,
                                                                        std::size_t                    kth_smallest,
                                                                        std::size_t                    n_features,
                                                                        const RangeComparisonFunction& compare_ranges) {
    std::size_t left_index  = 0;
    std::size_t right_index = common::utils::get_n_samples(first, last, n_features) - 1;

    while (true) {
        if (left_index == right_index) {
            return {first + left_index * n_features, first + right_index * n_features + n_features};
        }
        // std::size_t pivot_index = n_samples / 2;
        std::size_t pivot_index = median_index_of_three_ranges(
            first + left_index * n_features, first + right_index * n_features + n_features, n_features, compare_ranges);

        // partition the range around the pivot, which has moved to its sorted index
        // the pivot index starts from the left_index, so we need to shift it by the same amount
        pivot_index = left_index + partition_around_nth_range(first + left_index * n_features,
                                                              first + right_index * n_features + n_features,
                                                              pivot_index,
                                                              n_features,
                                                              compare_ranges);

        if (kth_smallest == pivot_index) {
            return {first + pivot_index * n_features, first + pivot_index * n_features + n_features};

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
template <typename RandomAccessIterator, typename RangeComparisonFunction>
std::size_t quicksort_range(RandomAccessIterator           first,
                            RandomAccessIterator           last,
                            std::size_t                    initial_pivot_index,
                            std::size_t                    n_features,
                            const RangeComparisonFunction& compare_ranges) {
    const std::size_t n_samples = common::utils::get_n_samples(first, last, n_features);

    // the pivot index is already correct if the number of samples is 0 or 1
    if (n_samples < 2) {
        return initial_pivot_index;
    }
    // partial sort ranges around new pivot index
    const std::size_t new_pivot_index =
        partition_around_nth_range(first, last, initial_pivot_index, n_features, compare_ranges);

    // compute the median of the subranges

    std::size_t pivot_index_subrange_left =
        median_index_of_three_ranges(first, first + new_pivot_index * n_features, n_features, compare_ranges);

    std::size_t pivot_index_subrange_right = median_index_of_three_ranges(
        first + new_pivot_index * n_features + n_features, last, n_features, compare_ranges);

    // the pivot range is included in the left subrange

    quicksort_range(/*first iterator, left subrange*/ first,
                    /*last iterator, left subrange*/ first + new_pivot_index * n_features,
                    pivot_index_subrange_left,
                    n_features,
                    compare_ranges);

    quicksort_range(/*first iterator, right subrange*/ first + new_pivot_index * n_features + n_features,
                    /*last iterator, right subrange*/ last,
                    pivot_index_subrange_right,
                    n_features,
                    compare_ranges);

    return new_pivot_index;
}

}  // namespace common::utils