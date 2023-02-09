#pragma once

#include "cpp_clustering/math/random/Distributions.hpp"

#include <algorithm>
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
    if (n_elements % n_features != 0) {
        throw std::invalid_argument("Input data missing values or wrong number of features specified.\n");
    }
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

}  // namespace common::utils