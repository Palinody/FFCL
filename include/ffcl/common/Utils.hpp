#pragma once

#include "ffcl/math/random/Distributions.hpp"

#include <sys/types.h>  // ssize_t
#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace common::utils {

template <typename... Args>
constexpr void ignore_parameters(Args&&...) noexcept {}

template <typename T>
static constexpr T infinity() {
    return std::numeric_limits<T>::max();
}

template <typename T>
constexpr T abs(const T& x) {
    if constexpr (std::is_integral_v<T>) {
        return x < static_cast<T>(0) ? -x : x;

    } else if constexpr (std::is_floating_point_v<T>) {
        return std::fabs(x);

    } else {
        throw std::invalid_argument("Data type not handled by any heuristic.");
    }
}

template <typename T, typename U>
constexpr bool equality(const T& a, const U& b) noexcept {
    if constexpr (std::is_integral_v<T> && std::is_integral_v<U>) {
        return a == b;

    } else if constexpr (std::is_floating_point_v<T> && std::is_floating_point_v<U>) {
        return std::fabs(b - a) <= std::numeric_limits<decltype(b - a)>::epsilon();

    } else if constexpr (std::is_floating_point_v<T> || std::is_floating_point_v<U>) {
        return std::abs(b - a) <= std::numeric_limits<decltype(b - a)>::epsilon();

    } else {
        static_assert(std::is_same_v<T, U>, "(in)equality comparison only supported for comparable types");
        return a == b;
    }
}

template <typename T, typename U>
constexpr bool inequality(const T& a, const U& b) noexcept {
    return !equality(a, b);
}

template <typename T, typename U>
constexpr auto division(const T& a, const U& b) -> std::common_type_t<T, U> {
    if (equality(b, static_cast<U>(0))) {
#if defined(VERBOSE) && VERBOSE == true
        printf("[WARN] attempted division by zero: returning zero.\n");
#endif
        return 0;
    }
    return a / b;
}

template <typename OutputContainer, typename Iterator>
OutputContainer abs_distances(const Iterator& data_first, const Iterator& data_last, const Iterator& other_first) {
    std::size_t n_elements = std::distance(data_first, data_last);

    auto abs_distances_values = OutputContainer(n_elements);

    for (std::size_t n = 0; n < n_elements; ++n) {
        abs_distances_values[n] = std::abs(data_first[n] - other_first[n]);
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
bool are_containers_equal(
    Iterator                      first1,
    Iterator                      last1,
    Iterator                      first2,
    typename Iterator::value_type tolerance = std::numeric_limits<typename Iterator::value_type>::epsilon()) {
    using InputType = typename Iterator::value_type;
    while (first1 != last1) {
        if constexpr (std::is_integral_v<InputType>) {
            if (*first1 != *first2) {
                return false;
            }
        } else if constexpr (std::is_floating_point_v<InputType>) {
            if (std::abs(*first1 - *first2) > tolerance) {
                return false;
            }
        } else {
            // Unsupported type
            static_assert(std::is_integral_v<InputType> || std::is_floating_point_v<InputType>, "Unsupported type");
        }
        ++first1;
        ++first2;
    }
    ignore_parameters(tolerance);
    return true;
}

template <typename Container>
bool are_containers_equal(
    const Container&                      first,
    const Container&                      second,
    const typename Container::value_type& tolerance = std::numeric_limits<typename Container::value_type>::epsilon()) {
    if (first.size() != second.size()) {
        ignore_parameters(tolerance);
        return false;
    }
    return are_containers_equal(first.begin(), first.end(), second.begin(), tolerance);
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

template <typename IndicesContainer, typename InputContainer>
InputContainer permutation_from_indices(const IndicesContainer& indices, const InputContainer& input) {
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

template <typename IndicesContainer, typename InputContainer>
InputContainer remap_ranges_from_indices(const IndicesContainer& indices,
                                         const InputContainer&   flattened_matrix,
                                         std::size_t             n_features) {
    using IndicesType = typename IndicesContainer::value_type;

    static_assert(std::is_integral_v<IndicesType>);

    if (flattened_matrix.size() / n_features != indices.size()) {
        throw std::invalid_argument("The number of elements in the flattened_matrix and indices containers must match. "
                                    "The flattened_matrix size or the "
                                    "n_features you provided is wrong.");
    }
    // Create a copy container to store the elements (overlapping indices wont be copied)
    auto swapped = flattened_matrix;
    for (IndicesType i = 0; i < static_cast<IndicesType>(indices.size()); ++i) {
        // indices must be non overlapping
        if (indices[i] != i) {
            std::copy(flattened_matrix.begin() + indices[i] * n_features,
                      flattened_matrix.begin() + indices[i] * n_features + n_features,
                      swapped.begin() + i * n_features);
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
    return !(std::find_if(data_first, data_last, [&element](const auto& pair) {
                 return equality(element, pair.first);
             }) == data_last);
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
    return std::find_if(data_first, data_last, [&element](const auto& pair) {
               return equality(element, pair.first);
           }) == data_last;
}

template <typename DataType>
std::vector<DataType> generate_values(const DataType& value_first, const DataType& value_last) {
    assert(value_last >= value_first);

    std::vector<DataType> elements(static_cast<std::size_t>(value_last - value_first));
    // construct the range
    std::iota(elements.begin(), elements.end(), value_first);
    return elements;
}

}  // namespace common::utils