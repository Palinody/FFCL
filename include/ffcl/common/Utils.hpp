#pragma once

#include <sys/types.h>  // ssize_t
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>  // std::size_t
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <iostream>

namespace ffcl::common {

// Primary template for general types (non-pointers)
template <typename T, typename = void>
struct remove_pointer {
    using type = T;
};

// Specialization for raw pointers
template <typename T>
struct remove_pointer<T, std::enable_if_t<std::is_pointer_v<T>>> {
    using type = std::remove_pointer_t<T>;
};

// Specialization for std::unique_ptr
template <typename T>
struct remove_pointer<std::unique_ptr<T>> {
    using type = T;
};

// Specialization for std::shared_ptr
template <typename T>
struct remove_pointer<std::shared_ptr<T>> {
    using type = T;
};

// Helper type alias
template <typename T>
using remove_pointer_t = typename remove_pointer<T>::type;

// A type trait to check for CRTP inheritance
template <typename Derived, template <typename> class Base>
class is_crtp_of {
  private:
    // We create two types: one that always exists (yes), and one that only exists on success (no).
    // The check(...) will always be chosen if it's valid, thanks to the ellipsis which has the lowest precedence.
    using yes = char;
    // 'no' type will be larger than 'yes'
    struct no {
        yes m[2];
    };
    // This check will be valid only if you can static_cast from Derived* to Base<Derived>*.
    // This is only possible if Derived inherits from Base<Derived>.
    static yes test(Base<Derived>*);
    static no  test(...);

  public:
    // The size of test<U>(nullptr) will be sizeof(yes) if Derived inherits from Base<Derived>
    // and sizeof(no) otherwise. We compare it with sizeof(yes) to find out if the inheritance is true.
    static constexpr bool value = sizeof(test(static_cast<Derived*>(nullptr))) == sizeof(yes);
};

// BEGIN: is_raw_or_smart_ptr
// Helper template to check if a type is a smart pointer (std::unique_ptr, std::shared_ptr, std::weak_ptr)
template <typename T>
struct is_smart_ptr : std::false_type {};

template <typename T>
struct is_smart_ptr<std::unique_ptr<T>> : std::true_type {};

template <typename T>
struct is_smart_ptr<std::shared_ptr<T>> : std::true_type {};

template <typename T>
struct is_smart_ptr<std::weak_ptr<T>> : std::true_type {};

// Function to check if T is a raw or smart pointer
template <typename T>
constexpr bool is_raw_or_smart_ptr() {
    return std::is_pointer<T>::value || is_smart_ptr<T>::value;
}
// END: is_raw_or_smart_ptr

// BEGIN: is_iterator_v
// Helper to check for the presence of nested types typical for iterators
template <typename T, typename = void>
struct is_iterator : std::false_type {};

template <typename T>
struct is_iterator<T,
                   std::void_t<typename std::iterator_traits<T>::difference_type,
                               typename std::iterator_traits<T>::value_type,
                               typename std::iterator_traits<T>::pointer,
                               typename std::iterator_traits<T>::reference,
                               typename std::iterator_traits<T>::iterator_category>> : std::true_type {};

template <typename T>
constexpr bool is_iterator_v() {
    return is_iterator<T>::value;
}
// END: is_iterator_v

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
constexpr auto division(const T& numerator, const U& denominator, const U& default_return_value = 0)
    -> std::common_type_t<T, U> {
    if (equality(denominator, static_cast<U>(0))) {
#if defined(VERBOSE) && VERBOSE == true
        printf("[WARN] attempted division by zero: returning default_return_value.\n");
#endif
        return default_return_value;
    }
    return numerator / denominator;
}

template <typename TargetType, typename DataIterator>
std::vector<TargetType> to_type(const DataIterator& data_range_first, const DataIterator& data_range_last) {
    using InputType = typename std::iterator_traits<DataIterator>::value_type;

    if constexpr (std::is_same_v<InputType, TargetType>) {
        // If InputType is already TargetType, return a copy of the input range
        return std::vector<TargetType>(data_range_first, data_range_last);
    }
    // Input is not of type TargetType, convert to TargetType and return the result
    auto result = std::vector<TargetType>(std::distance(data_range_first, data_range_last));

    std::transform(data_range_first, data_range_last, result.begin(), [](const auto& data) {
        return static_cast<TargetType>(data);
    });

    return result;
}

template <typename DataIterator>
bool are_containers_equal(DataIterator                                            left_data_range_first,
                          DataIterator                                            left_data_range_last,
                          DataIterator                                            right_data_range_first,
                          typename std::iterator_traits<DataIterator>::value_type tolerance =
                              std::numeric_limits<typename std::iterator_traits<DataIterator>::value_type>::epsilon()) {
    using InputType = typename std::iterator_traits<DataIterator>::value_type;
    while (left_data_range_first != left_data_range_last) {
        if constexpr (std::is_integral_v<InputType>) {
            if (*left_data_range_first != *right_data_range_first) {
                return false;
            }
        } else if constexpr (std::is_floating_point_v<InputType>) {
            if (std::abs(*left_data_range_first - *right_data_range_first) > tolerance) {
                return false;
            }
        } else {
            static_assert(std::is_integral_v<InputType> || std::is_floating_point_v<InputType>, "Unsupported type");
        }
        ++left_data_range_first;
        ++right_data_range_first;
    }
    ignore_parameters(tolerance);
    return true;
}

template <typename Container>
bool are_containers_equal(
    const Container&                      left_container,
    const Container&                      right_container,
    const typename Container::value_type& tolerance = std::numeric_limits<typename Container::value_type>::epsilon()) {
    if (left_container.size() != right_container.size()) {
        ignore_parameters(tolerance);
        return false;
    }
    return are_containers_equal(left_container.begin(), left_container.end(), right_container.begin(), tolerance);
}

template <typename LeftDataIterator, typename RightDataIterator>
std::size_t count_matches(LeftDataIterator  left_data_range_first,
                          LeftDataIterator  left_data_range_last,
                          RightDataIterator right_data_range_first) {
    std::size_t count = 0;

    while (left_data_range_first != left_data_range_last) {
        if (*(left_data_range_first++) == *(right_data_range_first++)) {
            ++count;
        }
    }
    return count;
}

template <typename LeftDataIterator, typename RightDataIterator, typename T>
std::size_t count_matches_for_value(LeftDataIterator  left_data_range_first,
                                    LeftDataIterator  left_data_range_last,
                                    RightDataIterator right_data_range_first,
                                    const T&          value) {
    std::size_t count = 0;

    while (left_data_range_first != left_data_range_last) {
        if (*right_data_range_first == value && *left_data_range_first == *right_data_range_first) {
            ++count;
        }
        ++left_data_range_first;
        ++right_data_range_first;
    }
    return count;
}

template <typename IndicesContainer, typename InputContainer>
InputContainer permutation_from_indices(const IndicesContainer& indices, const InputContainer& input) {
    if (input.size() != indices.size()) {
        throw std::invalid_argument("The number of elements in the input and indices datastruct must match.");
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
        throw std::invalid_argument("The number of elements in the flattened_matrix and indices datastruct must match. "
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

template <typename DataIterator>
std::size_t get_n_samples(const DataIterator& data_range_first,
                          const DataIterator& data_range_last,
                          std::size_t         n_features) {
    // get the total number of elements
    const std::size_t n_elements = std::distance(data_range_first, data_range_last);
    // divide by the number of features provided to get the number of samples (rows)
    assert(n_elements % n_features == 0 && "Input data missing values or wrong number of features specified.\n");

    return n_elements / n_features;
}

template <typename DataIterator>
bool is_element_in(const DataIterator&                                            data_range_first,
                   const DataIterator&                                            data_range_last,
                   const typename std::iterator_traits<DataIterator>::value_type& element) {
    // if std::find reaches the end of the container then nothing is found and it returns true
    // so we want is_element_in to return the opposite boolean value
    return !(std::find(data_range_first, data_range_last, element) == data_range_last);
}

template <typename DataIterator>
bool is_element_not_in(const DataIterator&                                            data_range_first,
                       const DataIterator&                                            data_range_last,
                       const typename std::iterator_traits<DataIterator>::value_type& element) {
    // if std::find reaches the end of the container then nothing is found and it returns true
    // so we want is_element_in to return the same boolean value
    return std::find(data_range_first, data_range_last, element) == data_range_last;
}

/**
 * @brief
 *
 * @tparam PairIterator
 * @param pairs_range_first: beginning of iterator of std::container<std::pair<T, U>>
 * @param pairs_range_last: end of iterator of std::container<std::pair<T, U>>
 * @param element: some element of type T
 * @return true
 * @return false
 */
template <typename PairIterator>
bool is_element_in_first(const PairIterator&                                  pairs_range_first,
                         const PairIterator&                                  pairs_range_last,
                         const typename PairIterator::value_type::first_type& element) {
    // if std::find reaches the end of the container then nothing is found and it returns true
    // so we want is_element_in to return the opposite boolean value
    return !(std::find_if(pairs_range_first, pairs_range_last, [&element](const auto& pair) {
                 return equality(element, pair.first);
             }) == pairs_range_last);
}

/**
 * @brief
 *
 * @tparam PairIterator
 * @param pairs_range_first: beginning of iterator of std::container<std::pair<T, U>>
 * @param pairs_range_last: end of iterator of std::container<std::pair<T, U>>
 * @param element: some element of type T
 * @return true
 * @return false
 */
template <typename PairIterator>
bool is_element_not_in_first(const PairIterator&                                  pairs_range_first,
                             const PairIterator&                                  pairs_range_last,
                             const typename PairIterator::value_type::first_type& element) {
    // if std::find reaches the end of the container then nothing is found and it returns true
    // so we want is_element_in to return the same boolean value
    return std::find_if(pairs_range_first, pairs_range_last, [&element](const auto& pair) {
               return equality(element, pair.first);
           }) == pairs_range_last;
}

template <typename DataType>
void print_flattened_vector_as_matrix(const std::vector<DataType>& data, std::size_t n_features) {
    const std::size_t n_samples = data.size() / n_features;

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            std::cout << data[sample_index * n_features + feature_index] << " ";
        }
        std::cout << "\n";
    }
}

}  // namespace ffcl::common