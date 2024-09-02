#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>  // std::size_t
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <iostream>

namespace ffcl::common {

/*
// Custom assertion with printing
#define ASSERT_WITH_MSG(condition, msg, ...)          \
    do {                                              \
        if (!(condition)) {                           \
            std::fprintf(stderr, (msg), __VA_ARGS__); \
            std::abort();                             \
        }                                             \
    } while (0)

ASSERT_WITH_MSG(inequality(min_distance, infinity<DistanceType>()),
                "Assertion failed: %s, query_node size: %ld, reference_node size: %ld\n",
                "min_distance != infinity",
                static_cast<long>(query_node->n_samples()),
                static_cast<long>(reference_node->n_samples()));
*/

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

// Primary template that defaults to false.
template <typename, typename Class, typename... Args>
struct does_constructor_signature_match_impl : std::false_type {};

// Helper function to try instantiation.
template <typename Class, typename... Args, typename = decltype(Class(std::declval<Args>()...))>
constexpr bool try_instantiate(int) {
    return true;
}

// Fallback if try_instantiate fails.
template <typename, typename...>
constexpr bool try_instantiate(...) {
    return false;
}

// Specialization that utilizes the helper function
template <typename Class, typename... Args>
struct does_constructor_signature_match_impl<std::void_t<decltype(try_instantiate<Class, Args...>(0))>, Class, Args...>
  : std::integral_constant<bool, try_instantiate<Class, Args...>(0)> {};

// User template that uses does_constructor_signature_match_impl, which enables template specialization.
template <typename Class, typename... Args>
using does_signature_match_with = does_constructor_signature_match_impl<void, Class, Args...>;

// Helper variable template for ease of use
template <typename Class, typename... Args>
inline constexpr bool is_constructible_with_v = does_signature_match_with<Class, Args...>::value;

// ---

// Attempt to select a constructible type from a list, or void if none match.
template <typename... Classes>
struct select_constructible_type;

template <typename FirstClass, typename... OtherClasses>
struct select_constructible_type<FirstClass, OtherClasses...> {
    template <typename... Args>
    struct from_signature {
        using type = typename std::conditional_t<
            std::is_constructible_v<FirstClass, Args...>,
            FirstClass,
            typename select_constructible_type<OtherClasses...>::template from_signature<Args...>::type>;
    };
};

// Specialization for when there are no classes left to check. Fallback to void.
template <>
struct select_constructible_type<> {
    template <typename... Args>
    struct from_signature {
        using type = void;
    };
};

template <typename... Classes>
struct select_trivially_constructible_type;

template <typename FirstClass, typename... OtherClasses>
struct select_trivially_constructible_type<FirstClass, OtherClasses...> {
    template <typename... Args>
    struct from_signature {
        using type = typename std::conditional_t<
            std::is_trivially_constructible_v<FirstClass, Args...>,
            FirstClass,
            typename select_trivially_constructible_type<OtherClasses...>::template from_signature<Args...>::type>;
    };
};

// Specialization for when there are no classes left to check. Fallback to void.
template <>
struct select_trivially_constructible_type<> {
    template <typename... Args>
    struct from_signature {
        using type = void;
    };
};

template <typename... Classes>
struct select_nothrow_constructible_type;

template <typename FirstClass, typename... OtherClasses>
struct select_nothrow_constructible_type<FirstClass, OtherClasses...> {
    template <typename... Args>
    struct from_signature {
        using type = typename std::conditional_t<
            std::is_nothrow_constructible_v<FirstClass, Args...>,
            FirstClass,
            typename select_nothrow_constructible_type<OtherClasses...>::template from_signature<Args...>::type>;
    };
};

// Specialization for when there are no classes left to check. Fallback to void.
template <>
struct select_nothrow_constructible_type<> {
    template <typename... Args>
    struct from_signature {
        using type = void;
    };
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

template <typename T>
inline constexpr bool is_raw_or_smart_ptr = std::is_pointer<T>::value || is_smart_ptr<T>::value;

// EXAMPLE USAGE
/*
static_assert(is_raw_or_smart_ptr<int*>, "int* should be considered a raw or smart pointer");
static_assert(is_raw_or_smart_ptr<std::unique_ptr<int>>,
              "std::unique_ptr<int> should be considered a raw or smart pointer");
static_assert(is_raw_or_smart_ptr<std::shared_ptr<int>>,
              "std::shared_ptr<int> should be considered a raw or smart pointer");
static_assert(!is_raw_or_smart_ptr<int>, "int should not be considered a raw or smart pointer");
*/
// END: is_raw_or_smart_ptr

// BEGIN: is_iterator_v
// Helper to check for the presence of nested types typical for iterators
template <typename T, typename = void>
struct is_iterator {
    static constexpr bool value = false;
};

template <typename T>
struct is_iterator<
    T,
    typename std::enable_if<!std::is_same<typename std::iterator_traits<T>::value_type, void>::value>::type> {
    static constexpr bool value = true;
};

template <typename T>
inline constexpr bool is_iterator_v = is_iterator<T>::value;

// Example usage
/*
static_assert(is_iterator_v<std::vector<int>::iterator>, "std::vector<int>::iterator should be an iterator");
static_assert(is_iterator_v<std::list<int>::iterator>, "std::list<int>::iterator should be an iterator");
static_assert(is_iterator_v<std::map<int, int>::iterator>, "std::map<int, int>::iterator should be an iterator");
static_assert(!is_iterator_v<int>, "int should not be an iterator");
static_assert(!is_iterator_v<std::vector<int>>, "std::vector<int> should not be an iterator");
*/
// END: is_iterator_v

// BEGIN: is_std_container
template <typename T>
struct is_std_container {
    template <typename U>
    static std::true_type test(typename U::iterator*);

    template <typename U>
    static std::false_type test(...);

    static constexpr bool value = decltype(test<T>(nullptr))::value;
};

template <typename T>
inline constexpr bool is_std_container_v = is_std_container<T>::value;

// Example usage
/*
static_assert(is_std_container_v<std::vector<int>>, "std::vector<int> is a container");
static_assert(!is_std_container_v<int>, "int is not a container");
*/
// END is_std_container

template <typename T, typename = void>
struct is_iterable : std::false_type {
    using value_type = void;
};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>>
  : std::true_type {
    using value_type = typename std::decay_t<T>::value_type;
};

template <typename T>
inline constexpr bool is_iterable_v = is_iterable<T>::value;

/*
static_assert(!is_iterable_v<int>, "int should not be iterable");
static_assert(is_iterable_v<std::vector<int>>, "std::vector<int> should be iterable");
static_assert(is_iterable_v<std::map<int, std::vector<int>>>,
              "is_iterable_v<std::map<int, std::vector<int>>> should be iterable");
static_assert(is_iterable_v<std::vector<int*>>, "std::vector<int*> should be iterable");
*/

// New trait to check if the iterable's value_type inherits TargetCRTP
template <typename Iterable, template <typename> class TargetCRTP, typename = void>
struct is_iterable_of_static_base : std::false_type {};

template <typename Iterable, template <typename> class TargetCRTP>
struct is_iterable_of_static_base<
    Iterable,
    TargetCRTP,
    std::void_t<
        std::enable_if_t<is_iterable_v<Iterable> && is_crtp_of<typename Iterable::value_type, TargetCRTP>::value>>>
  : std::true_type {};

template <typename T, typename = void>
struct nested_iterable_depth : std::integral_constant<std::size_t, 0> {};

template <typename T>
struct nested_iterable_depth<T, std::enable_if_t<is_iterable_v<T>>>
  : std::integral_constant<std::size_t, 1 + nested_iterable_depth<typename is_iterable<T>::value_type>::value> {};

template <typename T>
inline constexpr std::size_t nested_iterable_depth_v = nested_iterable_depth<T>::value;

/*
static_assert(nested_iterable_depth_v<std::vector<int>> == 1, "std::vector<int> depth should be 1");
static_assert(nested_iterable_depth_v<std::vector<std::vector<int>>> == 2,
              "std::vector<std::vector<int>> depth should be 2");
static_assert(nested_iterable_depth_v<std::vector<std::map<int, std::string>>> == 2,
              "std::vector<std::map<int, std::string>> depth should be 2");
*/

// Primary template for general types (base case, assumes T is not a container).
template <typename T, typename = void>
struct leaf_value_type {
    using type = T;
};

// Specialization for types that are iterable (have a value_type member).
template <typename T>
struct leaf_value_type<T, std::void_t<typename T::value_type>> {
    using type = typename leaf_value_type<typename T::value_type>::type;
};

template <typename T>
using leaf_value_type_t = typename leaf_value_type<T>::type;

/*
static_assert(std::is_same_v<leaf_value_type_t<int>, int>, "int nested value type should be int");
static_assert(std::is_same_v<leaf_value_type_t<float*>, float*>, "float* nested value type should be float*");
static_assert(std::is_same_v<leaf_value_type_t<std::vector<int>>, int>,
              "std::vector<int> nested value type should be int");
static_assert(std::is_same_v<leaf_value_type_t<std::vector<std::vector<std::vector<float>>>>, float>,
              "std::vector<std::vector<std::vector<float>>>> nested value type should be float");
static_assert(std::is_same_v<leaf_value_type_t<std::vector<int>>, int>,
              "std::vector<int> nested value type should be int");
static_assert(!std::is_same_v<leaf_value_type_t<std::vector<int*>>, int>,
              "std::vector<int*> nested value type should not be int");
*/

// Utility to get the value type of an iterable
template <typename T>
struct iterable_value_type {
    using type = typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type;
};

template <typename T>
using iterable_value_type_t = typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type;

// Utility to check if a type is homogeneous
template <typename T, typename = void>
struct is_leaf_homogeneous : std::is_trivial<T> {};  // True for trivial types

// Specialization for iterable types
template <typename T>
struct is_leaf_homogeneous<T, std::enable_if_t<is_iterable<T>::value>> : is_leaf_homogeneous<iterable_value_type_t<T>> {
};  // recursive call until it reaches a non iterable type. Returns false if the type is not trivial.

template <typename T>
inline constexpr bool is_leaf_homogeneous_v = is_leaf_homogeneous<T>::value;

/*
static_assert(is_leaf_homogeneous_v<int>, "int should be homogeneous");
static_assert(is_leaf_homogeneous_v<std::vector<int>>, "std::vector<int> should be homogeneous");
static_assert(is_leaf_homogeneous_v<std::vector<std::vector<int>>>,
              "std::vector<std::vector<int>> should be homogeneous");
static_assert(is_leaf_homogeneous_v<std::vector<std::vector<int*>>>,
              "std::vector<std::vector<int*>> should be homogeneous");
static_assert(!is_leaf_homogeneous_v<std::vector<std::map<int, float>>>,
              "std::vector<std::map<int, float>> should not be homogeneous");
*/

// Primary template for non-iterator types.
template <typename T, bool IsIterator = common::is_iterator<T>::value>
struct GetTypeFromIteratorOrTrivialType {
    static_assert(std::is_trivial_v<T>, "Non-iterator type must be trivial");
    using type = T;
};

// Specialization for iterator types.
template <typename T>
struct GetTypeFromIteratorOrTrivialType<T, true> {
    using type = typename std::iterator_traits<T>::value_type;
};

template <typename... Args>
constexpr void ignore_parameters(Args&&...) noexcept {}

template <typename T>
static constexpr T infinity() {
    return std::numeric_limits<T>::infinity();
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

    } else if constexpr (std::is_floating_point_v<T> || std::is_floating_point_v<U>) {
        return std::fabs(b - a) <= std::numeric_limits<std::common_type_t<T, U>>::epsilon();

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

template <typename T, typename U>
constexpr auto compute_center_with_left_rounding(const T& lower, const U& upper) -> std::common_type_t<T, U> {
    static_assert(std::is_trivial_v<T>, "T must be trivial.");
    static_assert(std::is_trivial_v<U>, "U must be trivial.");

    const auto lower_plus_upper = lower + upper;

    if constexpr (std::is_integral_v<std::common_type_t<T, U>>) {
        return lower_plus_upper < 0 ? lower_plus_upper / 2 - 1 : lower_plus_upper / 2;

    } else {
        // For floating point, regular calculation suffices
        return lower_plus_upper / 2;
    }
}

template <typename T, typename U>
constexpr auto compute_size_from_center_with_left_rounding(const T& lower, const U& upper) -> std::common_type_t<T, U> {
    static_assert(std::is_trivial_v<T>, "T must be trivial.");
    static_assert(std::is_trivial_v<U>, "U must be trivial.");

    // Adjust size for integer ranges
    if constexpr (std::is_integral_v<std::common_type_t<T, U>>) {
        const auto middle = compute_center_with_left_rounding(lower, upper);
        return 1 + upper - middle;

    } else {
        // For floating point, regular calculation suffices
        return (upper - lower) / 2;
    }
}

/*
// Example usage
int main() {
    using Type = int;
    const Type min = -2;
    const Type max = 0;
    std::cout << "Middle: " << compute_center_with_left_rounding(min, max) << "\n";
    std::cout << "Size: " << compute_size_from_center_with_left_rounding(min, max) << "\n";
    return 0;
}
*/

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