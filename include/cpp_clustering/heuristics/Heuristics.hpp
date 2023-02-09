#pragma once

#include "cpp_clustering/common/Utils.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace cpp_clustering::heuristic {

template <typename IteratorFloat1, typename IteratorFloat2>
typename IteratorFloat1::value_type euclidean_distance(const IteratorFloat1& first_sample_begin,
                                                       const IteratorFloat1& first_sample_end,
                                                       const IteratorFloat2& second_sample_begin) {
    static_assert(std::is_floating_point_v<typename IteratorFloat1::value_type>,
                  "Inputs should be floating point types.");

    static_assert(std::is_floating_point_v<typename IteratorFloat2::value_type>,
                  "Inputs should be floating point types.");

    using FloatType = typename IteratorFloat1::value_type;

    return std::sqrt(std::transform_reduce(first_sample_begin,
                                           first_sample_end,
                                           second_sample_begin,
                                           static_cast<FloatType>(0),
                                           std::plus<>(),
                                           [](const auto& lhs, const auto& rhs) {
                                               const auto tmp = lhs - rhs;
                                               return tmp * tmp;
                                           }));
}

template <typename IteratorFloat1, typename IteratorFloat2>
typename IteratorFloat1::value_type cached_euclidean_distance(const IteratorFloat1& first_sample_begin,
                                                              const IteratorFloat1& first_sample_end,
                                                              const IteratorFloat2& second_sample_begin) {
    static_assert(std::is_floating_point_v<typename IteratorFloat1::value_type>,
                  "Inputs should be floating point types.");

    static_assert(std::is_floating_point_v<typename IteratorFloat2::value_type>,
                  "Inputs should be floating point types.");

    using FloatType = typename IteratorFloat1::value_type;

    const std::size_t n_features = std::distance(first_sample_begin, first_sample_end);

    auto cache = std::vector<FloatType>(n_features);

    for (std::size_t i = 0; i < n_features; ++i) {
        cache[i] = *(first_sample_begin + i) - *(second_sample_begin + i);
        cache[i] = cache[i] * cache[i];
    }
    return std::sqrt(std::reduce(cache.begin(), cache.end(), static_cast<FloatType>(0), std::plus<>()));
}

template <typename Iterator1, typename Iterator2>
typename Iterator1::value_type manhattan_distance(const Iterator1& first_sample_begin,
                                                  const Iterator1& first_sample_end,
                                                  const Iterator2& second_sample_begin) {
    using DataType = typename Iterator1::value_type;

    return std::transform_reduce(first_sample_begin,
                                 first_sample_end,
                                 second_sample_begin,
                                 static_cast<DataType>(0),
                                 std::plus<>(),
                                 [](const auto& lhs, const auto& rhs) { return std::abs(lhs - rhs); });
}

template <typename IteratorUInt1, typename IteratorUInt2>
typename IteratorUInt1::value_type unsigned_manhattan_distance(const IteratorUInt1& first_sample_begin,
                                                               const IteratorUInt1& first_sample_end,
                                                               const IteratorUInt2& second_sample_begin) {
    static_assert(std::is_unsigned_v<typename IteratorUInt1::value_type>,
                  "Inputs should be unsigned integer point types.");

    static_assert(std::is_unsigned_v<typename IteratorUInt2::value_type>,
                  "Inputs should be unsigned integer point types.");

    using IntType = typename IteratorUInt1::value_type;

    return std::transform_reduce(first_sample_begin,
                                 first_sample_end,
                                 second_sample_begin,
                                 static_cast<IntType>(0),
                                 std::plus<>(),
                                 [](const auto& lhs, const auto& rhs) { return lhs > rhs ? lhs - rhs : rhs - lhs; });
}

template <typename IteratorFloat1, typename IteratorFloat2>
typename IteratorFloat1::value_type cosine_similarity(const IteratorFloat1& first_sample_begin,
                                                      const IteratorFloat1& first_sample_end,
                                                      IteratorFloat2        second_sample_begin) {
    static_assert(std::is_floating_point_v<typename IteratorFloat1::value_type>,
                  "Inputs should be floating point types.");

    static_assert(std::is_floating_point_v<typename IteratorFloat2::value_type>,
                  "Inputs should be floating point types.");

    using FloatType = typename IteratorFloat1::value_type;

    const std::size_t n_features = std::distance(first_sample_begin, first_sample_end);

    FloatType dot_product = std::inner_product(first_sample_begin, first_sample_end, second_sample_begin, 0.0);

    FloatType magnitude_1 = std::inner_product(first_sample_begin, first_sample_end, first_sample_begin, 0.0);

    FloatType magnitude_2 =
        std::inner_product(second_sample_begin, second_sample_begin + n_features, second_sample_begin, 0.0);

    if (!magnitude_1 || !magnitude_2) {
        return 0;
    }
    return dot_product / std::sqrt(magnitude_1 * magnitude_2);
}

template <typename Iterator1, typename Iterator2>
auto heuristic(const Iterator1& first_sample_begin,
               const Iterator1& first_sample_end,
               const Iterator2& second_sample_begin) {
    using ValueType = typename Iterator1::value_type;

    if constexpr (std::is_floating_point_v<ValueType>) {
        return euclidean_distance(first_sample_begin, first_sample_end, second_sample_begin);

    } else if constexpr (std::is_signed_v<ValueType>) {
        return manhattan_distance(first_sample_begin, first_sample_end, second_sample_begin);

    } else if constexpr (std::is_unsigned_v<ValueType>) {
        return unsigned_manhattan_distance(first_sample_begin, first_sample_end, second_sample_begin);
    }
    std::cout << "[WARN] requested type for heuristic not handled. Using default: euclidean.\n";
    return euclidean_distance(first_sample_begin, first_sample_end, second_sample_begin);
}

}  // namespace cpp_clustering::heuristic