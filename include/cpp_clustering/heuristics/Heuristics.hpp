#pragma once

#include "cpp_clustering/common/Utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace cpp_clustering::heuristic {

template <typename IteratorFloat1, typename IteratorFloat2>
typename IteratorFloat1::value_type euclidean_distance(const IteratorFloat1& feature_first,
                                                       const IteratorFloat1& feature_last,
                                                       const IteratorFloat2& other_feature_first) {
    static_assert(std::is_floating_point_v<typename IteratorFloat1::value_type>,
                  "Inputs should be floating point types.");

    static_assert(std::is_floating_point_v<typename IteratorFloat2::value_type>,
                  "Inputs should be floating point types.");

    using FloatType = typename IteratorFloat1::value_type;

    return std::sqrt(std::transform_reduce(feature_first,
                                           feature_last,
                                           other_feature_first,
                                           static_cast<FloatType>(0),
                                           std::plus<>(),
                                           [](const auto& lhs, const auto& rhs) {
                                               const auto tmp = lhs - rhs;
                                               return tmp * tmp;
                                           }));
}

template <typename IteratorFloat1, typename IteratorFloat2>
typename IteratorFloat1::value_type cached_euclidean_distance(const IteratorFloat1& feature_first,
                                                              const IteratorFloat1& feature_last,
                                                              const IteratorFloat2& other_feature_first) {
    static_assert(std::is_floating_point_v<typename IteratorFloat1::value_type>,
                  "Inputs should be floating point types.");

    static_assert(std::is_floating_point_v<typename IteratorFloat2::value_type>,
                  "Inputs should be floating point types.");

    using FloatType = typename IteratorFloat1::value_type;

    const std::size_t n_features = std::distance(feature_first, feature_last);

    auto cache = std::vector<FloatType>(n_features);

    for (std::size_t i = 0; i < n_features; ++i) {
        cache[i] = *(feature_first + i) - *(other_feature_first + i);
        cache[i] = cache[i] * cache[i];
    }
    return std::sqrt(std::reduce(cache.begin(), cache.end(), static_cast<FloatType>(0), std::plus<>()));
}

template <typename Iterator1, typename Iterator2>
typename Iterator1::value_type manhattan_distance(const Iterator1& feature_first,
                                                  const Iterator1& feature_last,
                                                  const Iterator2& other_feature_first) {
    using DataType = typename Iterator1::value_type;

    return std::transform_reduce(feature_first,
                                 feature_last,
                                 other_feature_first,
                                 static_cast<DataType>(0),
                                 std::plus<>(),
                                 [](const auto& lhs, const auto& rhs) { return std::abs(lhs - rhs); });
}

template <typename IteratorUInt1, typename IteratorUInt2>
typename IteratorUInt1::value_type unsigned_manhattan_distance(const IteratorUInt1& feature_first,
                                                               const IteratorUInt1& feature_last,
                                                               const IteratorUInt2& other_feature_first) {
    static_assert(std::is_unsigned_v<typename IteratorUInt1::value_type>,
                  "Inputs should be unsigned integer point types.");

    static_assert(std::is_unsigned_v<typename IteratorUInt2::value_type>,
                  "Inputs should be unsigned integer point types.");

    using IntType = typename IteratorUInt1::value_type;

    return std::transform_reduce(feature_first,
                                 feature_last,
                                 other_feature_first,
                                 static_cast<IntType>(0),
                                 std::plus<>(),
                                 [](const auto& lhs, const auto& rhs) { return lhs > rhs ? lhs - rhs : rhs - lhs; });
}

template <typename IteratorFloat1, typename IteratorFloat2>
typename IteratorFloat1::value_type cosine_similarity(const IteratorFloat1& feature_first,
                                                      const IteratorFloat1& feature_last,
                                                      const IteratorFloat2& other_feature_first) {
    static_assert(std::is_floating_point_v<typename IteratorFloat1::value_type>,
                  "Inputs should be floating point types.");

    static_assert(std::is_floating_point_v<typename IteratorFloat2::value_type>,
                  "Inputs should be floating point types.");

    using FloatType = typename IteratorFloat1::value_type;

    const std::size_t n_features = std::distance(feature_first, feature_last);

    FloatType dot_product =
        std::inner_product(feature_first, feature_last, other_feature_first, static_cast<FloatType>(0));

    FloatType magnitude_1 = std::inner_product(feature_first, feature_last, feature_first, static_cast<FloatType>(0));

    FloatType magnitude_2 = std::inner_product(
        other_feature_first, other_feature_first + n_features, other_feature_first, static_cast<FloatType>(0));

    if (!magnitude_1 || !magnitude_2) {
        return 0;
    }
    return dot_product / std::sqrt(magnitude_1 * magnitude_2);
}

template <typename Iterator1, typename Iterator2>
auto heuristic(const Iterator1& feature_first, const Iterator1& feature_last, const Iterator2& other_feature_first) {
    using ValueType = typename Iterator1::value_type;

    if constexpr (std::is_floating_point_v<ValueType>) {
        return euclidean_distance(feature_first, feature_last, other_feature_first);

    } else if constexpr (std::is_signed_v<ValueType>) {
        return manhattan_distance(feature_first, feature_last, other_feature_first);

    } else if constexpr (std::is_unsigned_v<ValueType>) {
        return unsigned_manhattan_distance(feature_first, feature_last, other_feature_first);
    }
#if defined(VERBOSE) && VERBOSE == true
    std::cout << "[WARN] requested type for heuristic not handled. Using default: euclidean.\n";
#endif
    return euclidean_distance(feature_first, feature_last, other_feature_first);
}

}  // namespace cpp_clustering::heuristic