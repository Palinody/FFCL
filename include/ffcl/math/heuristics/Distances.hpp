#pragma once

#include "ffcl/common/Utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace math::heuristics {

template <typename IteratorFloat1, typename IteratorFloat2>
typename IteratorFloat1::value_type squared_euclidean_distance(const IteratorFloat1& feature_first,
                                                               const IteratorFloat1& feature_last,
                                                               const IteratorFloat2& other_feature_first) {
    static_assert(std::is_floating_point_v<typename IteratorFloat1::value_type>,
                  "Inputs should be floating point types.");

    static_assert(std::is_floating_point_v<typename IteratorFloat2::value_type>,
                  "Inputs should be floating point types.");

    using FloatType = typename IteratorFloat1::value_type;

    return std::transform_reduce(feature_first,
                                 feature_last,
                                 other_feature_first,
                                 static_cast<FloatType>(0),
                                 std::plus<>(),
                                 [](const auto& lhs, const auto& rhs) {
                                     const auto tmp = lhs - rhs;
                                     return tmp * tmp;
                                 });
}

template <typename IteratorFloat1, typename IteratorFloat2>
typename IteratorFloat1::value_type euclidean_distance(const IteratorFloat1& feature_first,
                                                       const IteratorFloat1& feature_last,
                                                       const IteratorFloat2& other_feature_first) {
    return std::sqrt(squared_euclidean_distance(feature_first, feature_last, other_feature_first));
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

    using UIntType = typename IteratorUInt1::value_type;

    return std::transform_reduce(feature_first,
                                 feature_last,
                                 other_feature_first,
                                 static_cast<UIntType>(0),
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

    const FloatType dot_product =
        std::inner_product(feature_first, feature_last, other_feature_first, static_cast<FloatType>(0));

    const FloatType magnitude_1 =
        std::inner_product(feature_first, feature_last, feature_first, static_cast<FloatType>(0));

    const FloatType magnitude_2 = std::inner_product(
        other_feature_first, other_feature_first + n_features, other_feature_first, static_cast<FloatType>(0));

    if (!magnitude_1 || !magnitude_2) {
        return 0;
    }
    return dot_product / std::sqrt(magnitude_1 * magnitude_2);
}

template <typename Iterator1, typename Iterator2>
auto auto_distance(const Iterator1& feature_first,
                   const Iterator1& feature_last,
                   const Iterator2& other_feature_first) {
    using ValueType = typename Iterator1::value_type;

    if constexpr (std::is_floating_point_v<ValueType>) {
        return euclidean_distance(feature_first, feature_last, other_feature_first);

    } else if constexpr (std::is_signed_v<ValueType>) {
        return manhattan_distance(feature_first, feature_last, other_feature_first);

    } else if constexpr (std::is_unsigned_v<ValueType>) {
        return unsigned_manhattan_distance(feature_first, feature_last, other_feature_first);
    }
#if defined(VERBOSE) && VERBOSE == true
    std::cout << "[WARN] requested type for auto_distance not handled. Using default: euclidean.\n";
#endif
    return euclidean_distance(feature_first, feature_last, other_feature_first);
}

}  // namespace math::heuristics