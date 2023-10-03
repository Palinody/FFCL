#pragma once

#include "ffcl/common/Utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace math::heuristics {

template <typename FeatureIterator1, typename FeatureIterator2>
typename FeatureIterator1::value_type squared_euclidean_distance(const FeatureIterator1& feature_first,
                                                                 const FeatureIterator1& feature_last,
                                                                 const FeatureIterator2& other_feature_first) {
    static_assert(std::is_floating_point_v<typename FeatureIterator1::value_type>,
                  "Inputs should be floating point types.");

    static_assert(std::is_floating_point_v<typename FeatureIterator2::value_type>,
                  "Inputs should be floating point types.");

    using FloatType = typename FeatureIterator1::value_type;

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

template <typename FeatureIterator1, typename FeatureIterator2>
typename FeatureIterator1::value_type euclidean_distance(const FeatureIterator1& feature_first,
                                                         const FeatureIterator1& feature_last,
                                                         const FeatureIterator2& other_feature_first) {
    return std::sqrt(squared_euclidean_distance(feature_first, feature_last, other_feature_first));
}

template <typename FeatureIterator1, typename FeatureIterator2>
typename FeatureIterator1::value_type manhattan_distance(const FeatureIterator1& feature_first,
                                                         const FeatureIterator1& feature_last,
                                                         const FeatureIterator2& other_feature_first) {
    using DataType = typename FeatureIterator1::value_type;

    return std::transform_reduce(feature_first,
                                 feature_last,
                                 other_feature_first,
                                 static_cast<DataType>(0),
                                 std::plus<>(),
                                 [](const auto& lhs, const auto& rhs) { return std::abs(lhs - rhs); });
}

template <typename FeatureIterator1, typename FeatureIterator2>
typename FeatureIterator1::value_type unsigned_manhattan_distance(const FeatureIterator1& feature_first,
                                                                  const FeatureIterator1& feature_last,
                                                                  const FeatureIterator2& other_feature_first) {
    static_assert(std::is_unsigned_v<typename FeatureIterator1::value_type>,
                  "Inputs should be unsigned integer point types.");

    static_assert(std::is_unsigned_v<typename FeatureIterator2::value_type>,
                  "Inputs should be unsigned integer point types.");

    using UIntType = typename FeatureIterator1::value_type;

    return std::transform_reduce(feature_first,
                                 feature_last,
                                 other_feature_first,
                                 static_cast<UIntType>(0),
                                 std::plus<>(),
                                 [](const auto& lhs, const auto& rhs) { return lhs > rhs ? lhs - rhs : rhs - lhs; });
}

template <typename FeatureIterator1, typename FeatureIterator2>
typename FeatureIterator1::value_type cosine_similarity(const FeatureIterator1& feature_first,
                                                        const FeatureIterator1& feature_last,
                                                        const FeatureIterator2& other_feature_first) {
    static_assert(std::is_floating_point_v<typename FeatureIterator1::value_type>,
                  "Inputs should be floating point types.");

    static_assert(std::is_floating_point_v<typename FeatureIterator2::value_type>,
                  "Inputs should be floating point types.");

    using FloatType = typename FeatureIterator1::value_type;

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

template <typename FeatureIterator1, typename FeatureIterator2>
auto auto_distance(const FeatureIterator1& feature_first,
                   const FeatureIterator1& feature_last,
                   const FeatureIterator2& other_feature_first) {
    using ValueType1 = typename FeatureIterator1::value_type;
    using ValueType2 = typename FeatureIterator2::value_type;

    static_assert(std::is_same_v<ValueType1, ValueType2>);

    if constexpr (std::is_floating_point_v<ValueType1>) {
        return euclidean_distance(feature_first, feature_last, other_feature_first);

    } else if constexpr (std::is_signed_v<ValueType1>) {
        return manhattan_distance(feature_first, feature_last, other_feature_first);

    } else if constexpr (std::is_unsigned_v<ValueType1>) {
        return unsigned_manhattan_distance(feature_first, feature_last, other_feature_first);

    } else {
#if defined(VERBOSE) && VERBOSE == true
        std::cout << "[WARN] requested type for auto_distance not handled. Using default: euclidean.\n";
#endif
        return euclidean_distance(feature_first, feature_last, other_feature_first);
    }
}

}  // namespace math::heuristics