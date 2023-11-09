#pragma once

#include "ffcl/common/Utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ffcl::common::math::heuristics {

template <typename LeftFeaturesIterator, typename RightFeaturesIterator>
auto squared_euclidean_distance(const LeftFeaturesIterator&  left_features_range_first,
                                const LeftFeaturesIterator&  left_features_range_last,
                                const RightFeaturesIterator& right_features_range_first)
    -> decltype(std::declval<typename LeftFeaturesIterator::value_type>() *
                std::declval<typename LeftFeaturesIterator::value_type>()) {
    static_assert(std::is_floating_point_v<typename LeftFeaturesIterator::value_type>, "Input1 must be float.");
    static_assert(std::is_floating_point_v<typename RightFeaturesIterator::value_type>, "Input2 must be float.");

    using ResultType = decltype(std::declval<typename LeftFeaturesIterator::value_type>() *
                                std::declval<typename LeftFeaturesIterator::value_type>());

    return std::transform_reduce(left_features_range_first,
                                 left_features_range_last,
                                 right_features_range_first,
                                 static_cast<ResultType>(0),
                                 std::plus<>(),
                                 [](const auto& lhs, const auto& rhs) {
                                     const auto tmp = lhs - rhs;
                                     return tmp * tmp;
                                 });
}

template <typename LeftFeaturesIterator, typename RightFeaturesIterator>
auto euclidean_distance(const LeftFeaturesIterator&  left_features_range_first,
                        const LeftFeaturesIterator&  left_features_range_last,
                        const RightFeaturesIterator& right_features_range_first) {
    return std::sqrt(
        squared_euclidean_distance(left_features_range_first, left_features_range_last, right_features_range_first));
}

template <typename LeftFeaturesIterator, typename RightFeaturesIterator>
auto manhattan_distance(const LeftFeaturesIterator&  left_features_range_first,
                        const LeftFeaturesIterator&  left_features_range_last,
                        const RightFeaturesIterator& right_features_range_first)
    -> decltype(std::declval<typename LeftFeaturesIterator::value_type>() +
                std::declval<typename LeftFeaturesIterator::value_type>()) {
    static_assert(std::is_signed<typename LeftFeaturesIterator::value_type>::value, "Input1 must be signed.");
    static_assert(std::is_signed<typename RightFeaturesIterator::value_type>::value, "Input2 must be signed.");

    using ResultType = decltype(std::declval<typename LeftFeaturesIterator::value_type>() +
                                std::declval<typename LeftFeaturesIterator::value_type>());

    return std::transform_reduce(left_features_range_first,
                                 left_features_range_last,
                                 right_features_range_first,
                                 static_cast<ResultType>(0),
                                 std::plus<>(),
                                 [](const auto& lhs, const auto& rhs) { return std::abs(lhs - rhs); });
}

template <typename LeftFeaturesIterator, typename RightFeaturesIterator>
auto unsigned_manhattan_distance(const LeftFeaturesIterator&  left_features_range_first,
                                 const LeftFeaturesIterator&  left_features_range_last,
                                 const RightFeaturesIterator& right_features_range_first)
    -> decltype(std::declval<typename LeftFeaturesIterator::value_type>() -
                std::declval<typename RightFeaturesIterator::value_type>()) {
    static_assert(std::is_unsigned_v<typename LeftFeaturesIterator::value_type>, "Input1 must be unsigned integer.");
    static_assert(std::is_unsigned_v<typename RightFeaturesIterator::value_type>, "Input2 must be unsigned integer.");

    using ResultType = decltype(std::declval<typename LeftFeaturesIterator::value_type>() -
                                std::declval<typename RightFeaturesIterator::value_type>());

    return std::transform_reduce(left_features_range_first,
                                 left_features_range_last,
                                 right_features_range_first,
                                 static_cast<ResultType>(0),
                                 std::plus<>(),
                                 [](const auto& lhs, const auto& rhs) { return lhs > rhs ? lhs - rhs : rhs - lhs; });
}

template <typename LeftFeaturesIterator, typename RightFeaturesIterator>
auto cosine_similarity(const LeftFeaturesIterator&  left_features_range_first,
                       const LeftFeaturesIterator&  left_features_range_last,
                       const RightFeaturesIterator& right_features_range_first)
    -> decltype(std::declval<typename LeftFeaturesIterator::value_type>() *
                std::declval<typename RightFeaturesIterator::value_type>()) {
    static_assert(std::is_floating_point_v<typename LeftFeaturesIterator::value_type>, "Input1 must be float.");
    static_assert(std::is_floating_point_v<typename RightFeaturesIterator::value_type>, "Input2 must be float.");

    using ResultType = decltype(std::declval<typename LeftFeaturesIterator::value_type>() *
                                std::declval<typename RightFeaturesIterator::value_type>());

    const std::size_t n_features = std::distance(left_features_range_first, left_features_range_last);

    const ResultType dot_product = std::inner_product(
        left_features_range_first, left_features_range_last, right_features_range_first, static_cast<ResultType>(0));

    const ResultType magnitude_1 = sqrt(std::inner_product(
        left_features_range_first, left_features_range_last, left_features_range_first, static_cast<ResultType>(0)));

    const ResultType magnitude_2 = sqrt(std::inner_product(right_features_range_first,
                                                           right_features_range_first + n_features,
                                                           right_features_range_first,
                                                           static_cast<ResultType>(0)));
    // returns "infinity" if the denominator is zero
    return division(dot_product, magnitude_1 * magnitude_2, infinity<ResultType>());
}

// Iterative with two matrix rows https://en.wikipedia.org/wiki/Levenshtein_distance
template <typename LeftFeaturesIterator, typename RightFeaturesIterator>
std::size_t levenshtein_distance(const LeftFeaturesIterator&  left_features_range_first,
                                 const LeftFeaturesIterator&  left_features_range_last,
                                 const RightFeaturesIterator& right_features_range_first,
                                 const RightFeaturesIterator& right_features_range_last) {
    static_assert(std::is_integral_v<typename LeftFeaturesIterator::value_type>, "Input1 must be integral.");
    static_assert(std::is_integral_v<typename RightFeaturesIterator::value_type>, "Input2 must be integral.");

    const std::size_t n_features_left  = std::distance(left_features_range_first, left_features_range_last);
    const std::size_t n_features_right = std::distance(right_features_range_first, right_features_range_last);

    std::vector<std::size_t> previous_buffer(n_features_right + 1);
    std::vector<std::size_t> current_buffer(n_features_right + 1);

    std::iota(previous_buffer.begin(), previous_buffer.end(), static_cast<std::size_t>(0));

    for (std::size_t feature_index_left = 0; feature_index_left < n_features_left; ++feature_index_left) {
        current_buffer[0] = feature_index_left + 1;

        for (std::size_t feature_index_right = 0; feature_index_right < n_features_right; ++feature_index_right) {
            const std::size_t deletion_cost = previous_buffer[feature_index_right + 1] + 1;

            const std::size_t insertion_cost = current_buffer[feature_index_right] + 1;

            const std::size_t substitution_cost =
                left_features_range_first[feature_index_left] == right_features_range_first[feature_index_right]
                    ? previous_buffer[feature_index_right]
                    : previous_buffer[feature_index_right] + 1;

            current_buffer[feature_index_right + 1] = std::min({deletion_cost, insertion_cost, substitution_cost});
        }
        std::swap(previous_buffer, current_buffer);
    }
    return previous_buffer[n_features_right];
}

template <typename LeftFeaturesIterator, typename RightFeaturesIterator>
auto auto_distance(const LeftFeaturesIterator&  left_features_range_first,
                   const LeftFeaturesIterator&  left_features_range_last,
                   const RightFeaturesIterator& right_features_range_first) {
    using FeatureType1 = typename LeftFeaturesIterator::value_type;
    using FeatureType2 = typename RightFeaturesIterator::value_type;

    // The feature types must be the same for the auto_distance function since its only be used for practicality
    // purposes for now. Each function is compatible with different types.
    static_assert(std::is_same_v<FeatureType1, FeatureType2>);

    if constexpr (std::is_floating_point_v<FeatureType1>) {
        return euclidean_distance(left_features_range_first, left_features_range_last, right_features_range_first);

    } else if constexpr (std::is_signed_v<FeatureType1>) {
        return manhattan_distance(left_features_range_first, left_features_range_last, right_features_range_first);

    } else if constexpr (std::is_unsigned_v<FeatureType1>) {
        return unsigned_manhattan_distance(
            left_features_range_first, left_features_range_last, right_features_range_first);

    } else {
#if defined(VERBOSE) && VERBOSE == true
        std::cout << "[WARN] requested type for auto_distance not handled. Using default: euclidean.\n";
#endif
        return euclidean_distance(left_features_range_first, left_features_range_last, right_features_range_first);
    }
}

}  // namespace ffcl::common::math::heuristics