#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/math/linear_algebra/Transpose.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace math::statistics {

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
    assert(0 < n && n <= static_cast<std::ptrdiff_t>(std::distance(data_first, data_last)) &&
           "N-th smallest requested element shouldn't be greater than the container's size.");

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

template <typename Iterator>
auto compute_mean_per_feature(Iterator data_first, Iterator data_last, std::size_t n_features) {
    using DataType = typename Iterator::value_type;

    const auto n_samples = common::utils::get_n_samples(data_first, data_last, n_features);

    assert(n_samples > 0);

    auto mean_per_feature = std::vector<DataType>(n_features);

    for (; data_first != data_last; std::advance(data_first, n_features)) {
        std::transform(
            data_first, data_first + n_features, mean_per_feature.begin(), mean_per_feature.begin(), std::plus<>());
    }
    std::transform(
        mean_per_feature.begin(), mean_per_feature.end(), mean_per_feature.begin(), [n_samples](const auto& feature) {
            return feature / n_samples;
        });
    return mean_per_feature;
}

template <typename IndicesIterator, typename SamplesIterator>
auto compute_mean_per_feature(IndicesIterator indices_first,
                              IndicesIterator indices_last,
                              SamplesIterator samples_first,
                              SamplesIterator samples_last,
                              std::size_t     n_features) {
    common::utils::ignore_parameters(samples_last);

    using DataType = typename SamplesIterator::value_type;

    const std::size_t n_samples = std::distance(indices_first, indices_last);

    assert(n_samples > 0);

    auto mean_per_feature = std::vector<DataType>(n_features);

    for (; indices_first != indices_last; ++indices_first) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            mean_per_feature[feature_index] += samples_first[*indices_first * n_features + feature_index];
        }
    }
    std::transform(
        mean_per_feature.begin(), mean_per_feature.end(), mean_per_feature.begin(), [n_samples](const auto& feature) {
            return feature / n_samples;
        });
    return mean_per_feature;
}

template <typename IndicesIterator, typename SamplesIterator>
auto compute_mean_per_feature(IndicesIterator                 indices_first,
                              IndicesIterator                 indices_last,
                              SamplesIterator                 samples_first,
                              SamplesIterator                 samples_last,
                              std::size_t                     n_features,
                              const std::vector<std::size_t>& feature_mask) {
    common::utils::ignore_parameters(samples_last);

    using DataType = typename SamplesIterator::value_type;

    const std::size_t n_samples = std::distance(indices_first, indices_last);

    assert(n_samples > 0);

    auto mean_per_feature = std::vector<DataType>(feature_mask.size());

    for (; indices_first != indices_last; ++indices_first) {
        for (std::size_t mask_index = 0; mask_index < feature_mask.size(); ++mask_index) {
            mean_per_feature[mask_index] += samples_first[*indices_first * n_features + feature_mask[mask_index]];
        }
    }
    std::transform(
        mean_per_feature.begin(), mean_per_feature.end(), mean_per_feature.begin(), [n_samples](const auto& feature) {
            return feature / n_samples;
        });
    return mean_per_feature;
}

template <typename Iterator>
auto compute_variance_per_feature(Iterator data_first, Iterator data_last, std::size_t n_features) {
    using DataType = typename Iterator::value_type;

    const auto n_samples = common::utils::get_n_samples(data_first, data_last, n_features);

    assert(n_samples > 1);

    const auto mean_per_feature = compute_mean_per_feature(data_first, data_last, n_features);

    auto variance_per_feature = std::vector<DataType>(n_features);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        auto features_accumulator = std::vector<DataType>(n_features);

        std::transform(data_first + sample_index * n_features,
                       data_first + sample_index * n_features + n_features,
                       mean_per_feature.begin(),
                       features_accumulator.begin(),
                       [](const auto& x, const auto& mean) {
                           const auto temp = x - mean;
                           return temp * temp;
                       });

        std::transform(features_accumulator.begin(),
                       features_accumulator.end(),
                       variance_per_feature.begin(),
                       variance_per_feature.begin(),
                       std::plus<DataType>());
    }
    std::transform(variance_per_feature.begin(),
                   variance_per_feature.end(),
                   variance_per_feature.begin(),
                   [n_samples](const auto& feature) { return feature / (n_samples - 1); });

    return variance_per_feature;
}

template <typename Iterator>
std::size_t argmax_variance_per_feature(Iterator data_first, Iterator data_last, std::size_t n_features) {
    using DataType = typename Iterator::value_type;

    const auto n_samples = common::utils::get_n_samples(data_first, data_last, n_features);

    assert(n_samples > 1);

    const auto mean_per_feature = compute_mean_per_feature(data_first, data_last, n_features);

    auto variance_per_feature = std::vector<DataType>(n_features);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        auto features_accumulator = std::vector<DataType>(n_features);

        std::transform(data_first + sample_index * n_features,
                       data_first + sample_index * n_features + n_features,
                       mean_per_feature.begin(),
                       features_accumulator.begin(),
                       [](const auto& x, const auto& mean) {
                           const auto temp = x - mean;
                           return temp * temp;
                       });

        std::transform(features_accumulator.begin(),
                       features_accumulator.end(),
                       variance_per_feature.begin(),
                       variance_per_feature.begin(),
                       std::plus<DataType>());
    }
    return math::statistics::argmax(variance_per_feature.begin(), variance_per_feature.end());
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t argmax_variance_per_feature(const IndicesIterator& indices_first,
                                        const IndicesIterator& indices_last,
                                        SamplesIterator        data_first,
                                        SamplesIterator        data_last,
                                        std::size_t            n_features) {
    using DataType = typename SamplesIterator::value_type;

    const std::size_t n_samples = std::distance(indices_first, indices_last);

    assert(n_samples > 1);

    const auto mean_per_feature =
        compute_mean_per_feature(indices_first, indices_last, data_first, data_last, n_features);

    auto variance_per_feature = std::vector<DataType>(n_features);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        auto features_accumulator = std::vector<DataType>(n_features);

        std::transform(data_first + indices_first[sample_index] * n_features,
                       data_first + indices_first[sample_index] * n_features + n_features,
                       mean_per_feature.begin(),
                       features_accumulator.begin(),
                       [](const auto& x, const auto& mean) {
                           const auto temp = x - mean;
                           return temp * temp;
                       });

        std::transform(features_accumulator.begin(),
                       features_accumulator.end(),
                       variance_per_feature.begin(),
                       variance_per_feature.begin(),
                       std::plus<DataType>());
    }
    return math::statistics::argmax(variance_per_feature.begin(), variance_per_feature.end());
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t argmax_variance_per_feature(const IndicesIterator&          indices_first,
                                        const IndicesIterator&          indices_last,
                                        SamplesIterator                 samples_first,
                                        SamplesIterator                 samples_last,
                                        std::size_t                     n_features,
                                        const std::vector<std::size_t>& feature_mask) {
    using DataType = typename SamplesIterator::value_type;

    const std::size_t n_samples = std::distance(indices_first, indices_last);

    assert(n_samples > 1);

    const auto mean_per_feature =
        compute_mean_per_feature(indices_first, indices_last, samples_first, samples_last, n_features, feature_mask);

    auto variance_per_feature = std::vector<DataType>(feature_mask.size());

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        auto features_accumulator = std::vector<DataType>(feature_mask.size());

        for (std::size_t mask_index = 0; mask_index < feature_mask.size(); ++mask_index) {
            const auto temp =
                samples_first[*indices_first * n_features + feature_mask[mask_index]] - mean_per_feature[mask_index];
            variance_per_feature[mask_index] += temp * temp;
        }
    }
    // return the feature index
    return feature_mask[math::statistics::argmax(variance_per_feature.begin(), variance_per_feature.end())];
}

template <typename Iterator>
typename Iterator::value_type compute_variance(Iterator data_first, Iterator data_last) {
    using DataType = typename Iterator::value_type;

    const auto n_elements     = std::distance(data_first, data_last);
    const auto sum            = std::accumulate(data_first, data_last, static_cast<DataType>(0));
    const auto sum_of_squares = std::inner_product(data_first, data_last, data_first, static_cast<DataType>(0));

    return (sum_of_squares - sum * sum / n_elements) / (n_elements - 1);
}

}  // namespace math::statistics