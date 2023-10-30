#pragma once

#include "ffcl/common/Utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace math::statistics {

template <typename DataType = double, typename SamplesIterator>
std::vector<DataType> normalize_min_max(const SamplesIterator& samples_range_first,
                                        const SamplesIterator& samples_range_last) {
    const auto [min, max] = std::minmax_element(samples_range_first, samples_range_last);

    if (*min == *max) {
        const std::size_t n_elements = std::distance(samples_range_first, samples_range_last);
        // if all the values are the same distribute the weights equaly
        return std::vector<DataType>(n_elements, 1.0 / n_elements);
    }
    auto res = std::vector<DataType>(samples_range_last - samples_range_first);
    // closest objects get a higher score. Distance zero -> 1
    std::transform(samples_range_first, samples_range_last, res.begin(), [&min, &max](const auto& dist) {
        return static_cast<DataType>(1) - (dist - *min) / (*max - *min);
    });
    return res;
}

template <typename DataType = double, typename SamplesIterator>
std::vector<DataType> normalize(const SamplesIterator& samples_range_first, const SamplesIterator& samples_range_last) {
    auto normalized_vector = std::vector<DataType>(std::distance(samples_range_first, samples_range_last));

    // Calculate the sum of all elements in the vector
    const DataType sum = std::accumulate(samples_range_first, samples_range_last, static_cast<DataType>(0));
    // Divide each element by the sum to normalize the vector
    std::transform(samples_range_first, samples_range_last, normalized_vector.begin(), [sum](const DataType& val) {
        return val / sum;
    });

    return normalized_vector;
}

template <typename Container>
void min_container_inplace_left(Container& left_container, const Container& right_container) {
    // compute element-wise minimum and place the result in the left container
    std::transform(left_container.begin(),
                   left_container.end(),
                   right_container.begin(),
                   left_container.begin(),
                   [](const auto& x, const auto& y) { return x < y ? x : y; });
}

template <typename SamplesIterator>
typename SamplesIterator::value_type argmin(const SamplesIterator& samples_range_first,
                                            const SamplesIterator& samples_range_last) {
    auto min_it = std::min_element(samples_range_first, samples_range_last);
    return std::distance(samples_range_first, min_it);
}

template <typename SamplesIterator>
typename SamplesIterator::value_type argmax(const SamplesIterator& samples_range_first,
                                            const SamplesIterator& samples_range_last) {
    auto max_it = std::max_element(samples_range_first, samples_range_last);
    return std::distance(samples_range_first, max_it);
}

template <typename SamplesIterator>
std::pair<std::size_t, typename SamplesIterator::value_type> get_min_index_value_pair(
    const SamplesIterator& samples_range_first,
    const SamplesIterator& samples_range_last) {
    using DataType = typename SamplesIterator::value_type;

    const auto     min_element = std::min_element(samples_range_first, samples_range_last);
    std::size_t    min_index   = std::distance(samples_range_first, min_element);
    const DataType min_value   = *min_element;

    return {min_index, min_value};
}

template <typename SamplesIterator>
std::pair<std::size_t, typename SamplesIterator::value_type> get_max_index_value_pair(
    const SamplesIterator& samples_range_first,
    const SamplesIterator& samples_range_last) {
    using DataType = typename SamplesIterator::value_type;

    const auto     max_element = std::max_element(samples_range_first, samples_range_last);
    std::size_t    max_index   = std::distance(samples_range_first, max_element);
    const DataType max_value   = *max_element;

    return {max_index, max_value};
}

template <typename SamplesIterator>
std::pair<std::size_t, typename SamplesIterator::value_type> get_second_max_index_value_pair(
    const SamplesIterator& samples_range_first,
    const SamplesIterator& samples_range_last) {
    using DataType = typename SamplesIterator::value_type;

    // Get the index of the maximum element
    auto           max_it    = std::max_element(samples_range_first, samples_range_last);
    std::size_t    max_idx   = std::distance(samples_range_first, max_it);
    const DataType max_value = *max_it;

    if (std::distance(samples_range_first, samples_range_last) < 2) {
        return {max_idx, max_value};
    }
    // Replace the maximum element with the lowest possible value
    *max_it = std::numeric_limits<DataType>::lowest();

    // Get the index of the second maximum element
    const auto     second_max_it    = std::max_element(samples_range_first, samples_range_last);
    std::size_t    second_max_idx   = std::distance(samples_range_first, second_max_it);
    const DataType second_max_value = *second_max_it;

    // Restore the original maximum value
    *max_it = max_value;

    return {second_max_idx, second_max_value};
}

template <typename SamplesIterator>
std::pair<std::size_t, typename SamplesIterator::value_type> find_nth_smallest_index_and_element(
    const SamplesIterator& samples_range_first,
    const SamplesIterator& samples_range_last,
    std::size_t            n) {
    using DataType = typename SamplesIterator::value_type;

    if (n > static_cast<std::size_t>(samples_range_last - samples_range_first)) {
        throw std::invalid_argument("N-th smallest requested element shouldn't be greater that the container's size.");
    }
    assert(0 < n && n <= static_cast<std::ptrdiff_t>(std::distance(samples_range_first, samples_range_last)) &&
           "N-th smallest requested element shouldn't be greater than the container's size.");

    std::vector<DataType> data_sorted(samples_range_last - samples_range_first);
    std::copy(samples_range_first, samples_range_last, data_sorted.begin());
    std::sort(data_sorted.begin(), data_sorted.end());

    // get the n-th smallest element
    auto nth_smallest = data_sorted.at(n - 1);

    // find the index of the n-th smallest element in the original container
    auto it    = std::find(samples_range_first, samples_range_last, nth_smallest);
    auto index = std::distance(samples_range_first, it);

    return {index, nth_smallest};
}

template <typename SamplesIterator>
auto compute_mean_per_feature(SamplesIterator samples_range_first,
                              SamplesIterator samples_range_last,
                              std::size_t     n_features) {
    using DataType = typename SamplesIterator::value_type;

    const auto n_samples = common::utils::get_n_samples(samples_range_first, samples_range_last, n_features);

    assert(n_samples > 0);

    auto mean_per_feature = std::vector<DataType>(n_features);

    for (; samples_range_first != samples_range_last; std::advance(samples_range_first, n_features)) {
        std::transform(samples_range_first,
                       samples_range_first + n_features,
                       mean_per_feature.begin(),
                       mean_per_feature.begin(),
                       std::plus<>());
    }
    std::transform(
        mean_per_feature.begin(), mean_per_feature.end(), mean_per_feature.begin(), [n_samples](const auto& feature) {
            return feature / n_samples;
        });
    return mean_per_feature;
}

template <typename IndicesIterator, typename SamplesIterator>
auto compute_mean_per_feature(IndicesIterator indices_range_first,
                              IndicesIterator indices_range_last,
                              SamplesIterator samples_range_first,
                              SamplesIterator samples_range_last,
                              std::size_t     n_features) {
    common::utils::ignore_parameters(samples_range_last);

    using DataType = typename SamplesIterator::value_type;

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    assert(n_samples > 0);

    auto mean_per_feature = std::vector<DataType>(n_features);

    for (; indices_range_first != indices_range_last; ++indices_range_first) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            mean_per_feature[feature_index] += samples_range_first[*indices_range_first * n_features + feature_index];
        }
    }
    std::transform(
        mean_per_feature.begin(), mean_per_feature.end(), mean_per_feature.begin(), [n_samples](const auto& feature) {
            return feature / n_samples;
        });
    return mean_per_feature;
}

template <typename IndicesIterator, typename SamplesIterator>
auto compute_mean_per_feature(IndicesIterator                 indices_range_first,
                              IndicesIterator                 indices_range_last,
                              SamplesIterator                 samples_range_first,
                              SamplesIterator                 samples_range_last,
                              std::size_t                     n_features,
                              const std::vector<std::size_t>& feature_mask) {
    common::utils::ignore_parameters(samples_range_last);

    using DataType = typename SamplesIterator::value_type;

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    assert(n_samples > 0);

    auto mean_per_feature = std::vector<DataType>(feature_mask.size());

    for (; indices_range_first != indices_range_last; ++indices_range_first) {
        for (std::size_t mask_index = 0; mask_index < feature_mask.size(); ++mask_index) {
            mean_per_feature[mask_index] +=
                samples_range_first[*indices_range_first * n_features + feature_mask[mask_index]];
        }
    }
    std::transform(
        mean_per_feature.begin(), mean_per_feature.end(), mean_per_feature.begin(), [n_samples](const auto& feature) {
            return feature / n_samples;
        });
    return mean_per_feature;
}

template <typename SamplesIterator>
auto compute_variance_per_feature(SamplesIterator samples_range_first,
                                  SamplesIterator samples_range_last,
                                  std::size_t     n_features) {
    using DataType = typename SamplesIterator::value_type;

    const auto n_samples = common::utils::get_n_samples(samples_range_first, samples_range_last, n_features);

    assert(n_samples > 1);

    const auto mean_per_feature = compute_mean_per_feature(samples_range_first, samples_range_last, n_features);

    auto variance_per_feature = std::vector<DataType>(n_features);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        auto features_accumulator = std::vector<DataType>(n_features);

        std::transform(samples_range_first + sample_index * n_features,
                       samples_range_first + sample_index * n_features + n_features,
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

template <typename IndicesIterator, typename SamplesIterator>
auto compute_variance_per_feature(const IndicesIterator& indices_range_first,
                                  const IndicesIterator& indices_range_last,
                                  SamplesIterator        samples_range_first,
                                  SamplesIterator        samples_range_last,
                                  std::size_t            n_features) {
    using DataType = typename SamplesIterator::value_type;

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    assert(n_samples > 1);

    const auto mean_per_feature = compute_mean_per_feature(
        indices_range_first, indices_range_last, samples_range_first, samples_range_last, n_features);

    auto variance_per_feature = std::vector<DataType>(n_features);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        auto features_accumulator = std::vector<DataType>(n_features);

        std::transform(samples_range_first + indices_range_first[sample_index] * n_features,
                       samples_range_first + indices_range_first[sample_index] * n_features + n_features,
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

template <typename IndicesIterator, typename SamplesIterator>
auto compute_variance_per_feature(const IndicesIterator&          indices_range_first,
                                  const IndicesIterator&          indices_range_last,
                                  SamplesIterator                 samples_range_first,
                                  SamplesIterator                 samples_range_last,
                                  std::size_t                     n_features,
                                  const std::vector<std::size_t>& feature_mask) {
    using DataType = typename SamplesIterator::value_type;

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    assert(n_samples > 1);

    const auto mean_per_feature = compute_mean_per_feature(
        indices_range_first, indices_range_last, samples_range_first, samples_range_last, n_features, feature_mask);

    auto variance_per_feature = std::vector<DataType>(feature_mask.size());

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        auto features_accumulator = std::vector<DataType>(feature_mask.size());

        for (std::size_t mask_index = 0; mask_index < feature_mask.size(); ++mask_index) {
            const auto temp = samples_range_first[*indices_range_first * n_features + feature_mask[mask_index]] -
                              mean_per_feature[mask_index];
            variance_per_feature[mask_index] += temp * temp;
        }
    }
    std::transform(variance_per_feature.begin(),
                   variance_per_feature.end(),
                   variance_per_feature.begin(),
                   [n_samples](const auto& feature) { return feature / (n_samples - 1); });

    return variance_per_feature;
}

template <typename SamplesIterator>
std::size_t argmax_variance_per_feature(SamplesIterator samples_range_first,
                                        SamplesIterator samples_range_last,
                                        std::size_t     n_features) {
    using DataType = typename SamplesIterator::value_type;

    const auto n_samples = common::utils::get_n_samples(samples_range_first, samples_range_last, n_features);

    assert(n_samples > 1);

    const auto mean_per_feature = compute_mean_per_feature(samples_range_first, samples_range_last, n_features);

    auto variance_per_feature = std::vector<DataType>(n_features);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        auto features_accumulator = std::vector<DataType>(n_features);

        std::transform(samples_range_first + sample_index * n_features,
                       samples_range_first + sample_index * n_features + n_features,
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
std::size_t argmax_variance_per_feature(const IndicesIterator& indices_range_first,
                                        const IndicesIterator& indices_range_last,
                                        SamplesIterator        samples_range_first,
                                        SamplesIterator        samples_range_last,
                                        std::size_t            n_features) {
    using DataType = typename SamplesIterator::value_type;

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    assert(n_samples > 1);

    const auto mean_per_feature = compute_mean_per_feature(
        indices_range_first, indices_range_last, samples_range_first, samples_range_last, n_features);

    auto variance_per_feature = std::vector<DataType>(n_features);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        auto features_accumulator = std::vector<DataType>(n_features);

        std::transform(samples_range_first + indices_range_first[sample_index] * n_features,
                       samples_range_first + indices_range_first[sample_index] * n_features + n_features,
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
std::size_t argmax_variance_per_feature(const IndicesIterator&          indices_range_first,
                                        const IndicesIterator&          indices_range_last,
                                        SamplesIterator                 samples_range_first,
                                        SamplesIterator                 samples_range_last,
                                        std::size_t                     n_features,
                                        const std::vector<std::size_t>& feature_mask) {
    using DataType = typename SamplesIterator::value_type;

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    assert(n_samples > 1);

    const auto mean_per_feature = compute_mean_per_feature(
        indices_range_first, indices_range_last, samples_range_first, samples_range_last, n_features, feature_mask);

    auto variance_per_feature = std::vector<DataType>(feature_mask.size());

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        auto features_accumulator = std::vector<DataType>(feature_mask.size());

        for (std::size_t mask_index = 0; mask_index < feature_mask.size(); ++mask_index) {
            const auto temp = samples_range_first[*indices_range_first * n_features + feature_mask[mask_index]] -
                              mean_per_feature[mask_index];
            variance_per_feature[mask_index] += temp * temp;
        }
    }
    // return the feature index
    return feature_mask[math::statistics::argmax(variance_per_feature.begin(), variance_per_feature.end())];
}

template <typename SamplesIterator>
typename SamplesIterator::value_type compute_variance(SamplesIterator samples_range_first,
                                                      SamplesIterator samples_range_last) {
    using DataType = typename SamplesIterator::value_type;

    const auto n_elements = std::distance(samples_range_first, samples_range_last);
    const auto sum        = std::accumulate(samples_range_first, samples_range_last, static_cast<DataType>(0));
    const auto sum_of_squares =
        std::inner_product(samples_range_first, samples_range_last, samples_range_first, static_cast<DataType>(0));

    return (sum_of_squares - sum * sum / n_elements) / (n_elements - 1);
}

}  // namespace math::statistics