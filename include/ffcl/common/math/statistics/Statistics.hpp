#pragma once

#include "ffcl/common/Utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace ffcl::common::math::statistics {

template <typename DataType = double, typename Iterator>
std::vector<DataType> normalize_min_max(const Iterator& first, const Iterator& last) {
    const auto [min, max] = std::minmax_element(first, last);

    if (*min == *max) {
        const std::size_t n_elements = std::distance(first, last);
        // if all the values are the same distribute the weights equaly
        return std::vector<DataType>(n_elements, 1.0 / n_elements);
    }
    auto res = std::vector<DataType>(last - first);
    // closest objects get a higher score. Distance zero -> 1
    std::transform(first, last, res.begin(), [&min, &max](const auto& dist) {
        return static_cast<DataType>(1) - (dist - *min) / (*max - *min);
    });
    return res;
}

template <typename DataType = double, typename Iterator>
std::vector<DataType> normalize(const Iterator& first, const Iterator& last) {
    auto normalized_vector = std::vector<DataType>(std::distance(first, last));

    // Calculate the sum of all elements in the vector
    const DataType sum = std::accumulate(first, last, static_cast<DataType>(0));
    // Divide each element by the sum to normalize the vector
    std::transform(first, last, normalized_vector.begin(), [sum](const DataType& val) { return val / sum; });

    return normalized_vector;
}

template <typename Container>
void min_container_inplace_left(Container& left_container, const Container& right_container) {
    // compute element-wise minimum and place the result in the left container
    std::transform(left_container.begin(),
                   left_container.end(),
                   right_container.begin(),
                   left_container.begin(),
                   [](const auto& x, const auto& y) { return std::min(x, y); });
}

template <typename Iterator>
auto argmin(const Iterator& first, const Iterator& last) -> typename std::iterator_traits<Iterator>::value_type {
    auto min_it = std::min_element(first, last);
    return std::distance(first, min_it);
}

template <typename Iterator>
auto argmax(const Iterator& first, const Iterator& last) -> typename std::iterator_traits<Iterator>::value_type {
    auto max_it = std::max_element(first, last);
    return std::distance(first, max_it);
}

template <typename Iterator>
auto get_min_index_value_pair(const Iterator& first, const Iterator& last)
    -> std::pair<std::size_t, typename std::iterator_traits<Iterator>::value_type> {
    using DataType = typename std::iterator_traits<Iterator>::value_type;

    const auto        min_element = std::min_element(first, last);
    const std::size_t min_index   = std::distance(first, min_element);
    const DataType    min_value   = *min_element;

    return {min_index, min_value};
}

template <typename Iterator>
auto get_max_index_value_pair(const Iterator& first, const Iterator& last)
    -> std::pair<std::size_t, typename std::iterator_traits<Iterator>::value_type> {
    using DataType = typename std::iterator_traits<Iterator>::value_type;

    const auto        max_element = std::max_element(first, last);
    const std::size_t max_index   = std::distance(first, max_element);
    const DataType    max_value   = *max_element;

    return {max_index, max_value};
}

template <typename Iterator>
auto get_second_max_index_value_pair(const Iterator& first, const Iterator& last)
    -> std::pair<std::size_t, typename std::iterator_traits<Iterator>::value_type> {
    using DataType = typename std::iterator_traits<Iterator>::value_type;

    // Get the index of the maximum element
    auto           max_it    = std::max_element(first, last);
    std::size_t    max_idx   = std::distance(first, max_it);
    const DataType max_value = *max_it;

    if (std::distance(first, last) < 2) {
        return {max_idx, max_value};
    }
    // Replace the maximum element with the lowest possible value
    *max_it = std::numeric_limits<DataType>::lowest();

    // Get the index of the second maximum element
    const auto     second_max_it    = std::max_element(first, last);
    std::size_t    second_max_idx   = std::distance(first, second_max_it);
    const DataType second_max_value = *second_max_it;

    // Restore the original maximum value
    *max_it = max_value;

    return {second_max_idx, second_max_value};
}

template <typename Iterator>
auto find_nth_smallest_index_and_element(const Iterator& first, const Iterator& last, std::size_t n)
    -> std::pair<std::size_t, typename std::iterator_traits<Iterator>::value_type> {
    using DataType = typename std::iterator_traits<Iterator>::value_type;

    assert(n <= static_cast<std::size_t>(std::distance(first, last)) &&
           "N-th smallest requested element shouldn't be greater than the container's size.");

    std::vector<DataType> data_sorted(last - first);
    std::copy(first, last, data_sorted.begin());
    std::sort(data_sorted.begin(), data_sorted.end());

    // get the n-th smallest element
    auto nth_smallest = data_sorted.at(n - 1);

    // find the index of the n-th smallest element in the original container
    auto it    = std::find(first, last, nth_smallest);
    auto index = std::distance(first, it);

    return {index, nth_smallest};
}

template <typename SamplesIterator>
auto compute_mean_per_feature(const SamplesIterator& samples_range_first,
                              const SamplesIterator& samples_range_last,
                              std::size_t            n_features) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const auto n_samples = get_n_samples(samples_range_first, samples_range_last, n_features);

    auto mean_per_feature = std::vector<DataType>(n_features);

    for (auto samples_range_it = samples_range_first; samples_range_it != samples_range_last;
         samples_range_it += n_features) {
        std::transform(samples_range_it,
                       samples_range_it + n_features,
                       mean_per_feature.begin(),
                       mean_per_feature.begin(),
                       std::plus<>());
    }
    std::transform(/**/ mean_per_feature.begin(),
                   /**/ mean_per_feature.end(),
                   /**/ mean_per_feature.begin(),
                   [n_samples](const auto& feature) { return feature / n_samples; });
    return mean_per_feature;
}

template <typename IndicesIterator, typename SamplesIterator>
auto compute_mean_per_feature(const IndicesIterator& indices_range_first,
                              const IndicesIterator& indices_range_last,
                              const SamplesIterator& samples_range_first,
                              const SamplesIterator& samples_range_last,
                              std::size_t            n_features) {
    ignore_parameters(samples_range_last);

    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    auto mean_per_feature = std::vector<DataType>(n_features);

    for (auto indices_range_it = indices_range_first; indices_range_it != indices_range_last; ++indices_range_it) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            mean_per_feature[feature_index] += samples_range_first[*indices_range_it * n_features + feature_index];
        }
    }
    std::transform(/**/ mean_per_feature.begin(),
                   /**/ mean_per_feature.end(),
                   /**/ mean_per_feature.begin(),
                   [n_samples](const auto& feature) { return feature / n_samples; });
    return mean_per_feature;
}

template <typename IndicesIterator, typename SamplesIterator>
auto compute_mean_per_feature(const IndicesIterator&          indices_range_first,
                              const IndicesIterator&          indices_range_last,
                              const SamplesIterator&          samples_range_first,
                              const SamplesIterator&          samples_range_last,
                              std::size_t                     n_features,
                              const std::vector<std::size_t>& feature_mask) {
    ignore_parameters(samples_range_last);

    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    auto mean_per_feature = std::vector<DataType>(feature_mask.size());

    for (auto indices_range_it = indices_range_first; indices_range_it != indices_range_last; ++indices_range_it) {
        for (std::size_t mask_index = 0; mask_index < feature_mask.size(); ++mask_index) {
            mean_per_feature[mask_index] +=
                samples_range_first[*indices_range_it * n_features + feature_mask[mask_index]];
        }
    }
    std::transform(
        /**/ mean_per_feature.begin(),
        /**/ mean_per_feature.end(),
        /**/ mean_per_feature.begin(),
        [n_samples](const auto& feature) { return feature / n_samples; });
    return mean_per_feature;
}

template <typename SamplesIterator>
auto compute_variance_per_feature(const SamplesIterator& samples_range_first,
                                  const SamplesIterator& samples_range_last,
                                  std::size_t            n_features) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const auto n_samples = get_n_samples(samples_range_first, samples_range_last, n_features);

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
                                  const SamplesIterator& samples_range_first,
                                  const SamplesIterator& samples_range_last,
                                  std::size_t            n_features) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    const auto mean_per_feature = compute_mean_per_feature(/**/ indices_range_first,
                                                           /**/ indices_range_last,
                                                           /**/ samples_range_first,
                                                           /**/ samples_range_last,
                                                           /**/ n_features);

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
                                  const SamplesIterator&          samples_range_first,
                                  const SamplesIterator&          samples_range_last,
                                  std::size_t                     n_features,
                                  const std::vector<std::size_t>& feature_mask) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

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
std::size_t argmax_variance_per_feature(const SamplesIterator& samples_range_first,
                                        const SamplesIterator& samples_range_last,
                                        std::size_t            n_features) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const auto n_samples = get_n_samples(samples_range_first, samples_range_last, n_features);

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
                                        const SamplesIterator& samples_range_first,
                                        const SamplesIterator& samples_range_last,
                                        std::size_t            n_features) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

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
                                        const SamplesIterator&          samples_range_first,
                                        const SamplesIterator&          samples_range_last,
                                        std::size_t                     n_features,
                                        const std::vector<std::size_t>& feature_mask) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

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
auto compute_variance(SamplesIterator samples_range_first, SamplesIterator samples_range_last) ->
    typename std::iterator_traits<SamplesIterator>::value_type {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const auto n_elements = std::distance(samples_range_first, samples_range_last);

    const auto sum = std::accumulate(samples_range_first, samples_range_last, static_cast<DataType>(0));

    const auto sum_of_squares = std::inner_product(/**/ samples_range_first,
                                                   /**/ samples_range_last,
                                                   /**/ samples_range_first,
                                                   /**/ static_cast<DataType>(0));

    return (sum_of_squares - sum * sum / n_elements) / (n_elements - 1);
}

}  // namespace ffcl::common::math::statistics