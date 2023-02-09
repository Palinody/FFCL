#pragma once

#include "cpp_clustering/common/Utils.hpp"

#include <algorithm>  // std::minmax_element
#include <cmath>
#include <iterator>
#include <limits>  // std::numeric_limits<T>::max()
#include <vector>

namespace cpp_clustering::silhouette_method {

/**
 * @brief counts the number of centroid indices and puts them in increasing centroid index order
 * The size of the resulting container is max_element(in_container) + 1
 * E.g.:
 *    IN: {2, 2, 1, 0, 0, 1, 0, 2, 0, 5}
 *    OUT: {4, 2, 3, 0, 0, 1}
 *
 * @tparam IteratorInt
 * @param closest_centroids_indices_first
 * @param closest_centroids_indices_last
 * @return std::vector<std::size_t>
 */
template <typename IteratorInt>
std::vector<std::size_t> get_n_samples_per_centroid(IteratorInt closest_centroids_indices_first,
                                                    IteratorInt closest_centroids_indices_last) {
    static_assert(std::is_integral<typename IteratorInt::value_type>::value, "Data should be integer type.");

    const auto centroids_max_index = *std::max_element(closest_centroids_indices_first, closest_centroids_indices_last);

    auto centroids_indices_count = std::vector<std::size_t>(centroids_max_index + 1);

    while (closest_centroids_indices_first != closest_centroids_indices_last) {
        // increment the counter for the current centroid index and move on to next element
        ++centroids_indices_count[*(closest_centroids_indices_first++)];
    }
    return centroids_indices_count;
}
/**
 * @brief The cohesion can be interpreted as a measure of how well each data samples are assigned to a cluster.
 * FORMULA:
 *    a[i] = 1 / (|C_{I}| - 1) * sum_{j in C_{I}, i != j}{d(i, j)}
 * NOTATION:
 *    i: data sample index
 *    C_{I}: cluster of the data sample with |C_{I}| the number of points belonging to cluster C_{I}
 *    d(i, j): distance between data points i and j. We subtract 1 to the division because d(i, i) is excluded
 *
 * @tparam IteratorFloat
 * @tparam IteratorInt
 * @param data_first
 * @param data_last
 * @param closest_centroids_indices_first
 * @param closest_centroids_indices_last
 * @param n_features
 * @return std::vector<typename IteratorFloat::value_type>
 */
template <typename IteratorFloat, typename IteratorInt>
std::vector<typename IteratorFloat::value_type> cohesion(const IteratorFloat& data_first,
                                                         const IteratorFloat& data_last,
                                                         const IteratorInt&   closest_centroids_indices_first,
                                                         const IteratorInt&   closest_centroids_indices_last,
                                                         std::size_t          n_features) {
    static_assert(std::is_floating_point<typename IteratorFloat::value_type>::value,
                  "Data should be a floating point type.");
    static_assert(std::is_integral<typename IteratorInt::value_type>::value, "Data should be integer type.");

    const auto n_samples = common::utils::get_n_samples(data_first, data_last, n_features);
    const auto centroids_indices_count =
        get_n_samples_per_centroid(closest_centroids_indices_first, closest_centroids_indices_last);

    using FloatType              = typename IteratorFloat::value_type;
    auto samples_cohesion_values = std::vector<FloatType>(n_samples);

    for (std::size_t i = 0; i < n_samples; ++i) {
        // get the centroid associated to the current sample
        const std::size_t current_centroid_idx = *(closest_centroids_indices_first + i);
        // iterate over all the other data samples by skipping when i == j or C_I != C_J
        for (std::size_t j = 0; j < n_samples; ++j) {
            // get the centroid associated to the other sample
            const std::size_t other_centroid_idx = *(closest_centroids_indices_first + j);
            // compute distance only when sample is not itself and belongs to the same cluster
            if ((current_centroid_idx == other_centroid_idx) && (i != j)) {
                // accumulate the squared distances
                samples_cohesion_values[i] += cpp_clustering::heuristic::heuristic(
                    /*first sample begin=*/data_first + i * n_features,
                    /*first sample end=*/data_first + i * n_features + n_features,
                    /*other sample begin=*/data_first + j * n_features);
            }
        }
        // number of samples in the current centroid
        const auto centroid_samples_count = centroids_indices_count[current_centroid_idx];
        // normalise the sum of the distances from the current sample to all the other samples in the same centroid
        // divide by one if the cluster contains 0 or 1 sample
        samples_cohesion_values[i] /=
            (centroid_samples_count < 2 ? 1 : static_cast<FloatType>(centroid_samples_count - 1));
    }
    return samples_cohesion_values;
}
/**
 * @brief The separation measures the dissimilarity of a sample i from a cluster I to a cluster J, with I != J.
 * The cluster with the smallest mean dissimilarity is said to be the "neighboring cluster" of i because it is the next
 * best fit cluster for point i.
 * FORMULA:
 *    b[i] = min_{I!=J} 1/|C_{J}| * sum_{J in C_{J}}{d(i, j)}
 * NOTATION:
 *    i: data sample index
 *    C_{j}: cluster of the data sample with |C_{j}| the number of points belonging to cluster C_{j}
 *    d(i, j): distance between data points i and j
 *
 * @tparam IteratorFloat
 * @tparam IteratorInt
 * @param data_first
 * @param data_last
 * @param closest_centroids_indices_first
 * @param closest_centroids_indices_last
 * @param n_features
 * @return std::vector<typename IteratorFloat::value_type>
 */
template <typename IteratorFloat, typename IteratorInt>
std::vector<typename IteratorFloat::value_type> separation(const IteratorFloat& data_first,
                                                           const IteratorFloat& data_last,
                                                           const IteratorInt&   closest_centroids_indices_first,
                                                           const IteratorInt&   closest_centroids_indices_last,
                                                           std::size_t          n_features) {
    static_assert(std::is_floating_point<typename IteratorFloat::value_type>::value,
                  "Data should be a floating point type.");
    static_assert(std::is_integral<typename IteratorInt::value_type>::value, "Data should be integer type.");

    const auto n_samples = common::utils::get_n_samples(data_first, data_last, n_features);
    const auto centroids_indices_count =
        get_n_samples_per_centroid(closest_centroids_indices_first, closest_centroids_indices_last);

    using FloatType                = typename IteratorFloat::value_type;
    auto samples_separation_values = std::vector<FloatType>(n_samples);

    for (std::size_t i = 0; i < n_samples; ++i) {
        // get the centroid associated to the current sample
        const std::size_t current_centroid_idx = *(closest_centroids_indices_first + i);
        // the sum of distances from the current sample to other samples from different centroids
        auto sample_to_other_centroid_samples_distance_mean = std::vector<FloatType>(centroids_indices_count.size());
        // iterate over all the other data samples (j) that dont belong to the same cluster
        for (std::size_t j = 0; j < n_samples; ++j) {
            // get the centroid associated to the other sample
            const std::size_t other_centroid_idx = *(closest_centroids_indices_first + j);
            // compute distance only when sample belongs to a different cluster
            if (current_centroid_idx != other_centroid_idx) {
                // accumulate the squared distances for the correct centroid index
                sample_to_other_centroid_samples_distance_mean[other_centroid_idx] +=
                    cpp_clustering::heuristic::heuristic(
                        /*first sample begin=*/data_first + i * n_features,
                        /*first sample end=*/data_first + i * n_features + n_features,
                        /*other sample begin=*/data_first + j * n_features);
            }
        }
        // normalize each cluster mean distance sum by each cluster's number of samples
        std::transform(sample_to_other_centroid_samples_distance_mean.begin(),
                       sample_to_other_centroid_samples_distance_mean.end(),
                       centroids_indices_count.begin(),
                       sample_to_other_centroid_samples_distance_mean.begin(),
                       [](const auto& dist, const auto& idx_count) {
                           // if centroid size is zero, set distance mean to an arbitrary large value so that it doesnt
                           // get picked otherwise normalize by dividing normally
                           return (idx_count != 0 ? dist / static_cast<FloatType>(idx_count)
                                                  : std::numeric_limits<FloatType>::max());
                       });
        // set the current centroid index distance value to a fake large distance so that it doesnt get chosen
        sample_to_other_centroid_samples_distance_mean[current_centroid_idx] = std::numeric_limits<FloatType>::max();
        // normalise the sum of the distances from the current sample to all the other samples in the same centroid
        samples_separation_values[i] = *std::min_element(sample_to_other_centroid_samples_distance_mean.begin(),
                                                         sample_to_other_centroid_samples_distance_mean.end());
    }
    return samples_separation_values;
}
/**
 * @brief
 * FORMULA:
 *      s[i] = (b[i] - a[i]) / (max(a[i], b[i])) and s[i] = 0 if |C{I}| = 1
 *      s[i]:
 *            1 - a[i]/b[i], if a[i] < b[i]
 *            0            , if a[i] = b[i]
 *            b[i]/a[i]-1  , if a[i] > b[i]
 *      Thus: s[i] in [-1, 1]
 *
 * @tparam IteratorFloat
 * @tparam IteratorInt
 * @param data_first
 * @param data_last
 * @param closest_centroids_indices_first
 * @param closest_centroids_indices_last
 * @param n_features
 */
template <typename IteratorFloat, typename IteratorInt>
std::vector<typename IteratorFloat::value_type> silhouette(const IteratorFloat& data_first,
                                                           const IteratorFloat& data_last,
                                                           const IteratorInt&   closest_centroids_indices_first,
                                                           const IteratorInt&   closest_centroids_indices_last,
                                                           std::size_t          n_features) {
    static_assert(std::is_floating_point<typename IteratorFloat::value_type>::value,
                  "Data should be a floating point type.");
    static_assert(std::is_integral<typename IteratorInt::value_type>::value, "Data should be integer type.");

    using FloatType = typename IteratorFloat::value_type;

    const auto n_samples = common::utils::get_n_samples(data_first, data_last, n_features);

    const auto samples_cohesion_values =
        cohesion(data_first, data_last, closest_centroids_indices_first, closest_centroids_indices_last, n_features);

    const auto samples_separation_values =
        separation(data_first, data_last, closest_centroids_indices_first, closest_centroids_indices_last, n_features);

    auto samples_silhouette_values = std::vector<FloatType>(n_samples);

    for (std::size_t i = 0; i < n_samples; ++i) {
        const FloatType coh = samples_cohesion_values[i];
        const FloatType sep = samples_separation_values[i];

        if (coh < sep) {
            samples_silhouette_values[i] = static_cast<FloatType>(1) - coh / sep;
        } else if (coh == sep) {
            samples_silhouette_values[i] = 0;
        } else {
            samples_silhouette_values[i] = sep / coh - 1;
        }
    }
    return samples_silhouette_values;
}

template <typename IteratorFloat>
typename IteratorFloat::value_type get_mean_silhouette_coefficient(const IteratorFloat& samples_silhouette_first,
                                                                   const IteratorFloat& samples_silhouette_last) {
    using FloatType = typename IteratorFloat::value_type;

    FloatType n_elements = samples_silhouette_last - samples_silhouette_first;
    return std::accumulate(samples_silhouette_first, samples_silhouette_last, static_cast<FloatType>(0)) / n_elements;
}

}  // namespace cpp_clustering::silhouette_method