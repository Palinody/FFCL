#pragma once

#include "cpp_clustering/common/Utils.hpp"

#include <algorithm>  // std::minmax_element
#include <cmath>
#include <iterator>
#include <limits>  // std::numeric_limits<T>::max()
#include <vector>

namespace cpp_clustering::silhouette_method {

/**
 * @brief counts the number of samples associated to a centroid
 * The size of the resulting container is max_element(in_container) + 1
 * E.g.:
 *    IN: {2, 2, 1, 0, 0, 1, 0, 2, 0, 5}
 *    OUT: {4, 2, 3, 0, 0, 1}
 *
 * @tparam IteratorInt
 * @param sample_to_closest_centroid_index_first
 * @param sample_to_closest_centroid_index_last
 * @return std::vector<std::size_t>
 */
template <typename IteratorInt>
std::vector<std::size_t> get_n_samples_per_centroid(IteratorInt sample_to_closest_centroid_index_first,
                                                    IteratorInt sample_to_closest_centroid_index_last) {
    static_assert(std::is_integral<typename IteratorInt::value_type>::value, "Data should be integer type.");

    const auto greatest_label_index =
        *std::max_element(sample_to_closest_centroid_index_first, sample_to_closest_centroid_index_last);
    // n_labels = 1 + largest_label_index
    auto labels_histogram = std::vector<std::size_t>(greatest_label_index + 1);

    while (sample_to_closest_centroid_index_first != sample_to_closest_centroid_index_last) {
        ++labels_histogram[*(sample_to_closest_centroid_index_first++)];
    }
    return labels_histogram;
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
 * @param sample_first
 * @param sample_last
 * @param sample_to_closest_centroid_index_first
 * @param sample_to_closest_centroid_index_last
 * @param n_features
 * @return std::vector<typename IteratorFloat::value_type>
 */
template <typename IteratorFloat, typename IteratorInt>
std::vector<typename IteratorFloat::value_type> cohesion(const IteratorFloat& sample_first,
                                                         const IteratorFloat& sample_last,
                                                         const IteratorInt&   sample_to_closest_centroid_index_first,
                                                         const IteratorInt&   sample_to_closest_centroid_index_last,
                                                         std::size_t          n_features) {
    static_assert(std::is_floating_point<typename IteratorFloat::value_type>::value,
                  "Data should be a floating point type.");
    static_assert(std::is_integral<typename IteratorInt::value_type>::value, "Data should be integer type.");

    using FloatType = typename IteratorFloat::value_type;

    const auto n_samples = common::utils::get_n_samples(sample_first, sample_last, n_features);

    const auto centroids_indices_count =
        get_n_samples_per_centroid(sample_to_closest_centroid_index_first, sample_to_closest_centroid_index_last);

    auto samples_cohesion_values = std::vector<FloatType>(n_samples);

    for (std::size_t i = 0; i < n_samples; ++i) {
        // get the centroid associated to the current sample
        const std::size_t current_centroid_index = *(sample_to_closest_centroid_index_first + i);
        // iterate over all the other data samples by skipping when i == j or C_I != C_J
        for (std::size_t j = 0; j < n_samples; ++j) {
            // get the centroid associated to the other sample
            const std::size_t other_centroid_index = *(sample_to_closest_centroid_index_first + j);
            // compute distance only when sample is not itself and belongs to the same cluster
            if ((current_centroid_index == other_centroid_index) && (i != j)) {
                // accumulate the squared distances
                samples_cohesion_values[i] += cpp_clustering::heuristic::heuristic(
                    /*first sample begin=*/sample_first + i * n_features,
                    /*first sample end=*/sample_first + i * n_features + n_features,
                    /*other sample begin=*/sample_first + j * n_features);
            }
        }
        // number of samples in the current centroid
        const auto centroid_samples_count = centroids_indices_count[current_centroid_index];
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
 * @param sample_first
 * @param sample_last
 * @param sample_to_closest_centroid_index_first
 * @param sample_to_closest_centroid_index_last
 * @param n_features
 * @return std::vector<typename IteratorFloat::value_type>
 */
template <typename IteratorFloat, typename IteratorInt>
std::vector<typename IteratorFloat::value_type> separation(const IteratorFloat& sample_first,
                                                           const IteratorFloat& sample_last,
                                                           const IteratorInt&   sample_to_closest_centroid_index_first,
                                                           const IteratorInt&   sample_to_closest_centroid_index_last,
                                                           std::size_t          n_features) {
    static_assert(std::is_floating_point<typename IteratorFloat::value_type>::value,
                  "Data should be a floating point type.");
    static_assert(std::is_integral<typename IteratorInt::value_type>::value, "Data should be integer type.");

    using FloatType = typename IteratorFloat::value_type;

    const auto n_samples = common::utils::get_n_samples(sample_first, sample_last, n_features);

    const auto centroids_indices_count =
        get_n_samples_per_centroid(sample_to_closest_centroid_index_first, sample_to_closest_centroid_index_last);

    auto samples_separation_values = std::vector<FloatType>(n_samples);

    for (std::size_t i = 0; i < n_samples; ++i) {
        // get the centroid associated to the current sample
        const std::size_t current_centroid_index = *(sample_to_closest_centroid_index_first + i);
        // the sum of distances from the current sample to other samples from different centroids
        auto sample_to_other_centroid_samples_distance_mean = std::vector<FloatType>(centroids_indices_count.size());
        // iterate over all the other data samples (j) that dont belong to the same cluster
        for (std::size_t j = 0; j < n_samples; ++j) {
            // get the centroid associated to the other sample
            const std::size_t other_centroid_index = *(sample_to_closest_centroid_index_first + j);
            // compute distance only when sample belongs to a different cluster
            if (current_centroid_index != other_centroid_index) {
                // accumulate the squared distances for the correct centroid index
                sample_to_other_centroid_samples_distance_mean[other_centroid_index] +=
                    cpp_clustering::heuristic::heuristic(
                        /*first sample begin=*/sample_first + i * n_features,
                        /*first sample end=*/sample_first + i * n_features + n_features,
                        /*other sample begin=*/sample_first + j * n_features);
            }
        }
        // normalize each cluster mean distance sum by each cluster's number of samples
        // destination_i = source_i / n if n != 0 else infinity
        std::transform(
            /*source_first*/ sample_to_other_centroid_samples_distance_mean.begin(),
            /*source_last*/ sample_to_other_centroid_samples_distance_mean.end(),
            /*n*/ centroids_indices_count.begin(),
            /*destination*/ sample_to_other_centroid_samples_distance_mean.begin(),
            [](const auto& dist, const auto& idx_count) {
                // if centroid size is zero, set distance mean to infinity so that it doesnt
                // get picked otherwise normalize by dividing normally
                return (idx_count ? dist / static_cast<FloatType>(idx_count) : std::numeric_limits<FloatType>::max());
            });
        // set the current centroid index distance value to infinity so that it doesnt get chosen
        sample_to_other_centroid_samples_distance_mean[current_centroid_index] = std::numeric_limits<FloatType>::max();
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
 * @param sample_first
 * @param sample_last
 * @param sample_to_closest_centroid_index_first
 * @param sample_to_closest_centroid_index_last
 * @param n_features
 */
template <typename IteratorFloat, typename IteratorInt>
std::vector<typename IteratorFloat::value_type> silhouette(const IteratorFloat& sample_first,
                                                           const IteratorFloat& sample_last,
                                                           const IteratorInt&   sample_to_closest_centroid_index_first,
                                                           const IteratorInt&   sample_to_closest_centroid_index_last,
                                                           std::size_t          n_features) {
    static_assert(std::is_floating_point<typename IteratorFloat::value_type>::value,
                  "Data should be a floating point type.");
    static_assert(std::is_integral<typename IteratorInt::value_type>::value, "Data should be integer type.");

    using FloatType = typename IteratorFloat::value_type;

    const auto n_samples = common::utils::get_n_samples(sample_first, sample_last, n_features);

    const auto cohesion_values = cohesion(sample_first,
                                          sample_last,
                                          sample_to_closest_centroid_index_first,
                                          sample_to_closest_centroid_index_last,
                                          n_features);

    const auto separation_values = separation(sample_first,
                                              sample_last,
                                              sample_to_closest_centroid_index_first,
                                              sample_to_closest_centroid_index_last,
                                              n_features);

    auto silhouette_values = std::vector<FloatType>(n_samples);

    for (std::size_t i = 0; i < n_samples; ++i) {
        const FloatType coh = cohesion_values[i];
        const FloatType sep = separation_values[i];

        if (coh < sep) {
            silhouette_values[i] = static_cast<FloatType>(1) - coh / sep;
        } else if (coh == sep) {
            silhouette_values[i] = 0;
        } else {
            silhouette_values[i] = sep / coh - 1;
        }
    }
    return silhouette_values;
}

template <typename IteratorFloat>
typename IteratorFloat::value_type get_mean_silhouette_coefficient(const IteratorFloat& samples_silhouette_first,
                                                                   const IteratorFloat& samples_silhouette_last) {
    using FloatType = typename IteratorFloat::value_type;

    const auto n_elements = std::distance(samples_silhouette_first, samples_silhouette_last);
    return std::accumulate(samples_silhouette_first, samples_silhouette_last, static_cast<FloatType>(0)) / n_elements;
}

}  // namespace cpp_clustering::silhouette_method
