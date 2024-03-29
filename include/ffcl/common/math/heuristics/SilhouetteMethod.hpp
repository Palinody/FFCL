#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <vector>

namespace ffcl::common::math::heuristics {

/**
 * @brief counts the number of samples associated to a centroid
 * The size of the resulting container is max_element(in_container) + 1
 * E.g.:
 * Each element is the centroid index associated with a sample:
 *      IN: {2, 2, 1, 0, 0, 1, 0, 2, 0, 5}
 * Each element is the count of centroid indices found for each centroid. The number of elements in the array is equal
 * to the max centroid index found + 1:
 *      OUT: {4, 2, 3, 0, 0, 1}
 * WARN: this method will discard the last centroid(s) ind(ex/ices) is they don't have been associated with any sample
 * since the function won't be able to find them by index. However, if the last centroid index has been associated with
 * at least one sample, then the smaller centroid indices will be automatically set to 0 as in the example shown above.
 *
 * @tparam ClusterLabelsIterator
 * @param cluster_labels_range_first
 * @param cluster_labels_range_last
 * @return std::vector<IndexType>
 */
template <typename ClusterLabelsIterator>
auto get_cluster_sizes(const ClusterLabelsIterator& cluster_labels_range_first,
                       const ClusterLabelsIterator& cluster_labels_range_last) {
    using ClusterLabelType = typename std::iterator_traits<ClusterLabelsIterator>::value_type;

    static_assert(std::is_integral<ClusterLabelType>::value, "Cluster labels must be integer.");

    // n_centroids = 1 + greatest_centroid_index
    const auto n_centroids = 1 + *std::max_element(cluster_labels_range_first, cluster_labels_range_last);
    // n_labels = 1 + largest_label_index
    auto labels_histogram = std::vector<ClusterLabelType>(n_centroids);

    auto cluster_labels_range_it = cluster_labels_range_first;

    while (cluster_labels_range_it != cluster_labels_range_last) {
        ++labels_histogram[*(cluster_labels_range_it++)];
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
 * @tparam SamplesIterator
 * @tparam ClusterLabelsIterator
 * @param samples_range_first
 * @param samples_range_last
 * @param n_features
 * @param cluster_labels_range_first
 * @param cluster_labels_range_last
 * @return std::vector<typename std::iterator_traits<SamplesIterator>::value_type>
 */
template <typename SamplesIterator, typename ClusterLabelsIterator>
auto cohesion(const SamplesIterator&       samples_range_first,
              const SamplesIterator&       samples_range_last,
              std::size_t                  n_features,
              const ClusterLabelsIterator& cluster_labels_range_first,
              const ClusterLabelsIterator& cluster_labels_range_last) {
    using ElementType = typename std::iterator_traits<SamplesIterator>::value_type;

    using ClusterLabelType = typename std::iterator_traits<ClusterLabelsIterator>::value_type;

    static_assert(std::is_trivial_v<ElementType>, "ElementType must be trivial.");

    static_assert(std::is_integral<ClusterLabelType>::value, "Cluster labels must be integer.");

    const auto n_samples = get_n_samples(samples_range_first, samples_range_last, n_features);

    const auto cluster_sizes = get_cluster_sizes(cluster_labels_range_first, cluster_labels_range_last);

    auto samples_cohesion_scores = std::vector<ElementType>(n_samples);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // get the centroid associated to the current sample
        const std::size_t centroid_index = cluster_labels_range_first[sample_index];
        // iterate over all the other data samples by skipping when i == j or C_I != C_J
        for (std::size_t other_sample_index = 0; other_sample_index < n_samples; ++other_sample_index) {
            // get the centroid associated to the other sample
            const std::size_t other_centroid_index = cluster_labels_range_first[other_sample_index];
            // compute distance only when sample is not itself and belongs to the same cluster
            if ((centroid_index == other_centroid_index) && (sample_index != other_sample_index)) {
                // accumulate the squared distances
                samples_cohesion_scores[sample_index] +=
                    auto_distance(samples_range_first + sample_index * n_features,
                                  samples_range_first + sample_index * n_features + n_features,
                                  samples_range_first + other_sample_index * n_features,
                                  samples_range_first + other_sample_index * n_features + n_features);
            }
        }
        // number of samples in the current centroid
        const auto cluster_size = cluster_sizes[centroid_index];
        // normalise the sum of the distances from the current sample to all the other samples in the same centroid
        // divide by one if the cluster contains 0 or 1 sample
        if (cluster_size > 1) {
            samples_cohesion_scores[sample_index] /= static_cast<ElementType>(cluster_size - 1);
        }
    }
    return samples_cohesion_scores;
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
 * @tparam SamplesIterator
 * @tparam ClusterLabelsIterator
 * @param samples_range_first
 * @param samples_range_last
 * @param n_features
 * @param cluster_labels_range_first
 * @param cluster_labels_range_last
 * @return std::vector<typename std::iterator_traits<SamplesIterator>::value_type>
 */
template <typename SamplesIterator, typename ClusterLabelsIterator>
auto separation(const SamplesIterator&       samples_range_first,
                const SamplesIterator&       samples_range_last,
                std::size_t                  n_features,
                const ClusterLabelsIterator& cluster_labels_range_first,
                const ClusterLabelsIterator& cluster_labels_range_last) {
    using ElementType = typename std::iterator_traits<SamplesIterator>::value_type;

    using ClusterLabelType = typename std::iterator_traits<ClusterLabelsIterator>::value_type;

    static_assert(std::is_trivial_v<ElementType>, "ElementType must be trivial.");

    static_assert(std::is_integral<ClusterLabelType>::value, "Cluster labels must be integer.");

    const auto n_samples = get_n_samples(samples_range_first, samples_range_last, n_features);

    const auto cluster_sizes = get_cluster_sizes(cluster_labels_range_first, cluster_labels_range_last);

    auto samples_separation_scores = std::vector<ElementType>(n_samples);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // get the centroid associated to the current sample
        const std::size_t centroid_index = cluster_labels_range_first[sample_index];
        // the sum of distances from the current sample to other samples from different centroids
        auto sample_to_other_cluster_samples_distance_mean = std::vector<ElementType>(cluster_sizes.size());
        // iterate over all the other data samples (j) that dont belong to the same cluster
        for (std::size_t other_sample_index = 0; other_sample_index < n_samples; ++other_sample_index) {
            // get the centroid associated to the other sample
            const std::size_t other_centroid_index = cluster_labels_range_first[other_sample_index];
            // compute distance only when sample belongs to a different cluster
            if (centroid_index != other_centroid_index) {
                // accumulate the squared distances for the correct centroid index
                sample_to_other_cluster_samples_distance_mean[other_centroid_index] +=
                    auto_distance(samples_range_first + sample_index * n_features,
                                  samples_range_first + sample_index * n_features + n_features,
                                  samples_range_first + other_sample_index * n_features,
                                  samples_range_first + other_sample_index * n_features + n_features);
            }
        }
        // normalize each cluster mean distance sum by each cluster's number of samples
        // destination_i = source_i / n if n != 0 else infinity
        std::transform(sample_to_other_cluster_samples_distance_mean.begin(),
                       sample_to_other_cluster_samples_distance_mean.end(),
                       cluster_sizes.begin(),
                       sample_to_other_cluster_samples_distance_mean.begin(),
                       [](const auto& distance, const auto& cluster_size) {
                           // if cluster size is zero, set distance mean to infinity so that it doesnt
                           // get picked otherwise normalize by dividing normally
                           return (cluster_size ? distance / static_cast<ElementType>(cluster_size)
                                                : common::infinity<ElementType>());
                       });
        // set the current cluster index distance value to infinity so that it doesnt get chosen
        sample_to_other_cluster_samples_distance_mean[centroid_index] = common::infinity<ElementType>();
        // normalise the sum of the distances from the current sample to all the other samples in the same cluster
        samples_separation_scores[sample_index] = *std::min_element(
            sample_to_other_cluster_samples_distance_mean.begin(), sample_to_other_cluster_samples_distance_mean.end());
    }
    return samples_separation_scores;
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
 * @tparam SamplesIterator
 * @tparam ClusterLabelsIterator
 * @param samples_range_first
 * @param samples_range_last
 * @param n_features
 * @param cluster_labels_range_first
 * @param cluster_labels_range_last
 */
template <typename SamplesIterator, typename ClusterLabelsIterator>
auto silhouette(const SamplesIterator&       samples_range_first,
                const SamplesIterator&       samples_range_last,
                std::size_t                  n_features,
                const ClusterLabelsIterator& cluster_labels_range_first,
                const ClusterLabelsIterator& cluster_labels_range_last) {
    using ElementType = typename std::iterator_traits<SamplesIterator>::value_type;

    using ClusterLabelType = typename std::iterator_traits<ClusterLabelsIterator>::value_type;

    static_assert(std::is_trivial_v<ElementType>, "ElementType must be trivial.");

    static_assert(std::is_integral<ClusterLabelType>::value, "Cluster labels must be integer.");

    const auto n_samples = get_n_samples(samples_range_first, samples_range_last, n_features);

    const auto cohesion_values = cohesion(/**/ samples_range_first,
                                          /**/ samples_range_last,
                                          /**/ n_features,
                                          /**/ cluster_labels_range_first,
                                          /**/ cluster_labels_range_last);

    const auto separation_values = separation(/**/ samples_range_first,
                                              /**/ samples_range_last,
                                              /**/ n_features,
                                              /**/ cluster_labels_range_first,
                                              /**/ cluster_labels_range_last);

    auto samples_silhouette_scores = std::vector<ElementType>(n_samples);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        const auto coh = cohesion_values[sample_index];
        const auto sep = separation_values[sample_index];

        if (coh < sep) {
            samples_silhouette_scores[sample_index] = static_cast<ElementType>(1) - coh / sep;

        } else if (coh > sep) {
            samples_silhouette_scores[sample_index] = sep / coh - static_cast<ElementType>(1);

        } else {
            samples_silhouette_scores[sample_index] = 0;
        }
    }
    return samples_silhouette_scores;
}

template <typename SamplesIterator>
auto get_average_silhouette(const SamplesIterator& silhouette_range_first, const SamplesIterator& silhouette_range_last)
    -> typename std::iterator_traits<SamplesIterator>::value_type {
    using ElementType = typename std::iterator_traits<SamplesIterator>::value_type;

    const auto n_elements = std::distance(silhouette_range_first, silhouette_range_last);

    return std::accumulate(silhouette_range_first, silhouette_range_last, static_cast<ElementType>(0)) / n_elements;
}

}  // namespace ffcl::common::math::heuristics
