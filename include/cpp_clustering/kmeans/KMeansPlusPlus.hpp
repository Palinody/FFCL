#pragma once

#include "cpp_clustering/common/Utils.hpp"
#include "cpp_clustering/heuristics/Heuristics.hpp"
#include "cpp_clustering/kmeans/KMeansUtils.hpp"
#include "cpp_clustering/math/random/Distributions.hpp"
#include "cpp_clustering/math/random/VosesAliasMethod.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace cpp_clustering::kmeansplusplus {

/**
 * @brief Initializes the first centroid randomly and then weights subsequent centroids based on the distance to all
 * previous centroids, has the advantage of being more likely to find a better overall solution. By ensuring that each
 * subsequent centroid is well-separated from all previous centroids, this version of kmeans++ can help to avoid local
 * minima and improve the quality of the final clustering result.
 *
 * However, the disadvantage of this approach is that it can be computationally expensive, especially for larger
 * datasets or higher-dimensional data. Since each new centroid must be evaluated with respect to all previous
 * centroids, the algorithm may slow down significantly as the number of clusters increases or the dataset size grows.
 *
 * @tparam IteratorFloat
 * @param data_first
 * @param data_last
 * @param n_centroids
 * @param n_features
 * @return std::vector<typename IteratorFloat::value_type>
 */
template <typename IteratorFloat>
std::vector<typename IteratorFloat::value_type> make_centroids(const IteratorFloat& data_first,
                                                               const IteratorFloat& data_last,
                                                               std::size_t          n_centroids,
                                                               std::size_t          n_features) {
    static_assert(std::is_floating_point<typename IteratorFloat::value_type>::value,
                  "Data should be a floating point type.");

    auto centroids = common::utils::select_random_sample(data_first, data_last, n_features);

    for (std::size_t centroid_index = 1; centroid_index < n_centroids; ++centroid_index) {
        // recompute the distances from each sample to its closest centroid
        auto nearest_centroid_distances =
            kmeans::utils::samples_to_nearest_centroid_distances(data_first, data_last, n_features, centroids);
        // use these distances as weighted probabilities
        auto alias_method = math::random::VosesAliasMethod(
            common::utils::to_type<double>(nearest_centroid_distances.begin(), nearest_centroid_distances.end()));

        const auto random_index = alias_method();

        centroids.insert(centroids.end(),
                         data_first + random_index * n_features,
                         data_first + random_index * n_features + n_features);
    }
    return centroids;
}

/**
 * @brief The second version of kmeans++, which weights subsequent centroids based only on the distance to the previous
 * centroid, can be faster since it only requires evaluating the distance to a single previous centroid for each new
 * centroid.
 *
 * However, this approach may be more prone to getting stuck in local minima, since it does not take into
 * account the overall distribution of centroids.
 *
 * @tparam IteratorFloat
 * @param data_first
 * @param data_last
 * @param n_centroids
 * @param n_features
 * @return std::vector<typename IteratorFloat::value_type>
 */
template <typename IteratorFloat>
std::vector<typename IteratorFloat::value_type> make_centroids_from_previous_centroid(const IteratorFloat& data_first,
                                                                                      const IteratorFloat& data_last,
                                                                                      std::size_t          n_centroids,
                                                                                      std::size_t          n_features) {
    static_assert(std::is_floating_point<typename IteratorFloat::value_type>::value,
                  "Data should be a floating point type.");

    auto previous_centroid = common::utils::select_random_sample(data_first, data_last, n_features);

    auto centroids = previous_centroid;

    for (std::size_t centroid_index = 1; centroid_index < n_centroids; ++centroid_index) {
        // recompute the distances from each sample to the previous centroid
        auto previous_centroid_distances =
            kmeans::utils::samples_to_nearest_centroid_distances(data_first, data_last, n_features, previous_centroid);
        // use these distances as weighted probabilities
        auto alias_method = math::random::VosesAliasMethod(
            common::utils::to_type<double>(previous_centroid_distances.begin(), previous_centroid_distances.end()));

        const auto random_index = alias_method();

        std::copy(data_first + random_index * n_features,
                  data_first + random_index * n_features + n_features,
                  previous_centroid.begin());

        centroids.insert(centroids.end(), previous_centroid.begin(), previous_centroid.end());
    }
    return centroids;
}

}  // namespace cpp_clustering::kmeansplusplus