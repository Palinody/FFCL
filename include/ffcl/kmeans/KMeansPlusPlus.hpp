#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/kmeans/KMeansUtils.hpp"
#include "ffcl/math/heuristics/Distances.hpp"
#include "ffcl/math/random/Distributions.hpp"
#include "ffcl/math/random/Sampling.hpp"
#include "ffcl/math/random/VosesAliasMethod.hpp"

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

namespace ffcl::kmeansplusplus {

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
 * @tparam SamplesIterator
 * @param data_first
 * @param data_last
 * @param n_centroids
 * @param n_features
 * @return std::vector<typename SamplesIterator::value_type>
 */
template <typename SamplesIterator>
std::vector<typename SamplesIterator::value_type> make_centroids(const SamplesIterator& data_first,
                                                                 const SamplesIterator& data_last,
                                                                 std::size_t            n_centroids,
                                                                 std::size_t            n_features) {
    static_assert(std::is_floating_point<typename SamplesIterator::value_type>::value,
                  "Data should be a floating point type.");

    auto centroids = math::random::select_random_sample(data_first, data_last, n_features);

    for (std::size_t centroid_index = 1; centroid_index < n_centroids; ++centroid_index) {
        // recompute the distances from each sample to its closest centroid
        auto nearest_centroid_distances =
            kmeans::utils::samples_to_nearest_centroid_distances(data_first, data_last, n_features, centroids);
        // use these distances as weighted probabilities
        auto alias_method = math::random::VosesAliasMethod(nearest_centroid_distances);

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
 * @tparam SamplesIterator
 * @param data_first
 * @param data_last
 * @param n_centroids
 * @param n_features
 * @return std::vector<typename SamplesIterator::value_type>
 */
template <typename SamplesIterator>
std::vector<typename SamplesIterator::value_type> make_centroids_from_previous_centroid(
    const SamplesIterator& data_first,
    const SamplesIterator& data_last,
    std::size_t            n_centroids,
    std::size_t            n_features) {
    static_assert(std::is_floating_point<typename SamplesIterator::value_type>::value,
                  "Data should be a floating point type.");

    auto previous_centroid = math::random::select_random_sample(data_first, data_last, n_features);

    auto centroids = previous_centroid;

    for (std::size_t centroid_index = 1; centroid_index < n_centroids; ++centroid_index) {
        // recompute the distances from each sample to the previous centroid
        auto previous_centroid_distances =
            kmeans::utils::samples_to_nearest_centroid_distances(data_first, data_last, n_features, previous_centroid);
        // use these distances as weighted probabilities
        auto alias_method = math::random::VosesAliasMethod(previous_centroid_distances);

        const auto random_index = alias_method();

        std::copy(data_first + random_index * n_features,
                  data_first + random_index * n_features + n_features,
                  previous_centroid.begin());

        centroids.insert(centroids.end(), previous_centroid.begin(), previous_centroid.end());
    }
    return centroids;
}

}  // namespace ffcl::kmeansplusplus