#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"
#include "ffcl/common/math/random/Distributions.hpp"
#include "ffcl/common/math/random/Sampling.hpp"
#include "ffcl/common/math/random/VosesAliasMethod.hpp"
#include "ffcl/kmeans/KMeansUtils.hpp"

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
 * @param samples_range_first
 * @param samples_range_last
 * @param n_features
 * @param n_centroids
 * @return std::vector<typename std::iterator_traits<SamplesIterator>::value_type>
 */
template <typename SamplesIterator>
auto make_centroids(const SamplesIterator& samples_range_first,
                    const SamplesIterator& samples_range_last,
                    std::size_t            n_features,
                    std::size_t            n_centroids)
    -> std::vector<typename std::iterator_traits<SamplesIterator>::value_type> {
    static_assert(std::is_trivial_v<typename std::iterator_traits<SamplesIterator>::value_type>,
                  "Data must be trivial.");

    auto centroids = common::math::random::select_random_sample(samples_range_first, samples_range_last, n_features);

    for (std::size_t centroid_index = 1; centroid_index < n_centroids; ++centroid_index) {
        // recompute the distances from each sample to its closest centroid
        auto nearest_centroid_distances = kmeans::utils::samples_to_nearest_centroid_distances(
            samples_range_first, samples_range_last, n_features, centroids);
        // use these distances as weighted probabilities
        auto alias_method = common::math::random::VosesAliasMethod(nearest_centroid_distances);

        const auto random_index = alias_method();

        centroids.insert(centroids.end(),
                         samples_range_first + random_index * n_features,
                         samples_range_first + random_index * n_features + n_features);
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
 * @param samples_range_first
 * @param samples_range_last
 * @param n_features
 * @param n_centroids
 * @return std::vector<typename std::iterator_traits<SamplesIterator>::value_type>
 */
template <typename SamplesIterator>
auto make_centroids_from_previous_centroid(const SamplesIterator& samples_range_first,
                                           const SamplesIterator& samples_range_last,
                                           std::size_t            n_features,
                                           std::size_t            n_centroids)
    -> std::vector<typename std::iterator_traits<SamplesIterator>::value_type> {
    static_assert(std::is_trivial_v<typename std::iterator_traits<SamplesIterator>::value_type>,
                  "Data must be trivial.");

    auto previous_centroid =
        common::math::random::select_random_sample(samples_range_first, samples_range_last, n_features);

    auto centroids = previous_centroid;

    for (std::size_t centroid_index = 1; centroid_index < n_centroids; ++centroid_index) {
        // recompute the distances from each sample to the previous centroid
        auto previous_centroid_distances = kmeans::utils::samples_to_nearest_centroid_distances(
            samples_range_first, samples_range_last, n_features, previous_centroid);
        // use these distances as weighted probabilities
        auto alias_method = common::math::random::VosesAliasMethod(previous_centroid_distances);

        const auto random_index = alias_method();

        std::copy(samples_range_first + random_index * n_features,
                  samples_range_first + random_index * n_features + n_features,
                  previous_centroid.begin());

        centroids.insert(centroids.end(), previous_centroid.begin(), previous_centroid.end());
    }
    return centroids;
}

}  // namespace ffcl::kmeansplusplus