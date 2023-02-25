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

}  // namespace cpp_clustering::kmeansplusplus