#pragma once

#include "cpp_clustering/common/Utils.hpp"
#include "cpp_clustering/containers/LowerTriangleMatrix.hpp"
#include "cpp_clustering/heuristics/Heuristics.hpp"
#include "cpp_clustering/math/random/Distributions.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

namespace pam::utils {

/**
 * @brief division specific to PAM. Returns zero if the input denominator in an int and is zero
 *
 * @tparam T: the type of the numerator
 * @tparam U: the type of the denominator
 * @param a: numerator value
 * @param b: denominator value
 * @return auto: -> decltype(a / b)
 */
template <typename T, typename U>
auto division(const T& a, const U& b) {
    if constexpr (std::is_integral_v<U>) {
        if (!b) {
            return 0;
        }
    }
    return a / b;
};

template <typename Iterator>
std::vector<std::size_t> samples_to_nearest_medoid_indices(const Iterator&                 samples_first,
                                                           const Iterator&                 samples_last,
                                                           std::size_t                     n_features,
                                                           const std::vector<std::size_t>& medoids_indices) {
    using DataType = typename Iterator::value_type;

    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }
    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

    // the vector that will contain the distances from each sample to the nearest medoid
    auto nearest_medoid_indices = std::vector<std::size_t>(n_samples);

    auto compute_distance = [&](std::size_t left_idx, std::size_t right_idx) -> DataType {
        return cpp_clustering::heuristic::heuristic(
            /*first medoid begin=*/samples_first + left_idx * n_features,
            /*first medoid end=*/samples_first + left_idx * n_features + n_features,
            /*current sample begin=*/samples_first + right_idx * n_features);
    };
    // iterate over all the samples
    for (std::size_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        DataType    first_min_distance = std::numeric_limits<DataType>::max();
        std::size_t first_min_index    = 0;

        for (std::size_t idx = 0; idx < medoids_indices.size(); ++idx) {
            const auto nearest_candidate = compute_distance(medoids_indices[idx], sample_idx);

            if (nearest_candidate < first_min_distance) {
                first_min_distance = nearest_candidate;
                first_min_index    = idx;
            }
        }
        // assign the current sample with its nth nearest medoid distance
        nearest_medoid_indices[sample_idx] = first_min_index;
    }
    return nearest_medoid_indices;
}

template <typename Iterator>
std::vector<std::size_t> samples_to_nearest_medoid_indices(
    const cpp_clustering::containers::LowerTriangleMatrix<Iterator>& pairwise_distance_matrix,
    const std::vector<std::size_t>&                                  medoids_indices) {
    using DataType = typename Iterator::value_type;

    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }
    const std::size_t n_samples = pairwise_distance_matrix.n_samples();

    // the vector that will contain the distances from each sample to the nearest medoid
    auto nearest_medoid_indices = std::vector<std::size_t>(n_samples);

    for (std::size_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        DataType    first_min_distance = std::numeric_limits<DataType>::max();
        std::size_t first_min_index    = 0;

        for (std::size_t idx = 0; idx < medoids_indices.size(); ++idx) {
            const auto nearest_candidate = pairwise_distance_matrix(medoids_indices[idx], sample_idx);

            if (nearest_candidate < first_min_distance) {
                first_min_distance = nearest_candidate;
                first_min_index    = idx;
            }
        }
        // assign the current sample with its nth nearest medoid distance
        nearest_medoid_indices[sample_idx] = first_min_index;
    }
    return nearest_medoid_indices;
}

template <typename Iterator>
std::vector<std::size_t> samples_to_second_nearest_medoid_indices(const Iterator&                 samples_first,
                                                                  const Iterator&                 samples_last,
                                                                  std::size_t                     n_features,
                                                                  const std::vector<std::size_t>& medoids_indices) {
    using DataType = typename Iterator::value_type;

    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }
    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

    // the vector that will contain the distances from each sample to the nearest medoid
    auto second_nearest_medoid_indices = std::vector<std::size_t>(n_samples);

    auto compute_distance = [&](std::size_t left_idx, std::size_t right_idx) -> DataType {
        return cpp_clustering::heuristic::heuristic(
            /*first medoid begin=*/samples_first + left_idx * n_features,
            /*first medoid end=*/samples_first + left_idx * n_features + n_features,
            /*current sample begin=*/samples_first + right_idx * n_features);
    };
    // iterate over all the samples
    for (std::size_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        DataType    first_min_distance  = std::numeric_limits<DataType>::max();
        DataType    second_min_distance = std::numeric_limits<DataType>::max();
        std::size_t first_min_index     = 0;
        std::size_t second_min_index    = 0;

        for (std::size_t idx = 0; idx < medoids_indices.size(); ++idx) {
            const auto second_nearest_candidate = compute_distance(medoids_indices[idx], sample_idx);

            if (second_nearest_candidate < first_min_distance) {
                second_min_distance = first_min_distance;
                second_min_index    = first_min_index;
                first_min_distance  = second_nearest_candidate;
                first_min_index     = idx;

            } else if (second_nearest_candidate < second_min_distance &&
                       second_nearest_candidate != first_min_distance) {
                second_min_distance = second_nearest_candidate;
                second_min_index    = idx;
            }
        }
        // assign the current sample with its nth nearest medoid distance
        second_nearest_medoid_indices[sample_idx] = second_min_index;
    }
    return second_nearest_medoid_indices;
}

template <typename Iterator>
std::vector<std::size_t> samples_to_second_nearest_medoid_indices(
    const cpp_clustering::containers::LowerTriangleMatrix<Iterator>& pairwise_distance_matrix,
    const std::vector<std::size_t>&                                  medoids_indices) {
    using DataType = typename Iterator::value_type;

    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }
    const std::size_t n_samples = pairwise_distance_matrix.n_samples();

    // the vector that will contain the distances from each sample to the nearest medoid
    auto second_nearest_medoid_indices = std::vector<std::size_t>(n_samples);

    // iterate over all the samples
    for (std::size_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        DataType    first_min_distance  = std::numeric_limits<DataType>::max();
        DataType    second_min_distance = std::numeric_limits<DataType>::max();
        std::size_t first_min_index     = 0;
        std::size_t second_min_index    = 0;

        for (std::size_t idx = 0; idx < medoids_indices.size(); ++idx) {
            const auto second_nearest_candidate = pairwise_distance_matrix(medoids_indices[idx], sample_idx);

            if (second_nearest_candidate < first_min_distance) {
                second_min_distance = first_min_distance;
                second_min_index    = first_min_index;
                first_min_distance  = second_nearest_candidate;
                first_min_index     = idx;

            } else if (second_nearest_candidate < second_min_distance &&
                       second_nearest_candidate != first_min_distance) {
                second_min_distance = second_nearest_candidate;
                second_min_index    = idx;
            }
        }
        // assign the current sample with its nth nearest medoid distance
        second_nearest_medoid_indices[sample_idx] = second_min_index;
    }
    return second_nearest_medoid_indices;
}

template <typename Iterator>
std::vector<std::size_t> samples_to_third_nearest_medoid_indices(const Iterator&                 samples_first,
                                                                 const Iterator&                 samples_last,
                                                                 std::size_t                     n_features,
                                                                 const std::vector<std::size_t>& medoids_indices) {
    using DataType = typename Iterator::value_type;

    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }
    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

    // the vector that will contain the distances from each sample to the nearest medoid
    auto third_nearest_medoid_indices = std::vector<std::size_t>(n_samples);

    auto compute_distance = [&](std::size_t left_idx, std::size_t right_idx) -> DataType {
        return cpp_clustering::heuristic::heuristic(
            /*first medoid begin=*/samples_first + left_idx * n_features,
            /*first medoid end=*/samples_first + left_idx * n_features + n_features,
            /*current sample begin=*/samples_first + right_idx * n_features);
    };
    // iterate over all the samples
    for (std::size_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        DataType    first_min_distance  = std::numeric_limits<DataType>::max();
        DataType    second_min_distance = std::numeric_limits<DataType>::max();
        DataType    third_min_distance  = std::numeric_limits<DataType>::max();
        std::size_t first_min_index     = 0;
        std::size_t second_min_index    = 0;
        std::size_t third_min_index     = 0;

        for (std::size_t idx = 0; idx < medoids_indices.size(); ++idx) {
            const auto third_nearest_candidate = compute_distance(medoids_indices[idx], sample_idx);

            if (third_nearest_candidate < first_min_distance) {
                third_min_distance  = second_min_distance;
                third_min_index     = second_min_index;
                second_min_distance = first_min_distance;
                second_min_index    = first_min_index;
                first_min_distance  = third_nearest_candidate;
                first_min_index     = idx;

            } else if (third_nearest_candidate < second_min_distance && third_nearest_candidate != first_min_distance) {
                third_min_distance  = second_min_distance;
                third_min_index     = second_min_index;
                second_min_distance = third_nearest_candidate;
                second_min_index    = idx;

            } else if (third_nearest_candidate < third_min_distance && third_nearest_candidate != first_min_distance &&
                       third_nearest_candidate != second_min_distance) {
                third_min_distance = third_nearest_candidate;
                third_min_index    = idx;
            }
        }
        // assign the current sample with its nth nearest medoid distance
        third_nearest_medoid_indices[sample_idx] = third_min_index;
    }
    return third_nearest_medoid_indices;
}

template <typename Iterator>
std::vector<std::size_t> samples_to_third_nearest_medoid_indices(
    const cpp_clustering::containers::LowerTriangleMatrix<Iterator>& pairwise_distance_matrix,
    const std::vector<std::size_t>&                                  medoids_indices) {
    using DataType = typename Iterator::value_type;

    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }
    const std::size_t n_samples = pairwise_distance_matrix.n_samples();

    // the vector that will contain the distances from each sample to the nearest medoid
    auto third_nearest_medoid_indices = std::vector<std::size_t>(n_samples);

    // iterate over all the samples
    for (std::size_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        DataType    first_min_distance  = std::numeric_limits<DataType>::max();
        DataType    second_min_distance = std::numeric_limits<DataType>::max();
        DataType    third_min_distance  = std::numeric_limits<DataType>::max();
        std::size_t first_min_index     = 0;
        std::size_t second_min_index    = 0;
        std::size_t third_min_index     = 0;

        for (std::size_t idx = 0; idx < medoids_indices.size(); ++idx) {
            const auto third_nearest_candidate = pairwise_distance_matrix(medoids_indices[idx], sample_idx);

            if (third_nearest_candidate < first_min_distance) {
                third_min_distance  = second_min_distance;
                third_min_index     = second_min_index;
                second_min_distance = first_min_distance;
                second_min_index    = first_min_index;
                first_min_distance  = third_nearest_candidate;
                first_min_index     = idx;

            } else if (third_nearest_candidate < second_min_distance && third_nearest_candidate != first_min_distance) {
                third_min_distance  = second_min_distance;
                third_min_index     = second_min_index;
                second_min_distance = third_nearest_candidate;
                second_min_index    = idx;

            } else if (third_nearest_candidate < third_min_distance && third_nearest_candidate != first_min_distance &&
                       third_nearest_candidate != second_min_distance) {
                third_min_distance = third_nearest_candidate;
                third_min_index    = idx;
            }
        }
        // assign the current sample with its nth nearest medoid distance
        third_nearest_medoid_indices[sample_idx] = third_min_index;
    }
    return third_nearest_medoid_indices;
}

template <typename Iterator>
std::vector<std::size_t> samples_to_nth_nearest_medoid_indices(const Iterator&                 samples_first,
                                                               const Iterator&                 samples_last,
                                                               std::size_t                     n_features,
                                                               const std::vector<std::size_t>& medoids_indices,
                                                               std::size_t                     nth_closest = 1) {
    if (nth_closest == 0 || nth_closest > medoids_indices.size()) {
        throw std::invalid_argument("nth_closest value should be inside range ]0, n_medoids].");
    }
    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }

    if (nth_closest == 1) {
        return samples_to_nearest_medoid_indices(samples_first, samples_last, n_features, medoids_indices);

    } else if (nth_closest == 2) {
        return samples_to_second_nearest_medoid_indices(samples_first, samples_last, n_features, medoids_indices);

    } else if (nth_closest == 3) {
        return samples_to_third_nearest_medoid_indices(samples_first, samples_last, n_features, medoids_indices);

    } else {
        throw std::invalid_argument("Invalid nth_closest parameter.");
    }
}

template <typename Iterator>
std::vector<std::size_t> samples_to_nth_nearest_medoid_indices(
    const cpp_clustering::containers::LowerTriangleMatrix<Iterator>& pairwise_distance_matrix,
    const std::vector<std::size_t>&                                  medoids_indices,
    std::size_t                                                      nth_closest = 1) {
    if (nth_closest == 0 || nth_closest > medoids_indices.size()) {
        throw std::invalid_argument("nth_closest value should be inside range ]0, n_medoids].");
    }
    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }

    if (nth_closest == 1) {
        return samples_to_nearest_medoid_indices(pairwise_distance_matrix, medoids_indices);

    } else if (nth_closest == 2) {
        return samples_to_second_nearest_medoid_indices(pairwise_distance_matrix, medoids_indices);

    } else if (nth_closest == 3) {
        return samples_to_third_nearest_medoid_indices(pairwise_distance_matrix, medoids_indices);

    } else {
        throw std::invalid_argument("Invalid nth_closest parameter.");
    }
}

template <typename Iterator>
std::vector<typename Iterator::value_type> samples_to_nearest_medoid_distances(
    const Iterator&                 samples_first,
    const Iterator&                 samples_last,
    std::size_t                     n_features,
    const std::vector<std::size_t>& medoids_indices) {
    using DataType = typename Iterator::value_type;

    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }
    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

    // the vector that will contain the distances from each sample to the nearest medoid
    auto nearest_medoid_distances = std::vector<DataType>(n_samples);

    auto compute_distance = [&](std::size_t left_idx, std::size_t right_idx) -> DataType {
        return cpp_clustering::heuristic::heuristic(
            /*first medoid begin=*/samples_first + left_idx * n_features,
            /*first medoid end=*/samples_first + left_idx * n_features + n_features,
            /*current sample begin=*/samples_first + right_idx * n_features);
    };
    // iterate over all the samples
    for (std::size_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        auto first_min_distance = std::numeric_limits<DataType>::max();
        // iterate over the medoids indices
        for (const auto& medoid : medoids_indices) {
            const auto nearest_candidate = compute_distance(medoid, sample_idx);

            if (nearest_candidate < first_min_distance) {
                first_min_distance = nearest_candidate;
            }
        }
        // assign the current sample with its nth nearest medoid distance
        nearest_medoid_distances[sample_idx] = first_min_distance;
    }
    return nearest_medoid_distances;
}

template <typename Iterator>
std::vector<typename Iterator::value_type> samples_to_nearest_medoid_distances(
    const cpp_clustering::containers::LowerTriangleMatrix<Iterator>& pairwise_distance_matrix,
    const std::vector<std::size_t>&                                  medoids_indices) {
    using DataType = typename Iterator::value_type;

    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }
    const std::size_t n_samples = pairwise_distance_matrix.n_samples();

    // the vector that will contain the distances from each sample to the nearest medoid
    auto nearest_medoid_distances = std::vector<DataType>(n_samples);

    // iterate over all the samples
    for (std::size_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        auto first_min_distance = std::numeric_limits<DataType>::max();
        // iterate over the medoids indices
        for (const auto& medoid : medoids_indices) {
            const auto nearest_candidate = pairwise_distance_matrix(medoid, sample_idx);

            if (nearest_candidate < first_min_distance) {
                first_min_distance = nearest_candidate;
            }
        }
        // assign the current sample with its nth nearest medoid distance
        nearest_medoid_distances[sample_idx] = first_min_distance;
    }
    return nearest_medoid_distances;
}

template <typename Iterator>
std::vector<typename Iterator::value_type> samples_to_second_nearest_medoid_distances(
    const Iterator&                 samples_first,
    const Iterator&                 samples_last,
    std::size_t                     n_features,
    const std::vector<std::size_t>& medoids_indices) {
    using DataType = typename Iterator::value_type;

    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }
    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

    // the vector that will contain the distances from each sample to the nearest medoid
    auto second_nearest_medoid_distances = std::vector<DataType>(n_samples);

    auto compute_distance = [&](std::size_t left_idx, std::size_t right_idx) -> DataType {
        return cpp_clustering::heuristic::heuristic(
            /*first medoid begin=*/samples_first + left_idx * n_features,
            /*first medoid end=*/samples_first + left_idx * n_features + n_features,
            /*current sample begin=*/samples_first + right_idx * n_features);
    };
    // iterate over all the samples
    for (std::size_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        auto first_min_distance  = std::numeric_limits<DataType>::max();
        auto second_min_distance = std::numeric_limits<DataType>::max();
        // iterate over the medoids indices
        for (const auto& medoid : medoids_indices) {
            const auto second_nearest_candidate = compute_distance(medoid, sample_idx);

            if (second_nearest_candidate < first_min_distance) {
                second_min_distance = first_min_distance;
                first_min_distance  = second_nearest_candidate;

            } else if (second_nearest_candidate < second_min_distance &&
                       second_nearest_candidate != first_min_distance) {
                second_min_distance = second_nearest_candidate;
            }
        }
        // assign the current sample with its nth nearest medoid distance
        second_nearest_medoid_distances[sample_idx] = second_min_distance;
    }
    return second_nearest_medoid_distances;
}

template <typename Iterator>
std::vector<typename Iterator::value_type> samples_to_second_nearest_medoid_distances(
    const cpp_clustering::containers::LowerTriangleMatrix<Iterator>& pairwise_distance_matrix,
    const std::vector<std::size_t>&                                  medoids_indices) {
    using DataType = typename Iterator::value_type;

    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }
    const std::size_t n_samples = pairwise_distance_matrix.n_samples();

    // the vector that will contain the distances from each sample to the nearest medoid
    auto second_nearest_medoid_distances = std::vector<DataType>(n_samples);

    // iterate over all the samples
    for (std::size_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        auto first_min_distance  = std::numeric_limits<DataType>::max();
        auto second_min_distance = std::numeric_limits<DataType>::max();
        // iterate over the medoids indices
        for (const auto& medoid : medoids_indices) {
            const auto second_nearest_candidate = pairwise_distance_matrix(medoid, sample_idx);

            if (second_nearest_candidate < first_min_distance) {
                second_min_distance = first_min_distance;
                first_min_distance  = second_nearest_candidate;

            } else if (second_nearest_candidate < second_min_distance &&
                       second_nearest_candidate != first_min_distance) {
                second_min_distance = second_nearest_candidate;
            }
        }
        // assign the current sample with its nth nearest medoid distance
        second_nearest_medoid_distances[sample_idx] = second_min_distance;
    }
    return second_nearest_medoid_distances;
}

template <typename Iterator>
std::vector<typename Iterator::value_type> samples_to_third_nearest_medoid_distances(
    const Iterator&                 samples_first,
    const Iterator&                 samples_last,
    std::size_t                     n_features,
    const std::vector<std::size_t>& medoids_indices) {
    using DataType = typename Iterator::value_type;

    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }
    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

    // the vector that will contain the distances from each sample to the nearest medoid
    auto third_nearest_medoid_distances = std::vector<DataType>(n_samples);

    auto compute_distance = [&](std::size_t left_idx, std::size_t right_idx) -> DataType {
        return cpp_clustering::heuristic::heuristic(
            /*first medoid begin=*/samples_first + left_idx * n_features,
            /*first medoid end=*/samples_first + left_idx * n_features + n_features,
            /*current sample begin=*/samples_first + right_idx * n_features);
    };
    // iterate over all the samples
    for (std::size_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        auto first_min_distance  = std::numeric_limits<DataType>::max();
        auto second_min_distance = std::numeric_limits<DataType>::max();
        auto third_min_distance  = std::numeric_limits<DataType>::max();
        // iterate over the medoids indices
        for (const auto& medoid : medoids_indices) {
            const auto third_nearest_candidate = compute_distance(medoid, sample_idx);

            if (third_nearest_candidate < first_min_distance) {
                third_min_distance  = second_min_distance;
                second_min_distance = first_min_distance;
                first_min_distance  = third_nearest_candidate;

            } else if (third_nearest_candidate < second_min_distance && third_nearest_candidate != first_min_distance) {
                third_min_distance  = second_min_distance;
                second_min_distance = third_nearest_candidate;

            } else if (third_nearest_candidate < third_min_distance && third_nearest_candidate != first_min_distance &&
                       third_nearest_candidate != second_min_distance) {
                third_min_distance = third_nearest_candidate;
            }
        }
        // assign the current sample with its nth nearest medoid distance
        third_nearest_medoid_distances[sample_idx] = third_min_distance;
    }
    return third_nearest_medoid_distances;
}

template <typename Iterator>
std::vector<typename Iterator::value_type> samples_to_third_nearest_medoid_distances(
    const cpp_clustering::containers::LowerTriangleMatrix<Iterator>& pairwise_distance_matrix,
    const std::vector<std::size_t>&                                  medoids_indices) {
    using DataType = typename Iterator::value_type;

    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }
    const std::size_t n_samples = pairwise_distance_matrix.n_samples();

    // the vector that will contain the distances from each sample to the nearest medoid
    auto third_nearest_medoid_distances = std::vector<DataType>(n_samples);

    // iterate over all the samples
    for (std::size_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        auto first_min_distance  = std::numeric_limits<DataType>::max();
        auto second_min_distance = std::numeric_limits<DataType>::max();
        auto third_min_distance  = std::numeric_limits<DataType>::max();
        // iterate over the medoids indices
        for (const auto& medoid : medoids_indices) {
            const auto third_nearest_candidate = pairwise_distance_matrix(medoid, sample_idx);

            if (third_nearest_candidate < first_min_distance) {
                third_min_distance  = second_min_distance;
                second_min_distance = first_min_distance;
                first_min_distance  = third_nearest_candidate;

            } else if (third_nearest_candidate < second_min_distance && third_nearest_candidate != first_min_distance) {
                third_min_distance  = second_min_distance;
                second_min_distance = third_nearest_candidate;

            } else if (third_nearest_candidate < third_min_distance && third_nearest_candidate != first_min_distance &&
                       third_nearest_candidate != second_min_distance) {
                third_min_distance = third_nearest_candidate;
            }
        }
        // assign the current sample with its nth nearest medoid distance
        third_nearest_medoid_distances[sample_idx] = third_min_distance;
    }
    return third_nearest_medoid_distances;
}

template <typename Iterator>
std::vector<typename Iterator::value_type> samples_to_nth_nearest_medoid_distances(
    const Iterator&                 samples_first,
    const Iterator&                 samples_last,
    std::size_t                     n_features,
    const std::vector<std::size_t>& medoids_indices,
    std::size_t                     nth_closest = 1) {
    if (nth_closest == 0 || nth_closest > medoids_indices.size()) {
        throw std::invalid_argument("nth_closest value should be inside range ]0, n_medoids].");
    }
    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }

    if (nth_closest == 1) {
        return samples_to_nearest_medoid_distances(samples_first, samples_last, n_features, medoids_indices);

    } else if (nth_closest == 2) {
        return samples_to_second_nearest_medoid_distances(samples_first, samples_last, n_features, medoids_indices);

    } else if (nth_closest == 3) {
        return samples_to_third_nearest_medoid_distances(samples_first, samples_last, n_features, medoids_indices);

    } else {
        throw std::invalid_argument("Invalid nth_closest parameter.");
    }
}

template <typename Iterator>
std::vector<typename Iterator::value_type> samples_to_nth_nearest_medoid_distances(
    const cpp_clustering::containers::LowerTriangleMatrix<Iterator>& pairwise_distance_matrix,
    const std::vector<std::size_t>&                                  medoids_indices,
    std::size_t                                                      nth_closest = 1) {
    if (nth_closest == 0 || nth_closest > medoids_indices.size()) {
        throw std::invalid_argument("nth_closest value should be inside range ]0, n_medoids].");
    }
    if (medoids_indices.empty()) {
        throw std::invalid_argument("Medoids indices vector shouldn't be empty.");
    }

    if (nth_closest == 1) {
        return samples_to_nearest_medoid_distances(pairwise_distance_matrix, medoids_indices);

    } else if (nth_closest == 2) {
        return samples_to_second_nearest_medoid_distances(pairwise_distance_matrix, medoids_indices);

    } else if (nth_closest == 3) {
        return samples_to_third_nearest_medoid_distances(pairwise_distance_matrix, medoids_indices);

    } else {
        throw std::invalid_argument("Invalid nth_closest parameter.");
    }
}

template <typename DataType>
std::vector<DataType> compute_losses_with_closest_medoid_removal(
    const std::vector<std::size_t>& nearest_medoid_indices,
    const std::vector<DataType>&    nearest_medoid_distances,
    const std::vector<DataType>&    second_nearest_medoid_distances,
    std::size_t                     n_medoids) {
    const std::size_t n_samples = nearest_medoid_indices.size();

    // The positive loss of removing medoid m_i and assigning all of its members to the next best alternative
    auto delta_td_mi = std::vector<DataType>(n_medoids);

    for (std::size_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        // get the index of the nearest medoid w.r.t. the current sample
        const auto nearest_medoid_idx = nearest_medoid_indices[sample_idx];
        // accumulate the variation in total deviation for the correct medoid index
        delta_td_mi[nearest_medoid_idx] +=
            second_nearest_medoid_distances[sample_idx] - nearest_medoid_distances[sample_idx];
    }
    return delta_td_mi;
}

template <typename DataType>
std::vector<DataType> compute_losses_with_silhouette_medoid_removal(
    const std::vector<std::size_t>& nearest_medoid_indices,
    const std::vector<std::size_t>& second_nearest_medoid_indices,
    const std::vector<DataType>&    nearest_medoid_distances,
    const std::vector<DataType>&    second_nearest_medoid_distances,
    const std::vector<DataType>&    third_nearest_medoid_distances,
    std::size_t                     n_medoids) {
    const std::size_t n_samples = nearest_medoid_indices.size();

    // The positive loss of removing medoid m_i and assigning all of its members to the next best alternative
    auto delta_td_mi = std::vector<DataType>(n_medoids);

    for (std::size_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        // get the index of the nearest medoid w.r.t. the current sample
        const auto index_1 = nearest_medoid_indices[sample_idx];
        // get the index of the second nearest medoid w.r.t. the current sample
        const auto index_2 = second_nearest_medoid_indices[sample_idx];
        // distance from the current sample to its nearest medoid
        const auto distance_1 = nearest_medoid_distances[sample_idx];
        // distance from the current sample to its second nearest medoid
        const auto distance_2 = second_nearest_medoid_distances[sample_idx];
        // distance from the current sample to its third nearest medoid
        const auto distance_3 = third_nearest_medoid_distances[sample_idx];
        // accumulate the variation in total deviation for the correct medoid index
        delta_td_mi[index_1] += division(distance_1, distance_2) - division(distance_2, distance_3);
        delta_td_mi[index_2] += division(distance_1, distance_2) - division(distance_1, distance_3);
    }
    return delta_td_mi;
}

template <typename Iterator>
std::pair<typename Iterator::value_type, std::size_t> first_medoid_td_index_pair(const Iterator& data_first,
                                                                                 const Iterator& data_last,
                                                                                 std::size_t     n_features) {
    using DataType = typename Iterator::value_type;

    const std::size_t n_samples = common::utils::get_n_samples(data_first, data_last, n_features);

    std::size_t selected_medoid = 0;
    auto        total_deviation = common::utils::infinity<DataType>();
    // choose the first medoid
    for (std::size_t medoid_candidate_idx = 0; medoid_candidate_idx < n_samples; ++medoid_candidate_idx) {
        // total deviation accumulator w.r.t. current candidate medoid and all the other points
        DataType loss_acc = 0;
        for (std::size_t other_sample_idx = 0; other_sample_idx < n_samples; ++other_sample_idx) {
            // the following should be done if other_sample_idx != medoid_candidate_idx
            // but the distance would be 0 anyway with dist(other_sample, medoid_candidate)
            loss_acc += cpp_clustering::heuristic::heuristic(
                /*first sample begin=*/data_first + medoid_candidate_idx * n_features,
                /*first sample end=*/data_first + medoid_candidate_idx * n_features + n_features,
                /*other sample begin=*/data_first + other_sample_idx * n_features);
        }
        // if the candidate total deviation is lower than the current total deviation
        if (loss_acc < total_deviation) {
            // update the current total deviation
            total_deviation = loss_acc;
            // save the chosen medoid index
            selected_medoid = medoid_candidate_idx;
        }
    }
    return {total_deviation, selected_medoid};
}

template <typename Iterator>
std::pair<typename Iterator::value_type, std::size_t> first_medoid_td_index_pair(
    const cpp_clustering::containers::LowerTriangleMatrix<Iterator>& pairwise_distance_matrix) {
    using DataType = typename Iterator::value_type;

    const std::size_t n_samples = pairwise_distance_matrix.n_samples();

    std::size_t selected_medoid = 0;
    auto        total_deviation = common::utils::infinity<DataType>();
    // choose the first medoid
    for (std::size_t medoid_candidate_idx = 0; medoid_candidate_idx < n_samples; ++medoid_candidate_idx) {
        // total deviation accumulator w.r.t. current candidate medoid and all the other points
        DataType loss_acc = 0;
        for (std::size_t other_sample_idx = 0; other_sample_idx < n_samples; ++other_sample_idx) {
            // the following should be done if other_sample_idx != medoid_candidate_idx
            // but the distance would be 0 anyway with dist(other_sample, medoid_candidate)
            loss_acc += pairwise_distance_matrix(medoid_candidate_idx, other_sample_idx);
        }
        // if the candidate total deviation is lower than the current total deviation
        if (loss_acc < total_deviation) {
            // update the current total deviation
            total_deviation = loss_acc;
            // save the chosen medoid index
            selected_medoid = medoid_candidate_idx;
        }
    }
    return {total_deviation, selected_medoid};
}

}  // namespace pam::utils