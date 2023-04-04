#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/LowerTriangleMatrix.hpp"
#include "ffcl/math/heuristics/Distances.hpp"

#include <algorithm>
#include <vector>

namespace pam {

template <typename Iterator>
std::tuple<typename Iterator::value_type, std::vector<std::size_t>, std::vector<typename Iterator::value_type>>
build(const Iterator& samples_first, const Iterator& samples_last, std::size_t n_medoids, std::size_t n_features) {
    using DataType = typename Iterator::value_type;

    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

    // the first medoid chosen associated with the current total deviation cost
    auto [total_deviation, medoid_index_first] = first_medoid_td_index_pair(samples_first, samples_last, n_features);
    // Create and put the first chosen medoid index in the medoid indices vector
    auto medoids_indices = std::vector<std::size_t>(1, medoid_index_first);

    // initialize the distance from the first medoid to all samples
    auto samples_to_nearest_medoid_distance = samples_to_nth_nearest_medoid_distances(
        samples_first, samples_last, n_features, medoids_indices, /*nth_closest*/ 1);

    auto compute_distance = [&](std::size_t left_idx, std::size_t right_idx) -> DataType {
        return math::heuristics::auto_distance(
            /*first medoid begin=*/samples_first + left_idx * n_features,
            /*first medoid end=*/samples_first + left_idx * n_features + n_features,
            /*current sample begin=*/samples_first + right_idx * n_features);
    };

    // select the remaining medoids
    for (std::size_t medoid_index = 1; medoid_index < n_medoids; ++medoid_index) {
        // (∆TD*, x∗) ← (∞, null);
        auto selected_deviation_candidate = common::utils::infinity<DataType>();
        // the index of the next chosen medoid
        std::size_t selected_medoid_index = 0;
        // foreach x_c !∈ {m_1 , ..., m_i}
        for (std::size_t medoid_candidate_idx = 0; medoid_candidate_idx < n_samples; ++medoid_candidate_idx) {
            // execute only if the current candidate is not already selected as a medoid
            if (common::utils::is_element_not_in(
                    medoids_indices.begin(), medoids_indices.end(), medoid_candidate_idx)) {
                DataType loss_acc = 0;

                // foreach x_o !∈ {m_0, ..., m_i, x c}
                for (std::size_t other_sample_idx = 0; other_sample_idx < n_samples; ++other_sample_idx) {
                    if (medoid_candidate_idx != other_sample_idx &&
                        common::utils::is_element_not_in(
                            medoids_indices.begin(), medoids_indices.end(), other_sample_idx)) {
                        const auto candidate_to_other_distance =
                            compute_distance(medoid_candidate_idx, other_sample_idx);

                        if (candidate_to_other_distance < samples_to_nearest_medoid_distance[other_sample_idx]) {
                            // "-" to accumulate positive loss (acc of reduction of total deviation)
                            loss_acc -=
                                candidate_to_other_distance - samples_to_nearest_medoid_distance[other_sample_idx];
                        }
                    }
                }
                if (loss_acc < selected_deviation_candidate) {
                    selected_deviation_candidate = loss_acc;
                    selected_medoid_index        = medoid_candidate_idx;
                }
            }
        }
        medoids_indices.emplace_back(selected_medoid_index);
        total_deviation += selected_deviation_candidate;
        // Update the distances from each non medoid sample to its nearest medoid.
        // No need to take care about the medoids indices bc they are not swapped.
        auto samples_to_nearest_medoid_distance = samples_to_nth_nearest_medoid_distances(
            samples_first, samples_last, n_features, medoids_indices, /*nth_closest=*/1);
    }
    return {total_deviation, medoids_indices, samples_to_nearest_medoid_distance};
}

template <typename Iterator>
std::tuple<typename Iterator::value_type, std::vector<std::size_t>, std::vector<typename Iterator::value_type>> build(
    const ffcl::containers::LowerTriangleMatrix<Iterator>& pairwise_distance_matrix,
    std::size_t                                            n_medoids) {
    using DataType = typename Iterator::value_type;

    const std::size_t n_samples = pairwise_distance_matrix.n_samples();

    // the first medoid chosen associated with the current total deviation cost
    auto [total_deviation, medoid_index_first] = first_medoid_td_index_pair(pairwise_distance_matrix);
    // Create and put the first chosen medoid index in the medoid indices vector
    auto medoids_indices = std::vector<std::size_t>(1, medoid_index_first);

    // initialize the distance from the first medoid to all samples
    auto samples_to_nearest_medoid_distance =
        samples_to_nth_nearest_medoid_distances(pairwise_distance_matrix, medoids_indices, /*nth_closest*/ 1);

    // select the remaining medoids
    for (std::size_t medoid_index = 1; medoid_index < n_medoids; ++medoid_index) {
        // (∆TD*, x∗) ← (∞, null);
        auto selected_deviation_candidate = common::utils::infinity<DataType>();
        // the index of the next chosen medoid
        std::size_t selected_medoid_index = 0;
        // foreach x_c !∈ {m_1 , ..., m_i}
        for (std::size_t medoid_candidate_idx = 0; medoid_candidate_idx < n_samples; ++medoid_candidate_idx) {
            // execute only if the current candidate is not already selected as a medoid
            if (common::utils::is_element_not_in(
                    medoids_indices.begin(), medoids_indices.end(), medoid_candidate_idx)) {
                DataType loss_acc = 0;

                // foreach x_o !∈ {m_0, ..., m_i, x c}
                for (std::size_t other_sample_idx = 0; other_sample_idx < n_samples; ++other_sample_idx) {
                    if (medoid_candidate_idx != other_sample_idx &&
                        common::utils::is_element_not_in(
                            medoids_indices.begin(), medoids_indices.end(), other_sample_idx)) {
                        const auto candidate_to_other_distance =
                            pairwise_distance_matrix(medoid_candidate_idx, other_sample_idx);

                        if (candidate_to_other_distance < samples_to_nearest_medoid_distance[other_sample_idx]) {
                            // "-" to accumulate positive loss (acc of reduction of total deviation)
                            loss_acc -=
                                candidate_to_other_distance - samples_to_nearest_medoid_distance[other_sample_idx];
                        }
                    }
                }
                if (loss_acc < selected_deviation_candidate) {
                    selected_deviation_candidate = loss_acc;
                    selected_medoid_index        = medoid_candidate_idx;
                }
            }
        }
        medoids_indices.emplace_back(selected_medoid_index);
        total_deviation += selected_deviation_candidate;
        // Update the distances from each non medoid sample to its nearest medoid.
        // No need to take care about the medoids indices bc they are not swapped.
        auto samples_to_nearest_medoid_distance =
            samples_to_nth_nearest_medoid_distances(pairwise_distance_matrix, medoids_indices, /*nth_closest=*/1);
    }
    return {total_deviation, medoids_indices, samples_to_nearest_medoid_distance};
}

}  // namespace pam