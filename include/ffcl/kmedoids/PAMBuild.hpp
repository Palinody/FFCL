#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"
#include "ffcl/datastruct/matrix/PairwiseDistanceMatrix.hpp"

#include <algorithm>
#include <vector>

namespace pam {

template <typename SamplesIterator>
std::tuple<typename std::iterator_traits<SamplesIterator>::value_type,
           std::vector<std::size_t>,
           std::vector<typename std::iterator_traits<SamplesIterator>::value_type>>
build(const SamplesIterator& samples_range_first,
      const SamplesIterator& samples_range_last,
      std::size_t            n_features,
      std::size_t            n_medoids) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const std::size_t n_samples = ffcl::common::get_n_samples(samples_range_first, samples_range_last, n_features);

    // the first medoid chosen associated with the current total deviation cost
    auto [total_deviation, medoid_index_first] =
        first_medoid_td_index_pair(samples_range_first, samples_range_last, n_features);
    // Create and put the first chosen medoid index in the medoid indices vector
    auto medoids_indices = std::vector<std::size_t>(1, medoid_index_first);

    // initialize the distance from the first medoid to all samples
    auto sample_to_nearest_medoid_distances = samples_to_nth_nearest_medoid_distances(
        samples_range_first, samples_range_last, n_features, medoids_indices, /*nth_closest*/ 1);

    auto compute_distance = [&](std::size_t left_idx, std::size_t right_idx) -> DataType {
        return ffcl::common::math::heuristics::auto_distance(samples_range_first + left_idx * n_features,
                                                             samples_range_first + left_idx * n_features + n_features,
                                                             samples_range_first + right_idx * n_features);
    };

    // select the remaining medoids
    for (std::size_t medoid_index = 1; medoid_index < n_medoids; ++medoid_index) {
        // (∆TD*, x∗) ← (∞, null);
        auto selected_deviation_candidate = ffcl::common::infinity<DataType>();
        // the index of the next chosen medoid
        std::size_t selected_medoid_index = 0;
        // foreach x_c !∈ {m_1 , ..., m_i}
        for (std::size_t medoid_candidate_idx = 0; medoid_candidate_idx < n_samples; ++medoid_candidate_idx) {
            // execute only if the current candidate is not already selected as a medoid
            if (ffcl::common::is_element_not_in(medoids_indices.begin(), medoids_indices.end(), medoid_candidate_idx)) {
                DataType loss_accumulator = 0;

                // foreach x_o !∈ {m_0, ..., m_i, x c}
                for (std::size_t other_sample_idx = 0; other_sample_idx < n_samples; ++other_sample_idx) {
                    if (medoid_candidate_idx != other_sample_idx &&
                        ffcl::common::is_element_not_in(
                            medoids_indices.begin(), medoids_indices.end(), other_sample_idx)) {
                        const auto candidate_to_other_distance =
                            compute_distance(medoid_candidate_idx, other_sample_idx);

                        if (candidate_to_other_distance < sample_to_nearest_medoid_distances[other_sample_idx]) {
                            // "-" to accumulate positive loss (acc of reduction of total deviation)
                            loss_accumulator -=
                                candidate_to_other_distance - sample_to_nearest_medoid_distances[other_sample_idx];
                        }
                    }
                }
                if (loss_accumulator < selected_deviation_candidate) {
                    selected_deviation_candidate = loss_accumulator;
                    selected_medoid_index        = medoid_candidate_idx;
                }
            }
        }
        medoids_indices.emplace_back(selected_medoid_index);
        total_deviation += selected_deviation_candidate;
        // Update the distances from each non medoid sample to its nearest medoid.
        // No need to take care about the medoids indices bc they are not swapped.
        sample_to_nearest_medoid_distances = samples_to_nth_nearest_medoid_distances(
            samples_range_first, samples_range_last, n_features, medoids_indices, /*nth_closest=*/1);
    }
    return {total_deviation, medoids_indices, sample_to_nearest_medoid_distances};
}

template <typename SamplesIterator>
std::tuple<typename std::iterator_traits<SamplesIterator>::value_type,
           std::vector<std::size_t>,
           std::vector<typename std::iterator_traits<SamplesIterator>::value_type>>
build(const ffcl::datastruct::PairwiseDistanceMatrix<SamplesIterator>& pairwise_distance_matrix,
      std::size_t                                                      n_medoids) {
    return build(std::get<0>(pairwise_distance_matrix),
                 std::get<1>(pairwise_distance_matrix),
                 std::get<2>(pairwise_distance_matrix),
                 n_medoids);
}

}  // namespace pam