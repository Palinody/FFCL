#pragma once

#include "cpp_clustering/common/Utils.hpp"
#include "cpp_clustering/containers/LowerTriangleMatrix.hpp"
#include "cpp_clustering/heuristics/Heuristics.hpp"
#include "cpp_clustering/kmedoids/PAMUtils.hpp"
#include "cpp_clustering/math/random/Distributions.hpp"

#include <algorithm>  // std::minmax_element
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>  // std::numeric_limits<T>::max()
#include <memory>
#include <numeric>
#include <tuple>
#include <variant>
#include <vector>

namespace cpp_clustering {

template <typename Iterator>
class FasterMSC {
    // couldnt make FasterMSC stable with integers so it stays disabled for now
    static_assert(std::is_floating_point_v<typename Iterator::value_type>,
                  "FasterMSC only allows floating point types.");

  public:
    using DataType = typename Iterator::value_type;

    // {samples_first_, samples_last_, n_features_}
    using DatasetDescriptorType = std::tuple<Iterator, Iterator, std::size_t>;

    using FirstVariantType   = cpp_clustering::containers::LowerTriangleMatrixDynamic<Iterator>;
    using SecondVariantType  = cpp_clustering::containers::LowerTriangleMatrix<Iterator>;
    using StorageVariantType = std::variant<FirstVariantType, SecondVariantType>;

    FasterMSC(const DatasetDescriptorType& dataset_descriptor, const std::vector<std::size_t>& medoids_indices);

    FasterMSC(const DatasetDescriptorType&    dataset_descriptor,
              const std::vector<std::size_t>& medoids_indices,
              const DataType&                 loss);

    FasterMSC(const SecondVariantType& pairwise_distance_matrix, const std::vector<std::size_t>& medoids_indices);

    FasterMSC(const SecondVariantType&        pairwise_distance_matrix,
              const std::vector<std::size_t>& medoids_indices,
              const DataType&                 loss);

    FasterMSC(const FasterMSC&) = delete;

    DataType total_deviation() const;

    std::vector<std::size_t> step();

  private:
    struct Buffers {
        Buffers(const Iterator&                 samples_first,
                const Iterator&                 samples_last,
                std::size_t                     n_features,
                const std::vector<std::size_t>& medoids_indices);

        Buffers(const DatasetDescriptorType& dataset_descriptor, const std::vector<std::size_t>& medoids_indices);

        Buffers(const SecondVariantType& pairwise_distance_matrix, const std::vector<std::size_t>& medoids_indices);

        Buffers(const Buffers&) = delete;

        void update_losses_with_closest_medoid_removal(std::size_t n_medoids);

        std::vector<std::size_t> samples_to_nearest_medoid_indices_;
        std::vector<std::size_t> samples_to_second_nearest_medoid_indices_;
        std::vector<std::size_t> samples_to_third_nearest_medoid_indices_;
        std::vector<DataType>    samples_to_nearest_medoid_distances_;
        std::vector<DataType>    samples_to_second_nearest_medoid_distances_;
        std::vector<DataType>    samples_to_third_nearest_medoid_distances_;
        std::vector<DataType>    losses_with_closest_medoid_removal_;
    };

    std::pair<DataType, std::size_t> find_best_swap(std::size_t medoid_candidate_index) const;

    std::pair<DataType, std::size_t> find_best_swap_k2(std::size_t medoid_candidate_index) const;

    DataType swap_buffers(std::size_t medoid_candidate_index, std::size_t best_swap_index);

    DataType swap_buffers_k2(std::size_t medoid_candidate_index, std::size_t best_swap_index);

    StorageVariantType       storage_variant_;
    std::size_t              n_samples_;
    std::vector<std::size_t> medoids_;
    std::unique_ptr<Buffers> buffers_ptr_;
    DataType                 loss_;
};

template <typename Iterator>
FasterMSC<Iterator>::FasterMSC(const DatasetDescriptorType&    dataset_descriptor,
                               const std::vector<std::size_t>& medoids_indices)
  : FasterMSC<Iterator>::FasterMSC(dataset_descriptor, medoids_indices, common::utils::infinity<DataType>()) {
    // compute initial loss
    loss_ = std::accumulate(buffers_ptr_->samples_to_nearest_medoid_distances_.begin(),
                            buffers_ptr_->samples_to_nearest_medoid_distances_.end(),
                            static_cast<DataType>(0));
}

template <typename Iterator>
FasterMSC<Iterator>::FasterMSC(const DatasetDescriptorType&    dataset_descriptor,
                               const std::vector<std::size_t>& medoids_indices,
                               const DataType&                 loss)
  : storage_variant_{FirstVariantType(dataset_descriptor)}
  , n_samples_{std::get<FirstVariantType>(storage_variant_).n_samples()}
  , medoids_{medoids_indices}
  , buffers_ptr_{std::make_unique<Buffers>(dataset_descriptor, medoids_)}
  , loss_{loss} {}

template <typename Iterator>
FasterMSC<Iterator>::FasterMSC(const SecondVariantType&        pairwise_distance_matrix,
                               const std::vector<std::size_t>& medoids_indices)
  : FasterMSC<Iterator>::FasterMSC(pairwise_distance_matrix, medoids_indices, common::utils::infinity<DataType>()) {
    // compute initial loss
    loss_ = std::accumulate(buffers_ptr_->samples_to_nearest_medoid_distances_.begin(),
                            buffers_ptr_->samples_to_nearest_medoid_distances_.end(),
                            static_cast<DataType>(0));
}

template <typename Iterator>
FasterMSC<Iterator>::FasterMSC(const SecondVariantType&        pairwise_distance_matrix,
                               const std::vector<std::size_t>& medoids_indices,
                               const DataType&                 loss)
  : storage_variant_{pairwise_distance_matrix}
  , n_samples_{std::get<SecondVariantType>(storage_variant_).n_samples()}
  , medoids_{medoids_indices}
  , buffers_ptr_{std::make_unique<Buffers>(pairwise_distance_matrix, medoids_)}
  , loss_{loss} {}

template <typename Iterator>
typename FasterMSC<Iterator>::DataType FasterMSC<Iterator>::total_deviation() const {
    return loss_;
}

template <typename Iterator>
std::vector<std::size_t> FasterMSC<Iterator>::step() {
    const auto& samples_to_nearest_medoid_indices = buffers_ptr_->samples_to_nearest_medoid_indices_;

    for (std::size_t medoid_candidate_index = 0; medoid_candidate_index < n_samples_; ++medoid_candidate_index) {
        // execute only if the current candidate is not already selected as a medoid
        if (medoid_candidate_index != samples_to_nearest_medoid_indices[medoid_candidate_index]) {
            // handle the case chen the number of medoids is 2
            if (medoids_.size() == 2) {
                // the total deviation change for the best medoid index 0 or 1
                const auto [new_loss, best_medoid_index] = find_best_swap_k2(medoid_candidate_index);

                if (new_loss < loss_) {
                    loss_ = swap_buffers_k2(medoid_candidate_index, best_medoid_index);
                }
            } else {
                // the total deviation change for the best swap candidate and its index in the dataset
                const auto [best_swap_delta_td, best_swap_index] = find_best_swap(medoid_candidate_index);

                if (best_swap_delta_td > 0) {
                    // swap roles of medoid m* and non-medoid x_o
                    medoids_[best_swap_index] = medoid_candidate_index;
                    // update FasterMSC buffers
                    loss_ = swap_buffers(medoid_candidate_index, best_swap_index);

                    buffers_ptr_->update_losses_with_closest_medoid_removal(medoids_.size());
                }
            }
        }
    }
    return medoids_;
}

template <typename Iterator>
std::pair<typename Iterator::value_type, std::size_t> FasterMSC<Iterator>::find_best_swap(
    std::size_t medoid_candidate_index) const {
    // TD set to the positive loss of removing medoid mi and assigning all of its members to the next best
    // alternative
    auto delta_td_mi = buffers_ptr_->losses_with_closest_medoid_removal_;
    // The negative loss of adding the replacement medoid candidate x_c
    // and reassigning all objects closest to this new medoid
    DataType delta_td_xc = 0;

    const auto& samples_to_nearest_medoid_indices          = buffers_ptr_->samples_to_nearest_medoid_indices_;
    const auto& samples_to_second_nearest_medoid_indices   = buffers_ptr_->samples_to_second_nearest_medoid_indices_;
    const auto& samples_to_nearest_medoid_distances        = buffers_ptr_->samples_to_nearest_medoid_distances_;
    const auto& samples_to_second_nearest_medoid_distances = buffers_ptr_->samples_to_second_nearest_medoid_distances_;
    const auto& samples_to_third_nearest_medoid_distances  = buffers_ptr_->samples_to_third_nearest_medoid_distances_;

    for (std::size_t other_sample_index = 0; other_sample_index < n_samples_; ++other_sample_index) {
        // candidate_to_other_distance
        const auto distance_oc =
            std::holds_alternative<FirstVariantType>(storage_variant_)
                ? std::get<FirstVariantType>(storage_variant_)(medoid_candidate_index, other_sample_index)
                : std::get<SecondVariantType>(storage_variant_)(medoid_candidate_index, other_sample_index);

        // other_sample_to_nearest_medoid_index
        const auto& index_1 = samples_to_nearest_medoid_indices[other_sample_index];
        // other_sample_to_second_nearest_medoid_index
        const auto& index_2 = samples_to_second_nearest_medoid_indices[other_sample_index];
        // other_sample_to_nearest_medoid_distance
        const auto& distance_1 = samples_to_nearest_medoid_distances[other_sample_index];
        // other_sample_to_second_nearest_medoid_distance
        const auto& distance_2 = samples_to_second_nearest_medoid_distances[other_sample_index];
        // other_sample_to_third_nearest_medoid_distance
        const auto& distance_3 = samples_to_third_nearest_medoid_distances[other_sample_index];

        if (distance_oc < distance_1) /* new closest */ {
            delta_td_xc += pam::utils::division(distance_1, distance_2) - pam::utils::division(distance_oc, distance_1);

            delta_td_mi[index_1] += pam::utils::division(distance_oc, distance_1) +
                                    pam::utils::division(distance_2, distance_3) -
                                    pam::utils::division((distance_1 + distance_oc), distance_2);

            delta_td_mi[index_2] +=
                pam::utils::division(distance_1, distance_3) - pam::utils::division(distance_1, distance_2);

        } else if (distance_oc < distance_2) /* new first/second closest */ {
            delta_td_xc += pam::utils::division(distance_1, distance_2) - pam::utils::division(distance_1, distance_oc);

            delta_td_mi[index_1] += pam::utils::division(distance_1, distance_oc) +
                                    pam::utils::division(distance_2, distance_3) -
                                    pam::utils::division((distance_1 + distance_oc), distance_2);

            delta_td_mi[index_2] +=
                pam::utils::division(distance_1, distance_3) - pam::utils::division(distance_1, distance_2);

        } else if (distance_oc < distance_3) /* new second/third closest */ {
            delta_td_mi[index_1] +=
                pam::utils::division(distance_2, distance_3) - pam::utils::division(distance_2, distance_oc);

            delta_td_mi[index_2] +=
                pam::utils::division(distance_1, distance_3) - pam::utils::division(distance_1, distance_oc);
        }
    }
    // i ← argmin(∆TD_i), with i: index of medoids elements
    const auto [best_swap_index, best_swap_distance] =
        common::utils::get_max_index_value_pair(delta_td_mi.begin(), delta_td_mi.end());

    return {delta_td_xc + best_swap_distance, best_swap_index};
}

template <typename Iterator>
std::pair<typename Iterator::value_type, std::size_t> FasterMSC<Iterator>::find_best_swap_k2(
    std::size_t medoid_candidate_index) const {
    // TD set to the positive loss of removing medoid mi and assigning all of its members to the next best
    // alternative
    auto delta_td_mi = std::vector<DataType>(2);
    // The negative loss of adding the replacement medoid candidate x_c
    // and reassigning all objects closest to this new medoid
    DataType delta_td_xc = 0;

    const auto& samples_to_nearest_medoid_distances        = buffers_ptr_->samples_to_nearest_medoid_distances_;
    const auto& samples_to_second_nearest_medoid_distances = buffers_ptr_->samples_to_second_nearest_medoid_distances_;

    for (std::size_t other_sample_index = 0; other_sample_index < n_samples_; ++other_sample_index) {
        // candidate_to_other_distance
        const auto distance_oc =
            std::holds_alternative<FirstVariantType>(storage_variant_)
                ? std::get<FirstVariantType>(storage_variant_)(medoid_candidate_index, other_sample_index)
                : std::get<SecondVariantType>(storage_variant_)(medoid_candidate_index, other_sample_index);

        // other_sample_to_nearest_medoid_distance
        const auto& distance_1 = samples_to_nearest_medoid_distances[other_sample_index];
        // other_sample_to_second_nearest_medoid_distance
        const auto& distance_2 = samples_to_second_nearest_medoid_distances[other_sample_index];

        // We do not use the assignment here, because we stored d0/d1 by medoid position, not closeness
        delta_td_mi[0] += (distance_oc < distance_2) ? pam::utils::division(distance_oc, distance_2)
                                                     : pam::utils::division(distance_2, distance_oc);

        delta_td_mi[1] += (distance_oc < distance_1) ? pam::utils::division(distance_oc, distance_1)
                                                     : pam::utils::division(distance_1, distance_oc);
    }
    // i ← argmin(∆TD_i), with i: index of medoids elements
    const auto [best_swap_index, best_swap_distance] =
        common::utils::get_min_index_value_pair(delta_td_mi.begin(), delta_td_mi.end());

    return {delta_td_xc + best_swap_distance, best_swap_index};
}

template <typename Iterator>
typename Iterator::value_type FasterMSC<Iterator>::swap_buffers(std::size_t medoid_candidate_index,
                                                                std::size_t best_swap_index) {
    DataType loss = 0;

    auto& samples_to_nearest_medoid_indices          = buffers_ptr_->samples_to_nearest_medoid_indices_;
    auto& samples_to_second_nearest_medoid_indices   = buffers_ptr_->samples_to_second_nearest_medoid_indices_;
    auto& samples_to_third_nearest_medoid_indices    = buffers_ptr_->samples_to_third_nearest_medoid_indices_;
    auto& samples_to_nearest_medoid_distances        = buffers_ptr_->samples_to_nearest_medoid_distances_;
    auto& samples_to_second_nearest_medoid_distances = buffers_ptr_->samples_to_second_nearest_medoid_distances_;
    auto& samples_to_third_nearest_medoid_distances  = buffers_ptr_->samples_to_third_nearest_medoid_distances_;

    for (std::size_t other_sample_index = 0; other_sample_index < n_samples_; ++other_sample_index) {
        // other_sample_to_nearest_medoid_index
        auto& index_1 = samples_to_nearest_medoid_indices[other_sample_index];
        // other_sample_to_seacond_nearest_medoid_index
        auto& index_2 = samples_to_second_nearest_medoid_indices[other_sample_index];
        // other_sample_to_third_medoid_index
        auto& index_3 = samples_to_third_nearest_medoid_indices[other_sample_index];
        // other_sample_to_nearest_medoid_distance
        auto& distance_1 = samples_to_nearest_medoid_distances[other_sample_index];
        // other_sample_to_second_nearest_medoid_distance
        auto& distance_2 = samples_to_second_nearest_medoid_distances[other_sample_index];
        // other_sample_to_third_nearest_medoid_distance
        auto& distance_3 = samples_to_third_nearest_medoid_distances[other_sample_index];

        if (other_sample_index == medoid_candidate_index) {
            if (index_1 != best_swap_index) {
                if (index_2 != best_swap_index) {
                    index_3    = index_2;  // if we keep a record of third nearest indices
                    distance_3 = distance_2;
                }
                index_2    = index_1;
                distance_2 = distance_1;
            }
            index_1    = best_swap_index;
            distance_1 = 0;
            continue;
        }
        // candidate_to_other_distance
        const auto distance_oc =
            std::holds_alternative<FirstVariantType>(storage_variant_)
                ? std::get<FirstVariantType>(storage_variant_)(medoid_candidate_index, other_sample_index)
                : std::get<SecondVariantType>(storage_variant_)(medoid_candidate_index, other_sample_index);

        // nearest medoid is gone
        if (index_1 == best_swap_index) {
            if (distance_oc < distance_2) {
                index_1    = best_swap_index;
                distance_1 = distance_oc;

            } else if (distance_oc < distance_3 || index_3 == common::utils::infinity<std::size_t>()) {
                index_1    = index_2;
                distance_1 = distance_2;
                index_2    = best_swap_index;
                distance_2 = distance_oc;

            } else {
                index_1    = index_2;
                distance_1 = distance_2;
                index_2    = index_3;
                distance_2 = distance_3;
                // START: update third nearest medoid
                std::size_t index_tmp    = best_swap_index;
                auto        distance_tmp = distance_oc;
                for (std::size_t idx = 0; idx < medoids_.size(); ++idx) {
                    if (idx != index_1 && idx != best_swap_index && idx != index_2) {
                        // distance from other object to looped medoid
                        const auto distance_om =
                            std::holds_alternative<FirstVariantType>(storage_variant_)
                                ? std::get<FirstVariantType>(storage_variant_)(medoids_[idx], other_sample_index)
                                : std::get<SecondVariantType>(storage_variant_)(medoids_[idx], other_sample_index);

                        if (distance_om < distance_tmp) {
                            index_tmp    = idx;
                            distance_tmp = distance_om;
                        }
                    }
                }
                index_3    = index_tmp;
                distance_3 = distance_tmp;
                // END: update third nearest medoid
            }
        } else if (index_2 == best_swap_index) {
            // second nearest was replaced
            if (distance_oc < distance_1) {
                index_2    = index_1;
                distance_2 = distance_1;
                index_1    = best_swap_index;
                distance_1 = distance_oc;

            } else if (distance_oc < distance_3 || index_3 == common::utils::infinity<std::size_t>()) {
                index_2    = best_swap_index;
                distance_2 = distance_oc;

            } else {
                index_2    = index_3;
                distance_2 = distance_3;
                // START: update third nearest medoid
                std::size_t index_tmp    = best_swap_index;
                auto        distance_tmp = distance_oc;
                for (std::size_t idx = 0; idx < medoids_.size(); ++idx) {
                    if (idx != index_1 && idx != best_swap_index && idx != index_2) {
                        // distance from other object to looped medoid
                        const auto distance_om =
                            std::holds_alternative<FirstVariantType>(storage_variant_)
                                ? std::get<FirstVariantType>(storage_variant_)(medoids_[idx], other_sample_index)
                                : std::get<SecondVariantType>(storage_variant_)(medoids_[idx], other_sample_index);

                        if (distance_om < distance_tmp) {
                            index_tmp    = idx;
                            distance_tmp = distance_om;
                        }
                    }
                }
                index_3    = index_tmp;
                distance_3 = distance_tmp;
                // END: update third nearest medoid
            }
        } else {
            // nearest not removed
            if (distance_oc < distance_1) {
                index_3    = index_2;
                distance_3 = distance_2;
                index_2    = index_1;
                distance_2 = distance_1;
                index_1    = best_swap_index;
                distance_1 = distance_oc;

            } else if (distance_oc < distance_2) {
                index_3    = index_2;
                distance_3 = distance_2;
                index_2    = best_swap_index;
                distance_2 = distance_oc;

            } else if (distance_oc < distance_3 || index_3 == common::utils::infinity<std::size_t>()) {
                index_3    = best_swap_index;
                distance_3 = distance_oc;

            } else if (index_3 == best_swap_index) {
                // START: update third nearest medoid
                std::size_t index_tmp    = best_swap_index;
                auto        distance_tmp = distance_oc;
                for (std::size_t idx = 0; idx < medoids_.size(); ++idx) {
                    if (idx != index_1 && idx != best_swap_index && idx != index_2) {
                        // distance from other object to looped medoid
                        const auto distance_om =
                            std::holds_alternative<FirstVariantType>(storage_variant_)
                                ? std::get<FirstVariantType>(storage_variant_)(medoids_[idx], other_sample_index)
                                : std::get<SecondVariantType>(storage_variant_)(medoids_[idx], other_sample_index);

                        if (distance_om < distance_tmp) {
                            index_tmp    = idx;
                            distance_tmp = distance_om;
                        }
                    }
                }
                index_3    = index_tmp;
                distance_3 = distance_tmp;
                // END: update third nearest medoid
            }
        }
        loss += pam::utils::division(distance_1, distance_2);
    }
    return loss;
}

template <typename Iterator>
typename Iterator::value_type FasterMSC<Iterator>::swap_buffers_k2(std::size_t medoid_candidate_index,
                                                                   std::size_t best_swap_index) {
    medoids_[best_swap_index] = medoid_candidate_index;

    DataType loss = 0;

    auto& samples_to_nearest_medoid_indices          = buffers_ptr_->samples_to_nearest_medoid_indices_;
    auto& samples_to_nearest_medoid_distances        = buffers_ptr_->samples_to_nearest_medoid_distances_;
    auto& samples_to_second_nearest_medoid_distances = buffers_ptr_->samples_to_second_nearest_medoid_distances_;

    for (std::size_t other_sample_index = 0; other_sample_index < n_samples_; ++other_sample_index) {
        // other_sample_to_nearest_medoid_index
        auto& index_1 = samples_to_nearest_medoid_indices[other_sample_index];
        // other_sample_to_nearest_medoid_distance
        auto& distance_1 = samples_to_nearest_medoid_distances[other_sample_index];
        // other_sample_to_second_nearest_medoid_distance
        auto& distance_2 = samples_to_second_nearest_medoid_distances[other_sample_index];

        if (other_sample_index == medoid_candidate_index) {
            index_1    = best_swap_index;
            distance_1 = distance_2 = 0;
            continue;
        }
        // candidate_to_other_distance
        const auto distance_oc =
            std::holds_alternative<FirstVariantType>(storage_variant_)
                ? std::get<FirstVariantType>(storage_variant_)(medoid_candidate_index, other_sample_index)
                : std::get<SecondVariantType>(storage_variant_)(medoid_candidate_index, other_sample_index);

        if (best_swap_index == 0) {
            distance_1 = distance_oc;
            if (distance_oc < distance_2 || (distance_oc == distance_2 && index_1 == 0)) {
                index_1 = 0;
                loss += pam::utils::division(distance_oc, distance_2);
            } else {
                index_1 = 1;
                loss += pam::utils::division(distance_2, distance_oc);
            }
        } else {
            distance_2 = distance_oc;
            if (distance_oc < distance_1 || (distance_oc == distance_1 && index_1 == 1)) {
                index_1 = 1;
                loss += pam::utils::division(distance_oc, distance_1);
            } else {
                index_1 = 0;
                loss += pam::utils::division(distance_1, distance_oc);
            }
        }
    }
    return loss;
}

template <typename Iterator>
FasterMSC<Iterator>::Buffers::Buffers(const Iterator&                 samples_first,
                                      const Iterator&                 samples_last,
                                      std::size_t                     n_features,
                                      const std::vector<std::size_t>& medoids_indices)
  : samples_to_nearest_medoid_indices_{pam::utils::samples_to_nth_nearest_medoid_indices(samples_first,
                                                                                         samples_last,
                                                                                         n_features,
                                                                                         medoids_indices,
                                                                                         /*n_closest=*/1)}
  , /* Not required when n_medoids = 2 (K2) */
  samples_to_second_nearest_medoid_indices_{(medoids_indices.size() > 2)
                                                ? pam::utils::samples_to_nth_nearest_medoid_indices(samples_first,
                                                                                                    samples_last,
                                                                                                    n_features,
                                                                                                    medoids_indices,
                                                                                                    /*n_closest=*/2)
                                                : std::vector<std::size_t>({})}
  , /* Not required when n_medoids = 2 (K2) */
  samples_to_third_nearest_medoid_indices_{(medoids_indices.size() > 2)
                                               ? pam::utils::samples_to_nth_nearest_medoid_indices(samples_first,
                                                                                                   samples_last,
                                                                                                   n_features,
                                                                                                   medoids_indices,
                                                                                                   /*n_closest=*/3)
                                               : std::vector<std::size_t>({})}

  , samples_to_nearest_medoid_distances_{pam::utils::samples_to_nth_nearest_medoid_distances(samples_first,
                                                                                             samples_last,
                                                                                             n_features,
                                                                                             medoids_indices,
                                                                                             /*n_closest=*/1)}
  , samples_to_second_nearest_medoid_distances_{pam::utils::samples_to_nth_nearest_medoid_distances(samples_first,
                                                                                                    samples_last,
                                                                                                    n_features,
                                                                                                    medoids_indices,
                                                                                                    /*n_closest=*/2)}
  , /* Not required when n_medoids = 2 (K2) */
  samples_to_third_nearest_medoid_distances_{(medoids_indices.size() > 2)
                                                 ? pam::utils::samples_to_nth_nearest_medoid_distances(samples_first,
                                                                                                       samples_last,
                                                                                                       n_features,
                                                                                                       medoids_indices,
                                                                                                       /*n_closest=*/3)
                                                 : std::vector<DataType>({})}
  , /* Not required when n_medoids = 2 (K2) */
  losses_with_closest_medoid_removal_{(medoids_indices.size() > 2)
                                          ? pam::utils::compute_losses_with_silhouette_medoid_removal<DataType>(
                                                samples_to_nearest_medoid_indices_,
                                                samples_to_second_nearest_medoid_indices_,
                                                samples_to_nearest_medoid_distances_,
                                                samples_to_second_nearest_medoid_distances_,
                                                samples_to_third_nearest_medoid_distances_,
                                                medoids_indices.size())
                                          : std::vector<DataType>({})} {}

template <typename Iterator>
FasterMSC<Iterator>::Buffers::Buffers(const DatasetDescriptorType&    dataset_descriptor,
                                      const std::vector<std::size_t>& medoids_indices)
  : FasterMSC<Iterator>::Buffers::Buffers(std::get<0>(dataset_descriptor),
                                          std::get<1>(dataset_descriptor),
                                          std::get<2>(dataset_descriptor),
                                          medoids_indices) {}

template <typename Iterator>
FasterMSC<Iterator>::Buffers::Buffers(const SecondVariantType&        pairwise_distance_matrix,
                                      const std::vector<std::size_t>& medoids_indices)
  : samples_to_nearest_medoid_indices_{pam::utils::samples_to_nth_nearest_medoid_indices(pairwise_distance_matrix,
                                                                                         medoids_indices,
                                                                                         /*n_closest=*/1)}
  , /* Not required when n_medoids = 2 (K2) */
  samples_to_second_nearest_medoid_indices_{
      (medoids_indices.size() > 2) ? pam::utils::samples_to_nth_nearest_medoid_indices(pairwise_distance_matrix,
                                                                                       medoids_indices,
                                                                                       /*n_closest=*/2)
                                   : std::vector<std::size_t>({})}
  , /* Not required when n_medoids = 2 (K2) */
  samples_to_third_nearest_medoid_indices_{
      (medoids_indices.size() > 2) ? pam::utils::samples_to_nth_nearest_medoid_indices(pairwise_distance_matrix,
                                                                                       medoids_indices,
                                                                                       /*n_closest=*/3)
                                   : std::vector<std::size_t>({})}

  , samples_to_nearest_medoid_distances_{pam::utils::samples_to_nth_nearest_medoid_distances(pairwise_distance_matrix,
                                                                                             medoids_indices,
                                                                                             /*n_closest=*/1)}
  , samples_to_second_nearest_medoid_distances_{pam::utils::samples_to_nth_nearest_medoid_distances(
        pairwise_distance_matrix,
        medoids_indices,
        /*n_closest=*/2)}
  , /* Not required when n_medoids = 2 (K2) */
  samples_to_third_nearest_medoid_distances_{
      (medoids_indices.size() > 2) ? pam::utils::samples_to_nth_nearest_medoid_distances(pairwise_distance_matrix,
                                                                                         medoids_indices,
                                                                                         /*n_closest=*/3)
                                   : std::vector<DataType>({})}
  , /* Not required when n_medoids = 2 (K2) */
  losses_with_closest_medoid_removal_{(medoids_indices.size() > 2)
                                          ? pam::utils::compute_losses_with_silhouette_medoid_removal<DataType>(
                                                samples_to_nearest_medoid_indices_,
                                                samples_to_second_nearest_medoid_indices_,
                                                samples_to_nearest_medoid_distances_,
                                                samples_to_second_nearest_medoid_distances_,
                                                samples_to_third_nearest_medoid_distances_,
                                                medoids_indices.size())
                                          : std::vector<DataType>({})} {}

template <typename Iterator>
void FasterMSC<Iterator>::Buffers::update_losses_with_closest_medoid_removal(std::size_t n_medoids) {
    losses_with_closest_medoid_removal_ =
        pam::utils::compute_losses_with_silhouette_medoid_removal<DataType>(samples_to_nearest_medoid_indices_,
                                                                            samples_to_second_nearest_medoid_indices_,
                                                                            samples_to_nearest_medoid_distances_,
                                                                            samples_to_second_nearest_medoid_distances_,
                                                                            samples_to_third_nearest_medoid_distances_,
                                                                            n_medoids);
}

}  // namespace cpp_clustering