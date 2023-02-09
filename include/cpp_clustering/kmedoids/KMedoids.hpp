#pragma once

#include "cpp_clustering/common/Utils.hpp"
#include "cpp_clustering/heuristics/Heuristics.hpp"
#include "cpp_clustering/math/random/Distributions.hpp"

#include "cpp_clustering/kmedoids/FasterMSC.hpp"
#include "cpp_clustering/kmedoids/FasterPAM.hpp"

#include <algorithm>  // std::minmax_element
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>  // std::numeric_limits<T>::max()
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

namespace cpp_clustering {

template <typename T, bool PrecomputePairwiseDistanceMatrix = true>
class KMedoids {
  public:
    struct Options {
        Options& max_iter(std::size_t max_iter) {
            max_iter_ = max_iter;
            return *this;
        }

        Options& early_stopping(bool early_stopping) {
            early_stopping_ = early_stopping;
            return *this;
        }

        Options& patience(std::size_t patience) {
            patience_ = patience;
            return *this;
        }

        Options& n_init(std::size_t n_init) {
            n_init_ = n_init;
            return *this;
        }

        Options& operator=(const Options& options) {
            max_iter_       = options.max_iter_;
            early_stopping_ = options.early_stopping_;
            patience_       = options.patience_;
            n_init_         = options.n_init_;
            return *this;
        }

        std::size_t max_iter_       = 100;
        bool        early_stopping_ = true;
        std::size_t patience_       = 0;
        std::size_t n_init_         = 1;
    };

    KMedoids(std::size_t n_medoids, std::size_t n_features);

    KMedoids(std::size_t n_medoids, std::size_t n_features, const Options& options);

    KMedoids(std::size_t n_medoids, std::size_t n_features, const std::vector<std::size_t>& medoids_indices);

    KMedoids(std::size_t                     n_medoids,
             std::size_t                     n_features,
             const std::vector<std::size_t>& medoids_indices,
             const Options&                  options);

    KMedoids(const KMedoids&) = delete;

    KMedoids<T, PrecomputePairwiseDistanceMatrix>& set_options(const Options& options);

    template <template <typename, bool> class PAMClass, typename SamplesIterator>
    std::vector<T> fit(const SamplesIterator& data_first, const SamplesIterator& data_last);

    template <typename SamplesIterator>
    std::vector<T> fit(const SamplesIterator& data_first, const SamplesIterator& data_last);

    template <typename SamplesIterator>
    std::vector<T> forward(const SamplesIterator& data_first, const SamplesIterator& data_last) const;

    template <typename SamplesIterator>
    std::vector<std::size_t> predict(const SamplesIterator& data_first, const SamplesIterator& data_last) const;

    template <typename SamplesIterator, typename LabelsIterator>
    std::pair<std::size_t, std::vector<T>> swap_to_best_count_match(const SamplesIterator& samples_first,
                                                                    const SamplesIterator& samples_last,
                                                                    LabelsIterator         labels_first,
                                                                    LabelsIterator         labels_last);

    template <typename SamplesIterator, typename LabelsIterator>
    std::pair<std::size_t, std::vector<T>> remap_centroid_to_label_index(const SamplesIterator& samples_first,
                                                                         const SamplesIterator& samples_last,
                                                                         LabelsIterator         labels_first,
                                                                         LabelsIterator         labels_last,
                                                                         std::size_t            n_classes = 0);

  private:
    template <typename SamplesIterator>
    std::vector<T> medoids_to_centroids(const SamplesIterator& data_first, const SamplesIterator& data_last) const;

    // assign each sample (S) to its closest medoid (M)
    template <typename SamplesIterator>
    std::vector<std::size_t> assign(const SamplesIterator& data_first, const SamplesIterator& data_last) const;

    // number of medoids that a KMedoids instance should handle (could vary)
    std::size_t n_medoids_;
    // number of features (dimensions) that a KMedoids instance should handle
    std::size_t n_features_;
    // n_medoids_ x n_features_ vectorized matrix (n_medoids_ could vary)
    std::vector<std::size_t> medoids_;

    Options options_;
};

template <typename T, bool PrecomputePairwiseDistanceMatrix>
KMedoids<T, PrecomputePairwiseDistanceMatrix>::KMedoids(std::size_t n_medoids, std::size_t n_features)
  : n_medoids_{n_medoids}
  , n_features_{n_features} {}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
KMedoids<T, PrecomputePairwiseDistanceMatrix>::KMedoids(std::size_t    n_medoids,
                                                        std::size_t    n_features,
                                                        const Options& options)
  : n_medoids_{n_medoids}
  , n_features_{n_features}
  , options_{options} {}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
KMedoids<T, PrecomputePairwiseDistanceMatrix>::KMedoids(std::size_t                     n_medoids,
                                                        std::size_t                     n_features,
                                                        const std::vector<std::size_t>& medoids_indices)
  : n_medoids_{n_medoids}
  , n_features_{n_features}
  , medoids_{medoids_indices} {}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
KMedoids<T, PrecomputePairwiseDistanceMatrix>::KMedoids(std::size_t                     n_medoids,
                                                        std::size_t                     n_features,
                                                        const std::vector<std::size_t>& medoids_indices,
                                                        const Options&                  options)
  : n_medoids_{n_medoids}
  , n_features_{n_features}
  , medoids_{medoids_indices}
  , options_{options} {}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
KMedoids<T, PrecomputePairwiseDistanceMatrix>& KMedoids<T, PrecomputePairwiseDistanceMatrix>::set_options(
    const Options& options) {
    options_ = options;
    return *this;
}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<T> KMedoids<T, PrecomputePairwiseDistanceMatrix>::medoids_to_centroids(
    const SamplesIterator& data_first,
    const SamplesIterator& data_last) const {
    auto clusters = std::vector<T>(n_medoids_ * n_features_);

    for (std::size_t k = 0; k < n_medoids_; ++k) {
        const std::size_t sample_index = medoids_[k];
        std::copy(data_first + sample_index * n_features_,
                  data_first + sample_index * n_features_ + n_features_,
                  clusters.begin() + k * n_features_);
    }
    return clusters;
}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<std::size_t> KMedoids<T, PrecomputePairwiseDistanceMatrix>::assign(const SamplesIterator& data_first,
                                                                               const SamplesIterator& data_last) const {
    return pam_utils::samples_to_nth_nearest_medoid_indices(
        data_first, data_last, n_features_, medoids_, /*n_closest=*/1);
}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
template <template <typename, bool> class PAMClass, typename SamplesIterator>
std::vector<T> KMedoids<T, PrecomputePairwiseDistanceMatrix>::fit(const SamplesIterator& data_first,
                                                                  const SamplesIterator& data_last) {
    // contains the medoids indices for each tries which number is defined by options_.n_init_ if medoids_
    // werent already assigned
    auto medoids_indices_candidates = std::vector<std::vector<std::size_t>>();

    // default initialization of the medoids if not initialized
    if (medoids_.empty()) {
        for (std::size_t k = 0; k < options_.n_init_; ++k) {
            const auto random_medoids =
                common::utils::select_from_range(n_medoids_, {0, (data_last - data_first) / n_features_});

            // default initialization of the medoids indices if not initialized
            medoids_indices_candidates.emplace_back(random_medoids);
        }
    } else {
        // if the medoids indices were already assigned, copy them once
        medoids_indices_candidates.emplace_back(medoids_);
    }
    // make the losses buffer for each centroids candidates
    auto candidates_losses = std::vector<T>(medoids_indices_candidates.size());

    // creates a n_candidates vector of vectors (of n_medoids size with each elements initialized to infinity if we
    // wanted to be precise but common::utils::are_containers_equal checks for containers sizes. So we dont need to do
    // it). We could use only one candidate with a single thread but we make it thread safe this way we dont necessarily
    // need to initialize with vectors of infinities because
    auto medoids_indices_candidates_prev = std::vector<std::vector<std::size_t>>(medoids_indices_candidates.size());

    for (std::size_t k = 0; k < medoids_indices_candidates.size(); ++k) {
        // assign the centroids attributes to the current centroids
        medoids_ = medoids_indices_candidates[k];

        auto pam =
            PAMClass<SamplesIterator, PrecomputePairwiseDistanceMatrix>(data_first, data_last, n_features_, medoids_);

        std::size_t patience_iter = 0;

        for (std::size_t iter = 0; iter < options_.max_iter_; ++iter) {
            // loss before step to also get the initial loss
            std::cout << pam.total_deviation() << " ";

            medoids_ = pam.step();

            if (options_.early_stopping_ &&
                common::utils::are_containers_equal(medoids_, medoids_indices_candidates_prev[k])) {
                if (patience_iter == options_.patience_) {
                    break;
                }
                ++patience_iter;
            } else {
                // reset the patience iteration to zero if the medoids have changed
                patience_iter = 0;
            }
            medoids_indices_candidates_prev[k] = medoids_;
        }
        // last loss
        std::cout << pam.total_deviation() << " ";
        std::cout << "\n";

        // once the training loop is finished, update the medoids indices candidate
        medoids_indices_candidates[k] = medoids_;
        // save the loss for each candidate
        candidates_losses[k] = pam.total_deviation();
    }
    // find the index of the medoids indices container with the lowest loss
    const std::size_t min_loss_index = common::utils::argmin(candidates_losses.begin(), candidates_losses.end());
    // return best centroids accordingly to the lowest loss
    medoids_ = medoids_indices_candidates[min_loss_index];
    return medoids_to_centroids(data_first, data_last);
}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<T> KMedoids<T, PrecomputePairwiseDistanceMatrix>::fit(const SamplesIterator& data_first,
                                                                  const SamplesIterator& data_last) {
    // execute fit function with a default PAM algorithm
    return fit<cpp_clustering::FasterPAM>(data_first, data_last);
}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<T> KMedoids<T, PrecomputePairwiseDistanceMatrix>::forward(const SamplesIterator& data_first,
                                                                      const SamplesIterator& data_last) const {
    return pam_utils::samples_to_nth_nearest_medoid_distances(
        data_first, data_last, n_features_, medoids_, /*n_closest=*/1);
}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<std::size_t> KMedoids<T, PrecomputePairwiseDistanceMatrix>::predict(
    const SamplesIterator& data_first,
    const SamplesIterator& data_last) const {
    return assign(data_first, data_last);
}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator, typename LabelsIterator>
std::pair<std::size_t, std::vector<T>> KMedoids<T, PrecomputePairwiseDistanceMatrix>::swap_to_best_count_match(
    const SamplesIterator& samples_first,
    const SamplesIterator& samples_last,
    LabelsIterator         labels_first,
    LabelsIterator         labels_last) {
    const std::size_t n_labels      = std::distance(labels_first, labels_last);
    auto              medoids_order = std::vector<std::size_t>(n_medoids_);
    // Initial order is in increasing order (no swap)
    std::iota(medoids_order.begin(), medoids_order.end(), 0);

    auto predicted_labels = predict(samples_first, samples_last);
    // initialize the best match count with the current order
    std::size_t best_match_count =
        common::utils::count_matches(predicted_labels.begin(), predicted_labels.end(), labels_first);

    // if the first best math count matches all the labels then theres nothing to swap
    if (best_match_count == n_labels) {
        return {best_match_count, medoids_to_centroids(samples_first, samples_last)};
    }
    // keep a record of the best bedoid order
    auto best_medoids_indices = medoids_;
    // save a copy of the non swapped, original medoids
    const auto medoids_indices_orig = medoids_;

    while (std::next_permutation(medoids_order.begin(), medoids_order.end())) {
        // swap the medoids inplace
        medoids_ = common::utils::permutation_from_indices(medoids_indices_orig, medoids_order);
        // update the preedicted labels based on the current medoids ordering
        predicted_labels = predict(samples_first, samples_last);
        // recompute the match counter
        const std::size_t match_count_candidate =
            common::utils::count_matches(predicted_labels.begin(), predicted_labels.end(), labels_first);

        if (match_count_candidate > best_match_count) {
            best_match_count     = match_count_candidate;
            best_medoids_indices = medoids_;

            // if the count matches all the labels then we are sure that theres no best swap
            if (match_count_candidate == n_labels) {
                return {best_match_count, medoids_to_centroids(samples_first, samples_last)};
            }
        }
    }
    // finally save the best medoids order
    medoids_ = best_medoids_indices;
    // return the best match count with its associated reordered centroids
    return {best_match_count, medoids_to_centroids(samples_first, samples_last)};
}

template <typename T>
void print_matrix(const std::vector<std::vector<T>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

template <typename Container>
void print(const Container& container) {
    for (const auto& c : container) {
        std::cout << c << " ";
    }
    std::cout << "\n";
}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator, typename LabelsIterator>
std::pair<std::size_t, std::vector<T>> KMedoids<T, PrecomputePairwiseDistanceMatrix>::remap_centroid_to_label_index(
    const SamplesIterator& samples_first,
    const SamplesIterator& samples_last,
    LabelsIterator         labels_first,
    LabelsIterator         labels_last,
    std::size_t            n_classes) {
    // find the max element in the labels iterator if its not provided. Add one because the max element is an index
    n_classes = n_classes ? n_classes : 1 + *std::max_element(labels_first, labels_last);

    auto medoids_order = std::vector<std::size_t>(n_medoids_);
    std::iota(medoids_order.begin(), medoids_order.end(), 0);
    // buffer the scores obtained with each medoid for each class
    auto medoids_to_class_counts =
        std::vector<std::vector<std::size_t>>(n_medoids_, std::vector<std::size_t>(n_classes));
    // save a copy of the non swapped, original medoids
    const auto medoids_orig = medoids_;

    for (std::size_t k = 0; k < n_medoids_; ++k) {
        for (std::size_t c = 0; c < n_classes; ++c) {
            // swap medoid k with class label c
            std::swap(medoids_order[k], medoids_order[c]);
            // perform swap on the medoids k and c
            medoids_ = common::utils::permutation_from_indices(medoids_orig, medoids_order);

            auto predicted_labels = predict(samples_first, samples_last);
            // count matches for medoid k when we move it to cluster c with the same class index k
            const auto best_match_count = common::utils::count_matches_for_value(
                predicted_labels.begin(), predicted_labels.end(), labels_first, /*value=*/k);

            medoids_to_class_counts[k][c] = best_match_count;
            // reswap medoids k and c to their original index
            std::swap(medoids_order[k], medoids_order[c]);
        }
    }
    print_matrix(medoids_to_class_counts);

    std::size_t best_match_count = 0;
    for (std::size_t k = 0; k < n_medoids_; ++k) {
        const auto class_counts = medoids_to_class_counts[k];
        // get the index for the current medoid that maximizes its score for a class c
        const auto [max_index, max_value] =
            common::utils::get_max_index_value_pair(class_counts.begin(), class_counts.end());

        medoids_order[k] = max_index;
        best_match_count += max_value;
    }
    std::cout << "medoids order after\n";
    print(medoids_order);
    // finally save the best medoids order
    medoids_ = common::utils::permutation_from_indices(medoids_orig, medoids_order);

    // return the best match count with its associated reordered medoids
    return {best_match_count, medoids_to_centroids(samples_first, samples_last)};
}

}  // namespace cpp_clustering
