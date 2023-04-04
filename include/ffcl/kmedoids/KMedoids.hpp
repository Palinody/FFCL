#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/LowerTriangleMatrix.hpp"
#include "ffcl/heuristics/Heuristics.hpp"
#include "ffcl/math/random/Distributions.hpp"
#include "ffcl/math/random/Sampling.hpp"
#include "ffcl/math/statistics/Statistics.hpp"

#include "ffcl/kmedoids/FasterMSC.hpp"
#include "ffcl/kmedoids/FasterPAM.hpp"

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

namespace ffcl {

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

    template <template <typename> class KMedoidsAlgorithm, typename SamplesIterator>
    std::vector<std::size_t> fit(const SamplesIterator& data_first, const SamplesIterator& data_last);

    template <typename SamplesIterator>
    std::vector<std::size_t> fit(const SamplesIterator& data_first, const SamplesIterator& data_last);

    template <template <typename> class KMedoidsAlgorithm, typename SamplesIterator>
    std::vector<std::size_t> fit(
        const ffcl::containers::LowerTriangleMatrix<SamplesIterator>& pairwise_distance_matrix);

    template <typename SamplesIterator>
    std::vector<std::size_t> fit(
        const ffcl::containers::LowerTriangleMatrix<SamplesIterator>& pairwise_distance_matrix);

    template <typename SamplesIterator>
    std::vector<T> forward(const SamplesIterator& data_first, const SamplesIterator& data_last) const;

    template <typename SamplesIterator>
    std::vector<T> forward(
        const ffcl::containers::LowerTriangleMatrix<SamplesIterator>& pairwise_distance_matrix) const;

    template <typename SamplesIterator>
    std::vector<std::size_t> predict(const SamplesIterator& data_first, const SamplesIterator& data_last) const;

    template <typename SamplesIterator>
    std::vector<std::size_t> predict(
        const ffcl::containers::LowerTriangleMatrix<SamplesIterator>& pairwise_distance_matrix) const;

  private:
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
template <template <typename> class KMedoidsAlgorithm, typename SamplesIterator>
std::vector<std::size_t> KMedoids<T, PrecomputePairwiseDistanceMatrix>::fit(const SamplesIterator& data_first,
                                                                            const SamplesIterator& data_last) {
    using DatasetDescriptorType              = std::tuple<SamplesIterator, SamplesIterator, std::size_t>;
    DatasetDescriptorType dataset_descriptor = std::make_tuple(data_first, data_last, n_features_);

    // contains the medoids indices for each tries which number is defined by options_.n_init_ if medoids_
    // werent already assigned
    auto medoids_candidates = std::vector<std::vector<std::size_t>>();

    // default initialization of the medoids if not initialized
    if (medoids_.empty()) {
        for (std::size_t k = 0; k < options_.n_init_; ++k) {
            const auto random_medoids =
                math::random::select_from_range(n_medoids_, {0, std::distance(data_first, data_last) / n_features_});

            // default initialization of the medoids indices if not initialized
            medoids_candidates.emplace_back(random_medoids);
        }
    } else {
        // if the medoids indices were already assigned, copy them once
        medoids_candidates.emplace_back(medoids_);
    }
    // make the losses buffer for each centroids candidates
    auto candidates_losses = std::vector<T>(medoids_candidates.size());

    // creates a n_candidates vector of vectors (of n_medoids size with each elements initialized to infinity if we
    // wanted to be precise but common::utils::are_containers_equal checks for containers sizes. So we dont need to do
    // it). We could use only one candidate with a single thread but we make it thread safe this way we dont necessarily
    // need to initialize with vectors of infinities because
    auto medoids_candidates_prev = std::vector<std::vector<std::size_t>>(medoids_candidates.size());

    // instanciate a pairwise_distance_matrix only if PrecomputePairwiseDistanceMatrix is set to true
    std::unique_ptr<ffcl::containers::LowerTriangleMatrix<SamplesIterator>> pairwise_distance_matrix_ptr;

    if constexpr (PrecomputePairwiseDistanceMatrix) {
        pairwise_distance_matrix_ptr =
            std::make_unique<ffcl::containers::LowerTriangleMatrix<SamplesIterator>>(dataset_descriptor);
    }
#if defined(_OPENMP) && THREADS_ENABLED == true
#pragma omp parallel for
#endif
    for (std::size_t k = 0; k < medoids_candidates.size(); ++k) {
        auto kmedoids_algorithm =
            PrecomputePairwiseDistanceMatrix
                ? KMedoidsAlgorithm<SamplesIterator>(*pairwise_distance_matrix_ptr, medoids_candidates[k])
                : KMedoidsAlgorithm<SamplesIterator>(dataset_descriptor, medoids_candidates[k]);

        std::size_t patience_iter = 0;

        for (std::size_t iter = 0; iter < options_.max_iter_; ++iter) {
#if defined(VERBOSE) && VERBOSE == true
            // loss before step to also get the initial loss
            std::cout << kmedoids_algorithm.total_deviation() << " ";
#endif

            medoids_candidates[k] = kmedoids_algorithm.step();

            if (options_.early_stopping_ &&
                common::utils::are_containers_equal(medoids_candidates[k], medoids_candidates_prev[k])) {
                if (patience_iter == options_.patience_) {
                    break;
                }
                ++patience_iter;

            } else {
                // reset the patience iteration to zero if the medoids have changed
                patience_iter = 0;
            }
            // save the results from the current step
            medoids_candidates_prev[k] = medoids_candidates[k];
        }
#if defined(VERBOSE) && VERBOSE == true
        // final loss
        std::cout << kmedoids_algorithm.total_deviation() << " ";
        std::cout << "\n";
#endif
        // save the loss for each candidate
        candidates_losses[k] = kmedoids_algorithm.total_deviation();
    }
    // find the index of the medoids indices container with the lowest loss
    const std::size_t min_loss_index = math::statistics::argmin(candidates_losses.begin(), candidates_losses.end());
    // return best centroids accordingly to the lowest loss
    medoids_ = medoids_candidates[min_loss_index];

    return medoids_;
}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<std::size_t> KMedoids<T, PrecomputePairwiseDistanceMatrix>::fit(const SamplesIterator& data_first,
                                                                            const SamplesIterator& data_last) {
    // execute fit function with a default PAM algorithm
    return fit<ffcl::FasterPAM>(data_first, data_last);
}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
template <template <typename> class KMedoidsAlgorithm, typename SamplesIterator>
std::vector<std::size_t> KMedoids<T, PrecomputePairwiseDistanceMatrix>::fit(
    const ffcl::containers::LowerTriangleMatrix<SamplesIterator>& pairwise_distance_matrix) {
    // contains the medoids indices for each tries which number is defined by options_.n_init_ if medoids_
    // werent already assigned
    auto medoids_candidates = std::vector<std::vector<std::size_t>>();

    // default initialization of the medoids if not initialized
    if (medoids_.empty()) {
        for (std::size_t k = 0; k < options_.n_init_; ++k) {
            const auto random_medoids =
                math::random::select_from_range(n_medoids_, {0, pairwise_distance_matrix.n_samples()});

            // default initialization of the medoids indices if not initialized
            medoids_candidates.emplace_back(random_medoids);
        }
    } else {
        // if the medoids indices were already assigned, copy them once
        medoids_candidates.emplace_back(medoids_);
    }
    // make the losses buffer for each centroids candidates
    auto candidates_losses = std::vector<T>(medoids_candidates.size());

    // creates a n_candidates vector of vectors (of n_medoids size with each elements initialized to infinity if we
    // wanted to be precise but common::utils::are_containers_equal checks for containers sizes. So we dont need to do
    // it). We could use only one candidate with a single thread but we make it thread safe this way we dont necessarily
    // need to initialize with vectors of infinities because
    auto medoids_candidates_prev = std::vector<std::vector<std::size_t>>(medoids_candidates.size());

#if defined(_OPENMP) && THREADS_ENABLED == true
#pragma omp parallel for
#endif
    for (std::size_t k = 0; k < medoids_candidates.size(); ++k) {
        auto kmedoids_algorithm = KMedoidsAlgorithm<SamplesIterator>(pairwise_distance_matrix, medoids_candidates[k]);

        std::size_t patience_iter = 0;

        for (std::size_t iter = 0; iter < options_.max_iter_; ++iter) {
#if defined(VERBOSE) && VERBOSE == true
            // loss before step to also get the initial loss
            std::cout << kmedoids_algorithm.total_deviation() << " ";
#endif

            medoids_candidates[k] = kmedoids_algorithm.step();

            if (options_.early_stopping_ &&
                common::utils::are_containers_equal(medoids_candidates[k], medoids_candidates_prev[k])) {
                if (patience_iter == options_.patience_) {
                    break;
                }
                ++patience_iter;

            } else {
                // reset the patience iteration to zero if the medoids have changed
                patience_iter = 0;
            }
            // save the results from the current step
            medoids_candidates_prev[k] = medoids_candidates[k];
        }
#if defined(VERBOSE) && VERBOSE == true
        // last loss
        std::cout << kmedoids_algorithm.total_deviation() << " ";
        std::cout << "\n";
#endif
        // save the loss for each candidate
        candidates_losses[k] = kmedoids_algorithm.total_deviation();
    }
    // find the index of the medoids indices container with the lowest loss
    const std::size_t min_loss_index = math::statistics::argmin(candidates_losses.begin(), candidates_losses.end());
    // update the medoids accordingly to the lowest loss
    medoids_ = medoids_candidates[min_loss_index];

    return medoids_;
}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<std::size_t> KMedoids<T, PrecomputePairwiseDistanceMatrix>::fit(
    const ffcl::containers::LowerTriangleMatrix<SamplesIterator>& pairwise_distance_matrix) {
    // execute fit function with a default PAM algorithm
    return fit<ffcl::FasterPAM>(pairwise_distance_matrix);
}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<T> KMedoids<T, PrecomputePairwiseDistanceMatrix>::forward(const SamplesIterator& data_first,
                                                                      const SamplesIterator& data_last) const {
    return pam::utils::samples_to_nearest_medoid_distances(data_first, data_last, n_features_, medoids_);
}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<T> KMedoids<T, PrecomputePairwiseDistanceMatrix>::forward(
    const ffcl::containers::LowerTriangleMatrix<SamplesIterator>& pairwise_distance_matrix) const {
    return pam::utils::samples_to_nearest_medoid_distances(pairwise_distance_matrix, medoids_);
}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<std::size_t> KMedoids<T, PrecomputePairwiseDistanceMatrix>::predict(
    const SamplesIterator& data_first,
    const SamplesIterator& data_last) const {
    return pam::utils::samples_to_nth_nearest_medoid_indices(data_first, data_last, n_features_, medoids_);
}

template <typename T, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<std::size_t> KMedoids<T, PrecomputePairwiseDistanceMatrix>::predict(
    const ffcl::containers::LowerTriangleMatrix<SamplesIterator>& pairwise_distance_matrix) const {
    return pam::utils::samples_to_nearest_medoid_indices(pairwise_distance_matrix, medoids_);
}

}  // namespace ffcl
