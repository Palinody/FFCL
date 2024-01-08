#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"
#include "ffcl/common/math/random/Distributions.hpp"
#include "ffcl/common/math/random/Sampling.hpp"
#include "ffcl/common/math/statistics/Statistics.hpp"
#include "ffcl/datastruct/matrix/PairwiseDistanceMatrix.hpp"

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

template <typename Data, bool PrecomputePairwiseDistanceMatrix = true>
class KMedoids {
  public:
    static_assert(std::is_trivial_v<Data>, "Data must be trivial.");

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

    KMedoids(std::size_t n_medoids);

    KMedoids(std::size_t n_medoids, const Options& options);

    KMedoids(std::size_t n_medoids, const std::vector<std::size_t>& medoids_indices);

    KMedoids(std::size_t n_medoids, const std::vector<std::size_t>& medoids_indices, const Options& options);

    KMedoids(const KMedoids&) = delete;

    KMedoids<Data, PrecomputePairwiseDistanceMatrix>& set_options(const Options& options);

    template <template <typename> class KMedoidsAlgorithm, typename SamplesIterator>
    std::vector<std::size_t> fit(const SamplesIterator& samples_range_first,
                                 const SamplesIterator& samples_range_last,
                                 std::size_t            n_features);

    template <typename SamplesIterator>
    std::vector<std::size_t> fit(const SamplesIterator& samples_range_first,
                                 const SamplesIterator& samples_range_last,
                                 std::size_t            n_features);

    template <template <typename> class KMedoidsAlgorithm, typename SamplesIterator>
    std::vector<std::size_t> fit(const datastruct::PairwiseDistanceMatrix<SamplesIterator>& pairwise_distance_matrix);

    template <typename SamplesIterator>
    std::vector<std::size_t> fit(const datastruct::PairwiseDistanceMatrix<SamplesIterator>& pairwise_distance_matrix);

    template <typename SamplesIterator>
    std::vector<Data> forward(const SamplesIterator& samples_range_first,
                              const SamplesIterator& samples_range_last,
                              std::size_t            n_features) const;

    template <typename SamplesIterator>
    std::vector<Data> forward(
        const datastruct::PairwiseDistanceMatrix<SamplesIterator>& pairwise_distance_matrix) const;

    template <typename SamplesIterator>
    std::vector<std::size_t> predict(const SamplesIterator& samples_range_first,
                                     const SamplesIterator& samples_range_last,
                                     std::size_t            n_features) const;

    template <typename SamplesIterator>
    std::vector<std::size_t> predict(
        const datastruct::PairwiseDistanceMatrix<SamplesIterator>& pairwise_distance_matrix) const;

  private:
    // number of medoids that a KMedoids instance should handle (could vary)
    std::size_t n_medoids_;
    // n_medoids_ x n_features_ vectorized matrix (n_medoids_ could vary)
    std::vector<std::size_t> medoids_;

    Options options_;
};

template <typename Data, bool PrecomputePairwiseDistanceMatrix>
KMedoids<Data, PrecomputePairwiseDistanceMatrix>::KMedoids(std::size_t n_medoids)
  : n_medoids_{n_medoids} {}

template <typename Data, bool PrecomputePairwiseDistanceMatrix>
KMedoids<Data, PrecomputePairwiseDistanceMatrix>::KMedoids(std::size_t n_medoids, const Options& options)
  : n_medoids_{n_medoids}
  , options_{options} {}

template <typename Data, bool PrecomputePairwiseDistanceMatrix>
KMedoids<Data, PrecomputePairwiseDistanceMatrix>::KMedoids(std::size_t                     n_medoids,
                                                           const std::vector<std::size_t>& medoids_indices)
  : n_medoids_{n_medoids}
  , medoids_{medoids_indices} {}

template <typename Data, bool PrecomputePairwiseDistanceMatrix>
KMedoids<Data, PrecomputePairwiseDistanceMatrix>::KMedoids(std::size_t                     n_medoids,
                                                           const std::vector<std::size_t>& medoids_indices,
                                                           const Options&                  options)
  : n_medoids_{n_medoids}
  , medoids_{medoids_indices}
  , options_{options} {}

template <typename Data, bool PrecomputePairwiseDistanceMatrix>
KMedoids<Data, PrecomputePairwiseDistanceMatrix>& KMedoids<Data, PrecomputePairwiseDistanceMatrix>::set_options(
    const Options& options) {
    options_ = options;
    return *this;
}

template <typename Data, bool PrecomputePairwiseDistanceMatrix>
template <template <typename> class KMedoidsAlgorithm, typename SamplesIterator>
std::vector<std::size_t> KMedoids<Data, PrecomputePairwiseDistanceMatrix>::fit(
    const SamplesIterator& samples_range_first,
    const SamplesIterator& samples_range_last,
    std::size_t            n_features) {
    using DatasetDescriptorType              = std::tuple<SamplesIterator, SamplesIterator, std::size_t>;
    DatasetDescriptorType dataset_descriptor = std::make_tuple(samples_range_first, samples_range_last, n_features);

    // contains the medoids indices for each tries which number is defined by options_.n_init_ if medoids_
    // werent already assigned
    auto medoids_candidates = std::vector<std::vector<std::size_t>>();

    // default initialization of the medoids if not initialized
    if (medoids_.empty()) {
        for (std::size_t medoid_index = 0; medoid_index < options_.n_init_; ++medoid_index) {
            const auto random_medoids = common::math::random::select_from_range(
                n_medoids_, {0, common::get_n_samples(samples_range_first, samples_range_last, n_features)});

            // default initialization of the medoids indices if not initialized
            medoids_candidates.emplace_back(random_medoids);
        }
    } else {
        // if the medoids indices were already assigned, copy them once
        medoids_candidates.emplace_back(medoids_);
    }
    // make the losses buffer for each centroids candidates
    auto candidates_losses = std::vector<Data>(medoids_candidates.size());

    // creates a n_candidates vector of vectors (of n_medoids size with each elements initialized to infinity if we
    // wanted to be precise but common::are_containers_equal checks for datastruct sizes. So we dont need to do
    // it). We could use only one candidate with a single thread but we make it thread safe this way we dont necessarily
    // need to initialize with vectors of infinities because
    auto medoids_candidates_prev = std::vector<std::vector<std::size_t>>(medoids_candidates.size());

    // instanciate a pairwise_distance_matrix only if PrecomputePairwiseDistanceMatrix is set to true
    std::unique_ptr<datastruct::PairwiseDistanceMatrix<SamplesIterator>> pairwise_distance_matrix_ptr;

    if constexpr (PrecomputePairwiseDistanceMatrix) {
        pairwise_distance_matrix_ptr =
            std::make_unique<datastruct::PairwiseDistanceMatrix<SamplesIterator>>(dataset_descriptor);
    }
#if defined(_OPENMP) && THREADS_ENABLED == true
#pragma omp parallel for
#endif
    for (std::size_t medoid_index = 0; medoid_index < medoids_candidates.size(); ++medoid_index) {
        auto kmedoids_algorithm =
            PrecomputePairwiseDistanceMatrix
                ? KMedoidsAlgorithm(*pairwise_distance_matrix_ptr, medoids_candidates[medoid_index])
                : KMedoidsAlgorithm(dataset_descriptor, medoids_candidates[medoid_index]);

        std::size_t patience_iteration = 0;

        for (std::size_t iteration = 0; iteration < options_.max_iter_; ++iteration) {
#if defined(VERBOSE) && VERBOSE == true
            // loss before step to also get the initial loss
            std::cout << kmedoids_algorithm.total_deviation() << " ";
#endif

            medoids_candidates[medoid_index] = kmedoids_algorithm.step();

            if (options_.early_stopping_ &&
                common::are_containers_equal(medoids_candidates[medoid_index], medoids_candidates_prev[medoid_index])) {
                if (patience_iteration == options_.patience_) {
                    break;
                }
                ++patience_iteration;

            } else {
                // reset the patience iteration to zero if the medoids have changed
                patience_iteration = 0;
            }
            // save the results from the current step
            medoids_candidates_prev[medoid_index] = medoids_candidates[medoid_index];
        }
#if defined(VERBOSE) && VERBOSE == true
        // final loss
        std::cout << kmedoids_algorithm.total_deviation() << " ";
        std::cout << "\n";
#endif
        // save the loss for each candidate
        candidates_losses[medoid_index] = kmedoids_algorithm.total_deviation();
    }
    // find the index of the medoids indices container with the lowest loss
    const std::size_t min_loss_index =
        common::math::statistics::argmin(candidates_losses.begin(), candidates_losses.end());
    // return best centroids accordingly to the lowest loss
    medoids_ = medoids_candidates[min_loss_index];

    return medoids_;
}

template <typename Data, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<std::size_t> KMedoids<Data, PrecomputePairwiseDistanceMatrix>::fit(
    const SamplesIterator& samples_range_first,
    const SamplesIterator& samples_range_last,
    std::size_t            n_features) {
    // execute fit function with a default PAM algorithm
    return fit<FasterPAM>(samples_range_first, samples_range_last, n_features);
}

template <typename Data, bool PrecomputePairwiseDistanceMatrix>
template <template <typename> class KMedoidsAlgorithm, typename SamplesIterator>
std::vector<std::size_t> KMedoids<Data, PrecomputePairwiseDistanceMatrix>::fit(
    const datastruct::PairwiseDistanceMatrix<SamplesIterator>& pairwise_distance_matrix) {
    // contains the medoids indices for each tries which number is defined by options_.n_init_ if medoids_
    // werent already assigned
    auto medoids_candidates = std::vector<std::vector<std::size_t>>();

    // default initialization of the medoids if not initialized
    if (medoids_.empty()) {
        for (std::size_t medoid_index = 0; medoid_index < options_.n_init_; ++medoid_index) {
            const auto random_medoids =
                common::math::random::select_from_range(n_medoids_, {0, pairwise_distance_matrix.n_rows()});

            // default initialization of the medoids indices if not initialized
            medoids_candidates.emplace_back(random_medoids);
        }
    } else {
        // if the medoids indices were already assigned, copy them once
        medoids_candidates.emplace_back(medoids_);
    }
    // make the losses buffer for each centroids candidates
    auto candidates_losses = std::vector<Data>(medoids_candidates.size());

    // creates a n_candidates vector of vectors (of n_medoids size with each elements initialized to infinity if we
    // wanted to be precise but common::are_containers_equal checks for datastruct sizes. So we dont need to do
    // it). We could use only one candidate with a single thread but we make it thread safe this way we dont necessarily
    // need to initialize with vectors of infinities because
    auto medoids_candidates_prev = std::vector<std::vector<std::size_t>>(medoids_candidates.size());

#if defined(_OPENMP) && THREADS_ENABLED == true
#pragma omp parallel for
#endif
    for (std::size_t medoid_index = 0; medoid_index < medoids_candidates.size(); ++medoid_index) {
        auto kmedoids_algorithm = KMedoidsAlgorithm(pairwise_distance_matrix, medoids_candidates[medoid_index]);

        std::size_t patience_iteration = 0;

        for (std::size_t iteration = 0; iteration < options_.max_iter_; ++iteration) {
#if defined(VERBOSE) && VERBOSE == true
            // loss before step to also get the initial loss
            std::cout << kmedoids_algorithm.total_deviation() << " ";
#endif

            medoids_candidates[medoid_index] = kmedoids_algorithm.step();

            if (options_.early_stopping_ &&
                common::are_containers_equal(medoids_candidates[medoid_index], medoids_candidates_prev[medoid_index])) {
                if (patience_iteration == options_.patience_) {
                    break;
                }
                ++patience_iteration;

            } else {
                // reset the patience iteration to zero if the medoids have changed
                patience_iteration = 0;
            }
            // save the results from the current step
            medoids_candidates_prev[medoid_index] = medoids_candidates[medoid_index];
        }
#if defined(VERBOSE) && VERBOSE == true
        // last loss
        std::cout << kmedoids_algorithm.total_deviation() << " ";
        std::cout << "\n";
#endif
        // save the loss for each candidate
        candidates_losses[medoid_index] = kmedoids_algorithm.total_deviation();
    }
    // find the index of the medoids indices container with the lowest loss
    const std::size_t min_loss_index =
        common::math::statistics::argmin(candidates_losses.begin(), candidates_losses.end());
    // update the medoids accordingly to the lowest loss
    medoids_ = medoids_candidates[min_loss_index];

    return medoids_;
}

template <typename Data, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<std::size_t> KMedoids<Data, PrecomputePairwiseDistanceMatrix>::fit(
    const datastruct::PairwiseDistanceMatrix<SamplesIterator>& pairwise_distance_matrix) {
    // execute fit function with a default PAM algorithm
    return fit<FasterPAM>(pairwise_distance_matrix);
}

template <typename Data, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<Data> KMedoids<Data, PrecomputePairwiseDistanceMatrix>::forward(const SamplesIterator& samples_range_first,
                                                                            const SamplesIterator& samples_range_last,
                                                                            std::size_t            n_features) const {
    return pam::utils::samples_to_nearest_medoid_distances(
        samples_range_first, samples_range_last, n_features, medoids_);
}

template <typename Data, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<Data> KMedoids<Data, PrecomputePairwiseDistanceMatrix>::forward(
    const datastruct::PairwiseDistanceMatrix<SamplesIterator>& pairwise_distance_matrix) const {
    return pam::utils::samples_to_nearest_medoid_distances(pairwise_distance_matrix, medoids_);
}

template <typename Data, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<std::size_t> KMedoids<Data, PrecomputePairwiseDistanceMatrix>::predict(
    const SamplesIterator& samples_range_first,
    const SamplesIterator& samples_range_last,
    std::size_t            n_features) const {
    return pam::utils::samples_to_nth_nearest_medoid_indices(
        samples_range_first, samples_range_last, n_features, medoids_);
}

template <typename Data, bool PrecomputePairwiseDistanceMatrix>
template <typename SamplesIterator>
std::vector<std::size_t> KMedoids<Data, PrecomputePairwiseDistanceMatrix>::predict(
    const datastruct::PairwiseDistanceMatrix<SamplesIterator>& pairwise_distance_matrix) const {
    return pam::utils::samples_to_nearest_medoid_indices(pairwise_distance_matrix, medoids_);
}

}  // namespace ffcl
