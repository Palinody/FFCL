#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"
#include "ffcl/common/math/random/Distributions.hpp"
#include "ffcl/common/math/random/Sampling.hpp"
#include "ffcl/common/math/statistics/Statistics.hpp"
#include "ffcl/kmeans/Hamerly.hpp"
#include "ffcl/kmeans/KMeansPlusPlus.hpp"
#include "ffcl/kmeans/Lloyd.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

namespace ffcl {

template <typename DataType>
class KMeans {
    static_assert(std::is_floating_point<DataType>::value, "KMeans only allows floating point types.");

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

        Options& tolerance(const DataType& tolerance) {
            tolerance_ = tolerance;
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
            tolerance_      = options.tolerance_;
            n_init_         = options.n_init_;
            return *this;
        }

        std::size_t max_iter_       = 100;
        bool        early_stopping_ = true;
        std::size_t patience_       = 0;
        DataType    tolerance_      = 0;
        std::size_t n_init_         = 1;
    };

  public:
    KMeans(std::size_t n_centroids, std::size_t n_features);

    KMeans(std::size_t n_centroids, std::size_t n_features, const Options& options);

    KMeans(std::size_t n_centroids, std::size_t n_features, const std::vector<DataType>& centroids);

    KMeans(std::size_t                  n_centroids,
           std::size_t                  n_features,
           const std::vector<DataType>& centroids,
           const Options&               options);

    KMeans(const KMeans&) = delete;

    KMeans<DataType>& set_options(const Options& options);

    template <template <typename> class KMeansAlgorithm, typename SamplesIterator, typename Function>
    std::vector<DataType> fit(const SamplesIterator& samples_range_first,
                              const SamplesIterator& samples_range_last,
                              const Function&        centroids_initializer);

    template <template <typename> class KMeansAlgorithm, typename SamplesIterator>
    std::vector<DataType> fit(const SamplesIterator& samples_range_first, const SamplesIterator& samples_range_last);

    template <typename SamplesIterator, typename Function>
    std::vector<DataType> fit(const SamplesIterator& samples_range_first,
                              const SamplesIterator& samples_range_last,
                              const Function&        centroids_initializer);

    template <typename SamplesIterator>
    std::vector<DataType> fit(const SamplesIterator& samples_range_first, const SamplesIterator& samples_range_last);

    template <typename SamplesIterator>
    std::vector<DataType> forward(const SamplesIterator& samples_range_first,
                                  const SamplesIterator& samples_range_last) const;

    template <typename SamplesIterator>
    std::vector<std::size_t> predict(const SamplesIterator& samples_range_first,
                                     const SamplesIterator& samples_range_last) const;

  private:
    void prune_unassigned_centroids();  // NOT IMPLEMENTED

    // number of centroids that a KMeans instance should handle (could vary)
    std::size_t n_centroids_;
    // number of features (dimensions) that a KMeans instance should handle
    std::size_t n_features_;
    // n_centroids_ x n_features_ vectorized matrix (n_centroids_ could vary)
    std::vector<DataType> centroids_;

    Options options_;
};

template <typename DataType>
KMeans<DataType>::KMeans(std::size_t n_centroids, std::size_t n_features)
  : n_centroids_{n_centroids}
  , n_features_{n_features} {}

template <typename DataType>
KMeans<DataType>::KMeans(std::size_t n_centroids, std::size_t n_features, const Options& options)
  : n_centroids_{n_centroids}
  , n_features_{n_features}
  , options_{options} {}

template <typename DataType>
KMeans<DataType>::KMeans(std::size_t n_centroids, std::size_t n_features, const std::vector<DataType>& centroids)
  : n_centroids_{n_centroids}
  , n_features_{n_features}
  , centroids_{centroids} {}

template <typename DataType>
KMeans<DataType>::KMeans(std::size_t                  n_centroids,
                         std::size_t                  n_features,
                         const std::vector<DataType>& centroids,
                         const Options&               options)
  : n_centroids_{n_centroids}
  , n_features_{n_features}
  , centroids_{centroids}
  , options_{options} {}

template <typename DataType>
KMeans<DataType>& KMeans<DataType>::set_options(const Options& options) {
    options_ = options;
    return *this;
}

template <typename DataType>
template <template <typename> class KMeansAlgorithm, typename SamplesIterator, typename Function>
std::vector<DataType> KMeans<DataType>::fit(const SamplesIterator& samples_range_first,
                                            const SamplesIterator& samples_range_last,
                                            const Function&        centroids_initializer) {
    // contains the centroids for each tries which number is defined by options_.n_init_ if centroids_ werent already
    // assigned
    auto centroids_candidates = std::vector<std::vector<DataType>>();

    if (centroids_.empty()) {
        for (std::size_t centroid_index = 0; centroid_index < options_.n_init_; ++centroid_index) {
            // default initialization of the centroids if not initialized
            centroids_candidates.emplace_back(
                centroids_initializer(samples_range_first, samples_range_last, n_centroids_, n_features_));
        }
    } else {
        // if the centroids were already assigned, copy them once
        centroids_candidates.emplace_back(centroids_);
    }
    // make the losses buffer for each centroids candidates
    auto candidates_losses = std::vector<DataType>(centroids_candidates.size());

    // This creates a n_candidates vector of n_centroids vectors. The elements of each vector should be initialized to
    // infinity, but since common::are_containers_equal checks container sizes, it's not necessary to do so. We
    // could use only one candidate with a single thread but we make it thread safe this way we dont necessarily need to
    // initialize with vectors of infinities
    auto centroids_candidates_prev = std::vector<std::vector<DataType>>(centroids_candidates.size());

#if defined(_OPENMP) && THREADS_ENABLED == true
#pragma omp parallel for
#endif
    for (std::size_t centroid_index = 0; centroid_index < centroids_candidates.size(); ++centroid_index) {
#if defined(VERBOSE) && VERBOSE == true
        printf("---\nAttempt(%ld/%ld): ", centroid_index + 1, centroids_candidates.size());
#endif
        auto kmeans_algorithm = KMeansAlgorithm(std::make_tuple(samples_range_first, samples_range_last, n_features_),
                                                centroids_candidates[centroid_index]);

        std::size_t patience_iteration = 0;

        for (std::size_t iteration = 0; iteration < options_.max_iter_; ++iteration) {
#if defined(VERBOSE) && VERBOSE == true
            // loss before step to also get the initial loss
            printf("%.3f, ", kmeans_algorithm.total_deviation());
#endif

            centroids_candidates[centroid_index] = kmeans_algorithm.step();

            if (options_.early_stopping_ && common::are_containers_equal(centroids_candidates[centroid_index],
                                                                         centroids_candidates_prev[centroid_index],
                                                                         options_.tolerance_)) {
                if (patience_iteration == options_.patience_) {
                    break;
                }
                ++patience_iteration;

            } else {
                patience_iteration = 0;
            }
            // save the results from the current step
            centroids_candidates_prev[centroid_index] = centroids_candidates[centroid_index];
        }
#if defined(VERBOSE) && VERBOSE == true
        // final loss
        printf("%.3f\n", kmeans_algorithm.total_deviation());
#endif
        // save the loss for each candidate
        candidates_losses[centroid_index] = kmeans_algorithm.total_deviation();
    }
    // find the index of the centroids container with the lowest loss
    const std::size_t min_loss_index =
        common::math::statistics::argmin(candidates_losses.begin(), candidates_losses.end());
    // return best centroids accordingly to the lowest loss
    centroids_ = centroids_candidates[min_loss_index];

    return centroids_;
}

template <typename DataType>
template <template <typename> class KMeansAlgorithm, typename SamplesIterator>
std::vector<DataType> KMeans<DataType>::fit(const SamplesIterator& samples_range_first,
                                            const SamplesIterator& samples_range_last) {
    // execute fit function with a default initialization algorithm
    // kmeansplusplus::make_centroids || common::math::random::init_uniform
    return fit<KMeansAlgorithm>(
        samples_range_first, samples_range_last, kmeansplusplus::make_centroids<SamplesIterator>);
}

template <typename DataType>
template <typename SamplesIterator, typename Function>
std::vector<DataType> KMeans<DataType>::fit(const SamplesIterator& samples_range_first,
                                            const SamplesIterator& samples_range_last,
                                            const Function&        centroids_initializer) {
    // execute fit function with a default initialization algorithm
    // kmeansplusplus::make_centroids || common::math::random::init_uniform
    return fit<Hamerly>(samples_range_first, samples_range_last, centroids_initializer);
}

template <typename DataType>
template <typename SamplesIterator>
std::vector<DataType> KMeans<DataType>::fit(const SamplesIterator& samples_range_first,
                                            const SamplesIterator& samples_range_last) {
    // execute fit function with a default initialization algorithm
    // kmeansplusplus::make_centroids || common::math::random::init_uniform
    return fit<Hamerly>(samples_range_first, samples_range_last, kmeansplusplus::make_centroids<SamplesIterator>);
}

template <typename DataType>
template <typename SamplesIterator>
std::vector<DataType> KMeans<DataType>::forward(const SamplesIterator& samples_range_first,
                                                const SamplesIterator& samples_range_last) const {
    return kmeans::utils::samples_to_nearest_centroid_distances(
        samples_range_first, samples_range_last, n_features_, centroids_);
}

template <typename DataType>
template <typename SamplesIterator>
std::vector<std::size_t> KMeans<DataType>::predict(const SamplesIterator& samples_range_first,
                                                   const SamplesIterator& samples_range_last) const {
    return kmeans::utils::samples_to_nearest_centroid_indices(
        samples_range_first, samples_range_last, n_features_, centroids_);
}

}  // namespace ffcl
