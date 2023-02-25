#pragma once

#include "cpp_clustering/common/Utils.hpp"
#include "cpp_clustering/heuristics/Heuristics.hpp"
#include "cpp_clustering/kmeans/KMeansPlusPlus.hpp"
#include "cpp_clustering/kmeans/Lloyd.hpp"
#include "cpp_clustering/math/random/Distributions.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

namespace cpp_clustering {

template <typename T>
class KMeans {
    static_assert(std::is_floating_point<T>::value, "KMeans only allows floating point types.");

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

        Options& tolerance(const T& tolerance) {
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
        T           tolerance_      = 0;
        std::size_t n_init_         = 1;
    };

  public:
    KMeans(std::size_t n_centroids, std::size_t n_features);

    KMeans(std::size_t n_centroids, std::size_t n_features, const Options& options);

    KMeans(std::size_t n_centroids, std::size_t n_features, const std::vector<T>& centroids);

    KMeans(std::size_t n_centroids, std::size_t n_features, const std::vector<T>& centroids, const Options& options);

    KMeans(const KMeans&) = delete;

    KMeans<T>& set_options(const Options& options);

    template <typename SamplesIterator, typename Function>
    std::vector<T> fit(const SamplesIterator& data_first,
                       const SamplesIterator& data_last,
                       Function               centroids_initializer);

    template <typename SamplesIterator>
    std::vector<T> fit(const SamplesIterator& data_first, const SamplesIterator& data_last);

    template <typename SamplesIterator>
    std::vector<T> forward(const SamplesIterator& data_first, const SamplesIterator& data_last) const;

    template <typename SamplesIterator>
    std::vector<std::size_t> predict(const SamplesIterator& data_first, const SamplesIterator& data_last) const;

  private:
    // assign each sample (S) to its closest centroid (C)
    template <typename SamplesIterator>
    std::vector<std::size_t> assign(const SamplesIterator& data_first, const SamplesIterator& data_last) const;

    void prune_unassigned_centroids();  // NOT IMPLEMENTED

    // number of centroids that a KMeans instance should handle (could vary)
    std::size_t n_centroids_;
    // number of features (dimensions) that a KMeans instance should handle
    std::size_t n_features_;
    // n_centroids_ x n_features_ vectorized matrix (n_centroids_ could vary)
    std::vector<T> centroids_;

    Options options_;
};

template <typename T>
KMeans<T>::KMeans(std::size_t n_centroids, std::size_t n_features)
  : n_centroids_{n_centroids}
  , n_features_{n_features} {}

template <typename T>
KMeans<T>::KMeans(std::size_t n_centroids, std::size_t n_features, const Options& options)
  : n_centroids_{n_centroids}
  , n_features_{n_features}
  , options_{options} {}

template <typename T>
KMeans<T>::KMeans(std::size_t n_centroids, std::size_t n_features, const std::vector<T>& centroids)
  : n_centroids_{n_centroids}
  , n_features_{n_features}
  , centroids_{centroids} {}

template <typename T>
KMeans<T>::KMeans(std::size_t           n_centroids,
                  std::size_t           n_features,
                  const std::vector<T>& centroids,
                  const Options&        options)
  : n_centroids_{n_centroids}
  , n_features_{n_features}
  , centroids_{centroids}
  , options_{options} {}

template <typename T>
KMeans<T>& KMeans<T>::set_options(const Options& options) {
    options_ = options;
    return *this;
}

template <typename T>
template <typename SamplesIterator>
std::vector<std::size_t> KMeans<T>::assign(const SamplesIterator& data_first, const SamplesIterator& data_last) const {
    const std::size_t n_samples = common::utils::get_n_samples(data_first, data_last, n_features_);

    // keep track of the closest centroid to each sample by index
    auto samples_to_centroids_indices = std::vector<std::size_t>(n_samples);

    for (std::size_t i = 0; i < n_samples; ++i) {
        // distance buffer for a given data sample to each cluster
        auto        shortest_distance  = common::utils::infinity<T>();
        std::size_t min_centroid_index = 0;

        for (std::size_t k = 0; k < n_centroids_; ++k) {
            // sqrt(sum((a_j - b_j)²))
            const T sample_to_centroid_distance = cpp_clustering::heuristic::heuristic(
                /*data sample feature begin=*/data_first + i * n_features_,
                /*data sample feature end=*/data_first + i * n_features_ + n_features_,
                /*centroid sample feature begin=*/centroids_.begin() + k * n_features_);

            if (sample_to_centroid_distance < shortest_distance) {
                shortest_distance  = sample_to_centroid_distance;
                min_centroid_index = k;
            }
        }
        samples_to_centroids_indices[i] = min_centroid_index;
    }
    return samples_to_centroids_indices;
}

template <typename T>
template <typename SamplesIterator, typename Function>
std::vector<T> KMeans<T>::fit(const SamplesIterator& data_first,
                              const SamplesIterator& data_last,
                              Function               centroids_initializer) {
    // contains the centroids for each tries which number is defined by options_.n_init_ if centroids_ werent already
    // assigned
    auto centroids_candidates = std::vector<std::vector<T>>();

    if (centroids_.empty()) {
        for (std::size_t k = 0; k < options_.n_init_; ++k) {
            // default initialization of the centroids if not initialized
            centroids_candidates.emplace_back(centroids_initializer(data_first, data_last, n_centroids_, n_features_));
        }
    } else {
        // if the centroids were already assigned, copy them once
        centroids_candidates.emplace_back(centroids_);
    }
    // make the losses buffer for each centroids candidates
    auto candidates_losses = std::vector<T>(centroids_candidates.size());

    // creates a n_candidates vector of vectors (of n_centroids size with each elements initialized to infinity if we
    // wanted to be precise but common::utils::are_containers_equal checks for containers sizes. So we dont need to do
    // it). We could use only one candidate with a single thread but we make it thread safe this way we dont necessarily
    // need to initialize with vectors of infinities because
    auto centroids_candidates_prev = std::vector<std::vector<T>>(centroids_candidates.size());

    for (std::size_t k = 0; k < centroids_candidates.size(); ++k) {
        // assign the centroids attributes to the current centroids
        centroids_ = centroids_candidates[k];

        auto lloyd = cpp_clustering::Lloyd<SamplesIterator>({data_first, data_last, n_features_}, centroids_);

        std::size_t patience_iter = 0;

        for (std::size_t iter = 0; iter < options_.max_iter_; ++iter) {
#if defined(VERBOSE) && VERBOSE == true
            // loss before step to also get the initial loss
            std::cout << lloyd.total_deviation() << " ";
#endif

            centroids_ = lloyd.step();

            if (options_.early_stopping_ &&
                common::utils::are_containers_equal(centroids_, centroids_candidates_prev[k], options_.tolerance_)) {
                if (patience_iter == options_.patience_) {
                    break;
                }
                ++patience_iter;
            } else {
                patience_iter = 0;
            }
            centroids_candidates_prev[k] = centroids_;
        }
#if defined(VERBOSE) && VERBOSE == true
        // last loss
        std::cout << lloyd.total_deviation() << " ";
        std::cout << "\n";
#endif

        // once the training loop is finished, update the centroids candidate
        centroids_candidates[k] = centroids_;
        // save the loss for each candidate
        candidates_losses[k] = lloyd.total_deviation();
    }
    // find the index of the centroids container with the lowest loss
    const std::size_t min_loss_index = common::utils::argmin(candidates_losses.begin(), candidates_losses.end());
    // return best centroids accordingly to the lowest loss
    centroids_ = centroids_candidates[min_loss_index];
    return centroids_;
}

template <typename T>
template <typename SamplesIterator>
std::vector<T> KMeans<T>::fit(const SamplesIterator& data_first, const SamplesIterator& data_last) {
    // execute fit function with a default initialization algorithm
    // cpp_clustering::kmeansplusplus::make_centroids || common::utils::init_uniform
    return fit(data_first, data_last, common::utils::init_uniform<SamplesIterator>);
}

template <typename T>
template <typename SamplesIterator>
std::vector<T> KMeans<T>::forward(const SamplesIterator& data_first, const SamplesIterator& data_last) const {
    return samples_to_nearest_centroid_distances(data_first, data_last);
}

template <typename T>
template <typename SamplesIterator>
std::vector<std::size_t> KMeans<T>::predict(const SamplesIterator& data_first, const SamplesIterator& data_last) const {
    return assign(data_first, data_last);
}

}  // namespace cpp_clustering
