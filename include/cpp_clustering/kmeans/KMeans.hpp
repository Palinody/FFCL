#pragma once

#include "cpp_clustering/common/Utils.hpp"
#include "cpp_clustering/heuristics/Heuristics.hpp"
#include "cpp_clustering/math/random/Distributions.hpp"

#include "cpp_clustering/kmeans/KMeansPlusPlus.hpp"

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
    // loss: sum of samples distances from each centroid
    template <typename SamplesIterator>
    std::vector<T> distances_by_centroid(const SamplesIterator& data_first, const SamplesIterator& data_last) const;

    template <typename SamplesIterator>
    T loss(const SamplesIterator& data_first, const SamplesIterator& data_last) const;

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
    // assign each sample (S) to its closest centroid (C)
    template <typename SamplesIterator>
    std::vector<std::size_t> assign(const SamplesIterator& data_first, const SamplesIterator& data_last) const;

    template <typename SamplesIterator>
    std::vector<T> samples_to_nearest_centroid_distances(const SamplesIterator& data_first,
                                                         const SamplesIterator& data_last) const;

    // update the position of the centroids: C = mean(S(c))
    template <typename SamplesIterator>
    void update(const SamplesIterator&          data_first,
                const SamplesIterator&          data_last,
                const std::vector<std::size_t>& samples_to_closest_centroid_indices);

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
std::vector<T> KMeans<T>::distances_by_centroid(const SamplesIterator& data_first,
                                                const SamplesIterator& data_last) const {
    const std::size_t n_samples = common::utils::get_n_samples(data_first, data_last, n_features_);

    // the number of samples that have been associated to a centroid
    auto n_samples_by_centroid = std::vector<std::size_t>(n_centroids_);
    // sum of squared errors (SSE) for each cluster center (C could vary)
    auto centroids_sum_of_distances = std::vector<T>(n_centroids_);

    for (std::size_t i = 0; i < n_samples; ++i) {
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
        // increment the sample counter for the closest centroid
        ++n_samples_by_centroid[min_centroid_index];
        // sum the euclidean distances for the closest centroid
        centroids_sum_of_distances[min_centroid_index] += shortest_distance;
    }
    return centroids_sum_of_distances;
}

template <typename T>
template <typename SamplesIterator>
T KMeans<T>::loss(const SamplesIterator& data_first, const SamplesIterator& data_last) const {
    const auto distances = samples_to_nearest_centroid_distances(data_first, data_last);
    return std::accumulate(distances.begin(), distances.end(), 0) / static_cast<T>(distances.size());
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
template <typename SamplesIterator>
std::vector<T> KMeans<T>::samples_to_nearest_centroid_distances(const SamplesIterator& data_first,
                                                                const SamplesIterator& data_last) const {
    const std::size_t n_samples = common::utils::get_n_samples(data_first, data_last, n_features_);

    // keep track of the closest centroid to each sample by index
    auto samples_to_closest_centroid_distance = std::vector<T>(n_samples);

    for (std::size_t i = 0; i < n_samples; ++i) {
        auto shortest_distance = common::utils::infinity<T>();

        for (std::size_t k = 0; k < n_centroids_; ++k) {
            const auto sample_to_centroid_distance = cpp_clustering::heuristic::heuristic(
                /*data sample begin=*/data_first + i * n_features_,
                /*data sample end=*/data_first + i * n_features_ + n_features_,
                /*centroid sample feature begin=*/centroids_.begin() + k * n_features_);

            shortest_distance = std::min(sample_to_centroid_distance, shortest_distance);
        }
        samples_to_closest_centroid_distance[i] = shortest_distance;
    }
    return samples_to_closest_centroid_distance;
}

template <typename T>
template <typename SamplesIterator>
void KMeans<T>::update(const SamplesIterator&          data_first,
                       const SamplesIterator&          data_last,
                       const std::vector<std::size_t>& samples_to_closest_centroid_indices) {
    const std::size_t n_samples = common::utils::get_n_samples(data_first, data_last, n_features_);

    // the number of samples that have been associated to a centroid
    std::vector<std::size_t> n_assigned_samples_by_centroid(n_centroids_);
    // make a container to accumulate and update the new cluster positions
    // if a cluster k has n_assigned_samples_by_centroid[k] == 0
    // then the previous corresponding centroid position is copied
    auto new_centroids = std::vector<T>(n_centroids_ * n_features_);

    for (std::size_t i = 0; i < n_samples; ++i) {
        const auto curr_centroid_idx = samples_to_closest_centroid_indices[i];
        // increment the sample counter for the closest centroid
        ++n_assigned_samples_by_centroid[curr_centroid_idx];
        // select the correct centroid to accumulate sample values
        std::transform(
            /*data_first feature of k'th centroid*/ new_centroids.begin() + curr_centroid_idx * n_features_,
            /*data_last feature of k'th centroid*/ new_centroids.begin() + curr_centroid_idx * n_features_ +
                n_features_,
            /*data_first feature of current sample*/ data_first + i * n_features_,
            /*accumulate*/ new_centroids.begin() + curr_centroid_idx * n_features_,
            std::plus<>());
    }
    // kf: k(th) centroid and f(th) feature (vectorized matrix)
    for (std::size_t kf = 0; kf < new_centroids.size(); ++kf) {
        if (n_assigned_samples_by_centroid[kf / n_features_] > 0) {
            centroids_[kf] = new_centroids[kf] / static_cast<T>(n_assigned_samples_by_centroid[kf / n_features_]);
        }
    }
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

        std::size_t patience_iter = 0;

        for (std::size_t iter = 0; iter < options_.max_iter_; ++iter) {
            const auto samples_to_closest_centroid_indices = assign(data_first, data_last);
            update(data_first, data_last, samples_to_closest_centroid_indices);

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
        // once the training loop is finished, update the centroids candidate
        centroids_candidates[k] = centroids_;
        // save the loss for each candidate
        candidates_losses[k] = this->loss(data_first, data_last);
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

template <typename T>
template <typename SamplesIterator, typename LabelsIterator>
std::pair<std::size_t, std::vector<T>> KMeans<T>::swap_to_best_count_match(const SamplesIterator& samples_first,
                                                                           const SamplesIterator& samples_last,
                                                                           LabelsIterator         labels_first,
                                                                           LabelsIterator         labels_last) {
    const std::size_t n_labels        = std::distance(labels_first, labels_last);
    auto              centroids_order = std::vector<std::size_t>(n_centroids_);

    // Initial order is in increasing order (no swap)
    std::iota(centroids_order.begin(), centroids_order.end(), 0);

    auto predicted_labels = predict(samples_first, samples_last);
    // initialize the best match count with the current order
    std::size_t best_match_count =
        common::utils::count_matches(predicted_labels.begin(), predicted_labels.end(), labels_first);

    // if the first best math count matches all the labels then theres nothing to swap
    if (best_match_count == n_labels) {
        return {best_match_count, centroids_};
    }
    // keep a record of the best bedoid order
    auto best_centroids_order = centroids_order;
    // save a copy of the non swapped, original medoids
    const auto centroids_orig = centroids_;

    while (std::next_permutation(centroids_order.begin(), centroids_order.end())) {
        // swap the medoids inplace
        centroids_ = common::utils::range_permutation_from_indices(centroids_orig, centroids_order, n_features_);
        // update the preedicted labels based on the current medoids ordering
        predicted_labels = predict(samples_first, samples_last);
        // recompute the match counter
        const std::size_t match_count_candidate =
            common::utils::count_matches(predicted_labels.begin(), predicted_labels.end(), labels_first);

        if (match_count_candidate > best_match_count) {
            best_match_count     = match_count_candidate;
            best_centroids_order = centroids_order;

            // if the count matches all the labels then we are sure that theres no best swap
            if (match_count_candidate == n_labels) {
                return {best_match_count, centroids_};
            }
        }
    }
    // finally save the best centroids order
    centroids_ = common::utils::range_permutation_from_indices(centroids_orig, best_centroids_order, n_features_);
    // return the best match count with its associated reordered centroids
    return {best_match_count, centroids_};
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
    std::cout << std::endl;
}

template <typename T>
template <typename SamplesIterator, typename LabelsIterator>
std::pair<std::size_t, std::vector<T>> KMeans<T>::remap_centroid_to_label_index(const SamplesIterator& samples_first,
                                                                                const SamplesIterator& samples_last,
                                                                                LabelsIterator         labels_first,
                                                                                LabelsIterator         labels_last,
                                                                                std::size_t            n_classes) {
    // find the max element in the labels iterator if its not provided. Add one because the max element is an index
    n_classes = n_classes ? n_classes : 1 + *std::max_element(labels_first, labels_last);

    auto centroids_order = std::vector<std::size_t>(n_centroids_);
    std::iota(centroids_order.begin(), centroids_order.end(), 0);
    // buffer the scores obtained with each centroid for each class
    auto centroid_to_class_counts =
        std::vector<std::vector<std::size_t>>(n_centroids_, std::vector<std::size_t>(n_classes));
    // save a copy of the non swapped, original medoids
    const auto centroids_orig = centroids_;

    for (std::size_t k = 0; k < n_centroids_; ++k) {
        for (std::size_t c = 0; c < n_classes; ++c) {
            // swap centroid k with class label c
            std::swap(centroids_order[k], centroids_order[c]);
            // perform swap on the centroids k and c
            centroids_ = common::utils::range_permutation_from_indices(centroids_orig, centroids_order, n_features_);

            auto predicted_labels = predict(samples_first, samples_last);
            // count matches for centroid k when we move it to cluster c with the same class index k
            const auto best_match_count = common::utils::count_matches_for_value(
                predicted_labels.begin(), predicted_labels.end(), labels_first, /*value=*/k);

            centroid_to_class_counts[k][c] = best_match_count;
            // reswap centroids k and c to their original index
            std::swap(centroids_order[k], centroids_order[c]);
        }
    }
    print_matrix(centroid_to_class_counts);

    std::size_t best_match_count = 0;
    for (std::size_t k = 0; k < n_centroids_; ++k) {
        const auto class_counts = centroid_to_class_counts[k];
        // get the index for the current centroid that maximizes its score for a class c
        const auto [max_index, max_value] =
            common::utils::get_max_index_value_pair(class_counts.begin(), class_counts.end());

        centroids_order[k] = max_index;
        best_match_count += max_value;
    }
    std::cout << "centroids order after\n";
    print(centroids_order);
    // finally save the best centroids order
    centroids_ = common::utils::range_permutation_from_indices(centroids_orig, centroids_order, n_features_);

    // return the best match count with its associated reordered centroids
    return {best_match_count, centroids_};
}

}  // namespace cpp_clustering
