#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDTreeIndexed.hpp"

#include <cstddef>
#include <tuple>
#include <vector>

namespace ffcl {
template <typename T>
class DBSCAN {
    static_assert(std::is_floating_point<T>::value, "DBSCAN only allows floating point types.");

  public:
    using LabelType = ssize_t;

    enum class SampleStatus : LabelType { visited = -2, noise = -1, unknown = 0 };

    struct Options {
        Options& min_samples_in_radius(std::size_t min_samples_in_radius) {
            min_samples_in_radius_ = min_samples_in_radius;
            return *this;
        }

        Options& radius(const T& radius) {
            radius_ = radius;
            return *this;
        }

        Options& operator=(const Options& options) {
            min_samples_in_radius_ = options.min_samples_in_radius_;
            radius_                = options.radius_;
            return *this;
        }

        std::size_t min_samples_in_radius_ = 5;
        T           radius_                = 0.1;
    };

  public:
    DBSCAN() = default;

    DBSCAN(const Options& options);

    DBSCAN(const DBSCAN&) = delete;

    DBSCAN<T>& set_options(const Options& options);

    /**
     * @brief Predicts the cluster label of a range of indices in a global dataset.
     *
     * The client can use a subset of indices in the dataset that needs to be explored, or the entire range of the
     * indices vector. The indices point to samples in the global dataset. The range of indices specifies the portion of
     * the dataset to cluster, which can range from the empty set to [index.begin(), index.end()] or any variation in
     * between.
     *
     * @tparam Indexer can be a KDTreeIndexed or any other indexer (currently only KDTreeIndexed)
     * @param indexer the indexer that was used to index the dataset and rearranged the index
     * @return auto std::vector<LabelType> that has the same length as the input index range:
     * std::distance(global_index_first, global_index_last)
     */
    template <typename SamplesIterator, typename Indexer>
    auto predict(const SamplesIterator& samples_first,
                 const SamplesIterator& samples_last,
                 const Indexer&         indexer) const;

  private:
    Options options_;
};

template <typename T>
DBSCAN<T>::DBSCAN(const Options& options)
  : options_{options} {}

template <typename T>
DBSCAN<T>& DBSCAN<T>::set_options(const Options& options) {
    options_ = options;
    return *this;
}

template <typename T>
template <typename SamplesIterator, typename Indexer>
auto DBSCAN<T>::predict(const SamplesIterator& samples_first,
                        const SamplesIterator& samples_last,
                        const Indexer&         indexer) const {
    const std::size_t n_features = indexer.n_features();
    const std::size_t n_samples  = common::utils::get_n_samples(samples_first, samples_last, n_features);

    // vector keeping track of the cluster label for each index specified by the global index range
    auto predictions = std::vector<LabelType>(n_samples);

    // initialize the initial cluster counter that's in [0, n_samples). Noise samples will be set to -1
    LabelType cluster_label = static_cast<LabelType>(SampleStatus::unknown);

    // global_index means that it pertains to the entire dataset that has been indexed by the indexer
    for (std::size_t global_index = 0; global_index < n_samples; ++global_index) {
        // process the current sample only if its state is unknown
        if (predictions[global_index] == static_cast<LabelType>(SampleStatus::unknown)) {
            // the indices of the neighbors in the global dataset with their corresponding distances
            // the query sample is included
            auto seed_buffer =
                indexer.radius_search_around_query_sample(samples_first + global_index * n_features,
                                                          samples_first + global_index * n_features + n_features,
                                                          options_.radius_);
            // process only if the current sample query has enough neighbors
            if (seed_buffer.size() >= options_.min_samples_in_radius_) {
                ++cluster_label;

                while (!seed_buffer.empty()) {
                    std::size_t nn_index = seed_buffer.pop_and_get_index();

                    // set the current sample query as the current cluster label
                    predictions[nn_index] = cluster_label;

                    // then find its own neighbors (itself included)
                    auto nn_neighbors_buffer =
                        indexer.radius_search_around_query_sample(samples_first + nn_index * n_features,
                                                                  samples_first + nn_index * n_features + n_features,
                                                                  options_.radius_);

                    if (nn_neighbors_buffer.size() >= options_.min_samples_in_radius_) {
                        // iterate over the neighbors of the current sample
                        while (!nn_neighbors_buffer.empty()) {
                            const auto [nn_neighbors_index, nn_neighbors_distance] =
                                nn_neighbors_buffer.pop_and_get_index_distance_pair();
                            // enter the condition if it hasnt been already assigned to a cluster
                            if (predictions[nn_neighbors_index] == static_cast<LabelType>(SampleStatus::unknown) ||
                                predictions[nn_neighbors_index] == static_cast<LabelType>(SampleStatus::noise)) {
                                // insert the neighbor to the seed set if it isnt already labelled
                                if (predictions[nn_neighbors_index] == static_cast<LabelType>(SampleStatus::unknown)) {
                                    seed_buffer.emplace_back(std::make_pair(nn_neighbors_index, nn_neighbors_distance));
                                }
                                // label it
                                predictions[nn_neighbors_index] = cluster_label;
                            }
                        }
                    }
                }
            } else {
                // set the current sample query as noise if it doesnt have enough neighbors
                predictions[global_index] = static_cast<LabelType>(SampleStatus::noise);
            }
        }
    }
    return predictions;
}

/*
template <typename T>
template <typename SamplesIterator, typename Indexer>
auto DBSCAN<T>::predict(const SamplesIterator& samples_first,
                        const SamplesIterator& samples_last,
                        const Indexer&         indexer) const {
    const std::size_t n_features = indexer.n_features();
    const std::size_t n_samples  = common::utils::get_n_samples(samples_first, samples_last, n_features);

    // vector keeping track of the cluster label for each index specified by the global index range
    auto predictions = std::vector<LabelType>(n_samples);

    // initialize the initial cluster counter that's in [0, n_samples). Noise samples will be set to -1
    LabelType cluster_label = static_cast<LabelType>(SampleStatus::unknown);

    // global_index means that it pertains to the entire dataset that has been indexed by the indexer
    for (std::size_t global_index = 0; global_index < n_samples; ++global_index) {
        // process the current sample only if its state is unknown
        if (predictions[global_index] == static_cast<LabelType>(SampleStatus::unknown)) {
            predictions[global_index] = static_cast<LabelType>(SampleStatus::visited);

            // the indices of the neighbors in the global dataset with their corresponding distances
            // the query sample is included
            auto seed_buffer =
                indexer.radius_search_around_query_sample(samples_first + global_index * n_features,
                                                          samples_first + global_index * n_features + n_features,
                                                          options_.radius_);

            if (seed_buffer.size() < options_.min_samples_in_radius_) {
                predictions[global_index] = static_cast<LabelType>(SampleStatus::noise);
            } else {
                //
            }
        }
    }
    return predictions;
}
*/

}  // namespace ffcl