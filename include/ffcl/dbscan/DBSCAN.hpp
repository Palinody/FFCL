#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDTreeIndexed.hpp"

#include <cstddef>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace ffcl {
template <typename T>
class DBSCAN {
    static_assert(std::is_floating_point<T>::value, "DBSCAN only allows floating point types.");

  public:
    using LabelType = ssize_t;

    enum class SampleStatus : LabelType {
        noise              = -1,
        unknown            = 0,
        reachable          = 1,
        directly_reachable = 2,
        core_point         = 3
    };

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
    template <typename Indexer>
    auto predict(const Indexer& indexer) const;

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
template <typename Indexer>
auto DBSCAN<T>::predict(const Indexer& indexer) const {
    const std::size_t n_samples = indexer.n_samples();

    // the current state of the samples. Default: SampleStatus::unknown
    // auto samples_state = std::vector<SampleStatus>(n_samples);

    // vector keeping track of the cluster label for each index specified by the global index range
    auto predictions = std::vector<LabelType>(n_samples);
    // initialize the initial cluster counter that's in [0, n_samples). Noise samples will be set to -1
    LabelType cluster_label = 1;
    for (std::size_t global_index = 0; global_index < n_samples; ++global_index) {
        // process the current sample only if its state is unknown
        if (predictions[global_index] == static_cast<LabelType>(SampleStatus::unknown)) {
            // the indices of the neighbors in the global dataset with their corresponding distances
            // assumes that the query sample is NOT RETURNED from the query function
            const auto [nn_indices, _1] = indexer.radius_search_around_query_index(global_index, options_.radius_);
            // process only if the current sample query has enough neighbors
            // N.B.: +1 because the query function assumes that the query isnt returned
            if (nn_indices.size() + 1 > options_.min_samples_in_radius_) {
                // set the current sample query as the current cluster label
                predictions[global_index] = cluster_label;
                // all points in seeds are density-reachable from the sample query
                for (const auto& nn_index : nn_indices) {
                    predictions[nn_index] = cluster_label;
                }
                // initialize the seed set with the neighbors indices that are around the initial sample query index
                std::unordered_set<std::size_t> seed_set(nn_indices.begin(), nn_indices.end());

                while (!seed_set.empty()) {
                    const auto nn_index = *seed_set.begin();
                    seed_set.erase(seed_set.begin());

                    // then find its own neighbors
                    const auto [nn_neighbors_indices, _2] =
                        indexer.radius_search_around_query_index(nn_index, options_.radius_);

                    if (nn_neighbors_indices.size() + 1 > options_.min_samples_in_radius_) {
                        //
                        for (const auto& nn_neighbors_index : nn_neighbors_indices) {
                            //
                            if (predictions[nn_neighbors_index] == static_cast<LabelType>(SampleStatus::unknown) ||
                                predictions[nn_neighbors_index] == static_cast<LabelType>(SampleStatus::noise)) {
                                //
                                if (predictions[nn_neighbors_index] == static_cast<LabelType>(SampleStatus::unknown)) {
                                    seed_set.insert(nn_neighbors_index);
                                }
                                predictions[nn_neighbors_index] = cluster_label;
                            }
                        }
                    }
                    // printf("Seed set size: %ld\n", global_index);
                }
                ++cluster_label;
            } else {
                // set the current sample query as noise if it doesnt have enough neighbors
                predictions[global_index] = static_cast<LabelType>(SampleStatus::noise);
            }
        }
    }
    return predictions;
}

}  // namespace ffcl