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

    template <typename Indexer>
    auto predict_with_buffers(const Indexer& indexer) const;

  private:
    enum class SampleStatus : LabelType { noise = 0 /*, unknown = 0 */ };

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

    // vector keeping track of the cluster label for each index specified by the global index range
    auto predictions = std::vector<LabelType>(n_samples);

    // initialize the initial cluster counter that's in [0, n_samples)
    LabelType cluster_label = static_cast<LabelType>(SampleStatus::noise);

    std::vector<std::size_t> initial_neighbors_indices;

    // global_index means that it pertains to the entire dataset that has been indexed by the indexer
    for (std::size_t global_index = 0; global_index < n_samples; ++global_index) {
        // process the current sample only if it's not visited
        if (predictions[global_index] == static_cast<LabelType>(SampleStatus::noise)) {
            // the indices of the neighbors in the global dataset with their corresponding distances
            // the query sample is not included
            auto initial_neighbors_buffer = indexer.radius_search_around_query_index(global_index, options_.radius_);

            if (initial_neighbors_buffer.size() + 1 >= options_.min_samples_in_radius_) {
                ++cluster_label;

                predictions[global_index] = cluster_label;

                initial_neighbors_indices = initial_neighbors_buffer.extract_indices();

                // iterate over the samples that are assigned to the current cluster
                for (std::size_t cluster_sample_index = 0; cluster_sample_index < initial_neighbors_indices.size();
                     ++cluster_sample_index) {
                    const auto neighbor_index = initial_neighbors_indices[cluster_sample_index];
                    if (predictions[neighbor_index] == static_cast<LabelType>(SampleStatus::noise)) {
                        predictions[neighbor_index] = cluster_label;

                        auto current_neighbors_buffer =
                            indexer.radius_search_around_query_index(neighbor_index, options_.radius_);

                        if (current_neighbors_buffer.size() + 1 >= options_.min_samples_in_radius_) {
                            auto current_neighbors_indices = current_neighbors_buffer.extract_indices();

                            initial_neighbors_indices.insert(initial_neighbors_indices.end(),
                                                             std::make_move_iterator(current_neighbors_indices.begin()),
                                                             std::make_move_iterator(current_neighbors_indices.end()));
                        }
                    }
                }
            }
        }
    }
    return predictions;
}

template <typename T>
template <typename Indexer>
auto DBSCAN<T>::predict_with_buffers(const Indexer& indexer) const {
    const std::size_t n_samples = indexer.n_samples();

    // vector keeping track of the cluster label for each index specified by the global index range
    auto predictions = std::vector<LabelType>(n_samples);

    // initialize the initial cluster counter that's in [0, n_samples). Noise samples will be set to -1
    LabelType cluster_label = static_cast<LabelType>(1);

    // maps each sample to its nearest neighbors w.r.t. radius and min samples in radius options
    auto precomputed_neighborhood = std::vector<std::vector<std::size_t>>(n_samples);
    // booloean that assesses whether the current sample is core or non core
    auto precomputed_is_core = std::vector<bool>(n_samples);

    for (std::size_t global_index = 0; global_index < n_samples; ++global_index) {
        auto current_neighborhood_indices =
            indexer.radius_search_around_query_index(global_index, options_.radius_).extract_indices();

        precomputed_is_core[global_index] = current_neighborhood_indices.size() + 1 >= options_.min_samples_in_radius_;

        precomputed_neighborhood[global_index] = std::move(current_neighborhood_indices);
    }

    auto neighbors_stack = std::vector<std::size_t>();

    for (std::size_t global_index = 0; global_index < n_samples; ++global_index) {
        if (predictions[global_index] == static_cast<LabelType>(SampleStatus::noise) &&
            precomputed_is_core[global_index]) {
            while (true) {
                if (predictions[global_index] == static_cast<LabelType>(SampleStatus::noise)) {
                    predictions[global_index] = cluster_label;

                    if (precomputed_is_core[global_index]) {
                        for (const auto& neighbor_index : precomputed_neighborhood[global_index]) {
                            if (predictions[neighbor_index] == static_cast<LabelType>(SampleStatus::noise)) {
                                neighbors_stack.emplace_back(neighbor_index);
                            }
                        }
                    }
                }
                if (neighbors_stack.empty()) {
                    break;
                }
                global_index = neighbors_stack.back();
                neighbors_stack.pop_back();
            }
            ++cluster_label;
        }
    }
    return predictions;
}

}  // namespace ffcl