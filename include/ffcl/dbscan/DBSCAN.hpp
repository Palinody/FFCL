#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDTreeIndexed.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace ffcl {

template <typename Indexer>
class DBSCAN {
  public:
    using DataType = typename Indexer::DataType;
    static_assert(std::is_floating_point<DataType>::value, "DBSCAN only allows floating point types.");
    using LabelType = ssize_t;

    struct Options {
        Options() = default;

        Options(const Options& other) = default;

        Options& min_samples(std::size_t min_samples) {
            min_samples_ = min_samples;
            return *this;
        }

        Options& radius(const DataType& radius) {
            radius_ = radius;
            return *this;
        }

        Options& operator=(const Options& options) {
            min_samples_ = options.min_samples_;
            radius_      = options.radius_;
            return *this;
        }

        std::size_t min_samples_ = 5;
        DataType    radius_      = 0.1;
    };

  public:
    DBSCAN() = default;

    DBSCAN(const Options& options);

    DBSCAN(const DBSCAN&) = delete;

    DBSCAN<Indexer>& set_options(const Options& options);

    /**
     * @brief Predicts the cluster label of a range of indices in a global dataset.
     *
     * The client can use a subset of indices in the dataset that needs to be explored, or the entire range of the
     * indices vector. The indices point to samples in the global dataset. The range of indices specifies the portion of
     * the dataset to cluster, which can range from the empty set to [index.begin(), index.end()] or any variation in
     * between.
     *
     * @param indexer the indexer that was used to index the dataset and rearranged the index
     * @return auto std::vector<LabelType> that has the same length as the input index range:
     * std::distance(global_index_first, global_index_last)
     */
    template <typename IndexerFunction, typename... Args>
    auto predict(const Indexer& indexer, IndexerFunction&& indexer_function, Args&&... args) const;

    auto predict(const Indexer& indexer) const;

    template <typename IndexerFunction, typename... Args>
    auto predict_with_buffers(const Indexer& indexer, IndexerFunction&& indexer_function, Args&&... args) const;

    auto predict_with_buffers(const Indexer& indexer) const;

  private:
    template <typename NeighborsIndicesType,
              typename VisitedIndicesType,
              typename PredictionsType,
              typename IndexerFunction,
              typename... Args>
    void predict_inner(NeighborsIndicesType& neighbors_indices,
                       VisitedIndicesType&   visited_indices,
                       PredictionsType&      predictions,
                       const LabelType&      cluster_label,
                       const Indexer&        indexer,
                       IndexerFunction&&     indexer_function,
                       Args&&... args) const;

    enum class SampleStatus : LabelType {
        noise = 0 /*, unknown = 0 */
    };

    Options options_;
};

template <typename Indexer>
DBSCAN<Indexer>::DBSCAN(const Options& options)
  : options_{options} {}

template <typename Indexer>
DBSCAN<Indexer>& DBSCAN<Indexer>::set_options(const Options& options) {
    options_ = options;
    return *this;
}

template <typename Indexer>
template <typename IndexerFunction, typename... Args>
auto DBSCAN<Indexer>::predict(const Indexer& indexer, IndexerFunction&& indexer_function, Args&&... args) const {
    // the query function that should be a member of the indexer
    auto query_function = [&indexer, indexer_function = std::forward<IndexerFunction>(indexer_function)](
                              std::size_t sample_index, auto&&... funcArgs) mutable {
        return std::invoke(indexer_function, indexer, sample_index, std::forward<decltype(funcArgs)>(funcArgs)...);
    };

    // the total number of samples that will be searched by index
    const std::size_t n_samples = indexer.n_samples();

    // vector keeping track of the cluster label for each index specified by the global index range
    auto predictions = std::vector<LabelType>(n_samples);

    // initialize the initial cluster counter that's in [0, n_samples)
    LabelType cluster_label = static_cast<LabelType>(SampleStatus::noise);

    // boolean buffer that keep tracks of the samples that have been already visited
    auto visited_indices = std::make_unique<bool[]>(n_samples);

    // global_index means that it pertains to the entire dataset that has been indexed by the indexer
    for (std::size_t global_index = 0; global_index < n_samples; ++global_index) {
        // process the current sample only if it's not visited
        if (!visited_indices[global_index]) {
            // mark the current sample index as visited
            visited_indices[global_index] = true;
            // the indices of the neighbors in the global dataset with their corresponding distances
            // the query sample is not included
            auto neighbors_buffer = query_function(global_index, std::forward<Args>(args)...);

            if (neighbors_buffer.size() > options_.min_samples_) {
                ++cluster_label;

                predictions[global_index] = cluster_label;

                auto neighbors_indices = neighbors_buffer.extract_indices();

                predict_inner(neighbors_indices,
                              visited_indices,
                              predictions,
                              cluster_label,
                              indexer,
                              indexer_function,
                              std::forward<Args>(args)...);
            } else {
                predictions[global_index] = static_cast<LabelType>(SampleStatus::noise);
            }
        }
    }
    return predictions;
}

template <typename Indexer>
template <typename NeighborsIndicesType,
          typename VisitedIndicesType,
          typename PredictionsType,
          typename IndexerFunction,
          typename... Args>
void DBSCAN<Indexer>::predict_inner(NeighborsIndicesType& neighbors_indices,
                                    VisitedIndicesType&   visited_indices,
                                    PredictionsType&      predictions,
                                    const LabelType&      cluster_label,
                                    const Indexer&        indexer,
                                    IndexerFunction&&     indexer_function,
                                    Args&&... args) const {
    // the query function that should be a member of the indexer
    auto query_function = [&indexer, indexer_function = std::forward<IndexerFunction>(indexer_function)](
                              std::size_t sample_index, auto&&... funcArgs) mutable {
        return std::invoke(indexer_function, indexer, sample_index, std::forward<decltype(funcArgs)>(funcArgs)...);
    };

    // iterate over the samples that are assigned to the current cluster
    for (std::size_t cluster_sample_index = 0; cluster_sample_index < neighbors_indices.size();
         ++cluster_sample_index) {
        const auto neighbor_index = neighbors_indices[cluster_sample_index];
        // enter the condition if the current neighbor index hasnt been visited
        if (!visited_indices[neighbor_index]) {
            // mark the current neighbor index as visited
            visited_indices[neighbor_index] = true;

            // for each neighbor query, retrieve its neighbors
            auto inner_neighbors_buffer = query_function(neighbor_index, std::forward<Args>(args)...);

            if (inner_neighbors_buffer.size() > options_.min_samples_) {
                auto inner_neighbors_indices = inner_neighbors_buffer.extract_indices();

                // iterate over each neighbor's neighbors and add them to the neighbors to visit only if
                // they havent been visited already
                std::copy_if(std::make_move_iterator(inner_neighbors_indices.begin()),
                             std::make_move_iterator(inner_neighbors_indices.end()),
                             std::back_inserter(neighbors_indices),
                             [&visited_indices](const auto& inner_neighbor_index) {
                                 return !visited_indices[inner_neighbor_index];
                             });
            }
        }
        // assign neighbor_index to a cluster if its not already the case
        if (predictions[neighbor_index] == static_cast<LabelType>(SampleStatus::noise)) {
            predictions[neighbor_index] = cluster_label;
        }
    }
}

/*
template <typename Indexer>
template <typename IndexerFunction, typename... Args>
auto DBSCAN<Indexer>::predict(const Indexer& indexer, IndexerFunction&& indexer_function, Args&&... args) const {
    // the query function that should be a member of the indexer
    auto query_function = [&indexer, indexer_function = std::forward<IndexerFunction>(indexer_function)](
                              std::size_t sample_index, auto&&... funcArgs) mutable {
        return std::invoke(indexer_function, indexer, sample_index, std::forward<decltype(funcArgs)>(funcArgs)...);
    };

    // the total number of samples that will be searched by index
    const std::size_t n_samples = indexer.n_samples();

    // vector keeping track of the cluster label for each index specified by the global index range
    auto predictions = std::vector<LabelType>(n_samples);

    // initialize the initial cluster counter that's in [0, n_samples)
    LabelType cluster_label = static_cast<LabelType>(SampleStatus::noise);

    // boolean buffer that keep tracks of the samples that have been already visited
    auto visited_indices = std::make_unique<bool[]>(n_samples);

    // global_index means that it pertains to the entire dataset that has been indexed by the indexer
    for (std::size_t global_index = 0; global_index < n_samples; ++global_index) {
        // process the current sample only if it's not visited
        if (!visited_indices[global_index]) {
            // mark the current sample index as visited
            visited_indices[global_index] = true;
            // the indices of the neighbors in the global dataset with their corresponding distances
            // the query sample is not included
            auto neighbors_buffer = query_function(global_index, std::forward<Args>(args)...);

            if (neighbors_buffer.size() > options_.min_samples_) {
                ++cluster_label;

                predictions[global_index] = cluster_label;

                // iterate over the samples that are assigned to the current cluster
                while (!neighbors_buffer.empty()) {
                    const auto neighbor_index = neighbors_buffer.pop_and_get_index();
                    // enter the condition if the current neighbor index hasnt been visited
                    if (!visited_indices[neighbor_index]) {
                        // mark the current neighbor index as visited
                        visited_indices[neighbor_index] = true;

                        auto inner_neighbors_buffer = query_function(neighbor_index, std::forward<Args>(args)...);

                        if (inner_neighbors_buffer.size() > options_.min_samples_) {
                            neighbors_buffer.add(std::move(inner_neighbors_buffer));
                        }
                    }
                    // assign neighbor_index to a cluster if its not already the case
                    if (predictions[neighbor_index] == static_cast<LabelType>(SampleStatus::noise)) {
                        predictions[neighbor_index] = cluster_label;
                    }
                }
            } else {
                predictions[global_index] = static_cast<LabelType>(SampleStatus::noise);
            }
        }
    }
    return predictions;
}
*/

template <typename Indexer>
auto DBSCAN<Indexer>::predict(const Indexer& indexer) const {
    return predict(indexer, &Indexer::radius_search_around_query_index, options_.radius_);
}

template <typename Indexer>
template <typename IndexerFunction, typename... Args>
auto DBSCAN<Indexer>::predict_with_buffers(const Indexer&    indexer,
                                           IndexerFunction&& indexer_function,
                                           Args&&... args) const {
    // the query function that should be a member of the indexer
    auto query_function = [&indexer, indexer_function = std::forward<IndexerFunction>(indexer_function)](
                              std::size_t sample_index, auto&&... funcArgs) mutable {
        return std::invoke(indexer_function, indexer, sample_index, std::forward<decltype(funcArgs)>(funcArgs)...);
    };

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
        auto current_neighborhood_indices = query_function(global_index, std::forward<Args>(args)...).extract_indices();

        precomputed_is_core[global_index] = current_neighborhood_indices.size() + 1 >= options_.min_samples_;

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

template <typename Indexer>
auto DBSCAN<Indexer>::predict_with_buffers(const Indexer& indexer) const {
    return predict_with_buffers(indexer, &Indexer::radius_search_around_query_index, options_.radius_);
}

}  // namespace ffcl