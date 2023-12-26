#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/Unsorted.hpp"

#include "ffcl/datastruct/bounds/Ball.hpp"

#include "ffcl/search/Search.hpp"

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

    using LabelType = std::size_t;

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
    auto predict(Indexer&& indexer) const;

  private:
    template <typename NeighborsIndicesType, typename VisitedIndicesType, typename PredictionsType>
    void predict_inner(NeighborsIndicesType&      neighbors_indices,
                       VisitedIndicesType&        visited_indices,
                       PredictionsType&           predictions,
                       const LabelType&           cluster_label,
                       search::Searcher<Indexer>& searcher) const;

    enum class SampleStatus : LabelType { noise = 0 };

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
auto DBSCAN<Indexer>::predict(Indexer&& indexer) const {
    auto searcher = search::Searcher(std::forward<Indexer>(indexer));

    // the total number of samples that will be searched by index
    const std::size_t n_samples = searcher.n_samples();

    // vector keeping track of the cluster label for each index specified by the global index range
    auto predictions = std::vector<LabelType>(n_samples);

    // initialize the initial cluster counter that's in [0, n_samples)
    LabelType cluster_label = static_cast<LabelType>(SampleStatus::noise);

    // boolean buffer that keep tracks of the samples that have been already visited
    auto visited_indices = std::make_unique<bool[]>(n_samples);

    // entry_point_candidate_index is the first sample index that is a candidate to initiate a cluster
    for (std::size_t entry_point_candidate_index = 0; entry_point_candidate_index < n_samples;
         ++entry_point_candidate_index) {
        // process the current sample only if it's not visited
        if (!visited_indices[entry_point_candidate_index]) {
            // mark the current sample index as visited
            visited_indices[entry_point_candidate_index] = true;

            auto ball_view = datastruct::bounds::BallView(searcher.features_range_first(entry_point_candidate_index),
                                                          searcher.features_range_last(entry_point_candidate_index),
                                                          options_.radius_);

            auto nn_buffer_query = searcher(search::buffer::Unsorted(std::move(ball_view)));

            if (nn_buffer_query.size() > options_.min_samples_) {
                ++cluster_label;

                predictions[entry_point_candidate_index] = cluster_label;

                auto neighbors_indices = std::move(nn_buffer_query).indices();

                predict_inner(neighbors_indices, visited_indices, predictions, cluster_label, searcher);
            } else {
                predictions[entry_point_candidate_index] = static_cast<LabelType>(SampleStatus::noise);
            }
        }
    }
    return predictions;
}

template <typename Indexer>
template <typename NeighborsIndicesType, typename VisitedIndicesType, typename PredictionsType>
void DBSCAN<Indexer>::predict_inner(NeighborsIndicesType&      neighbors_indices,
                                    VisitedIndicesType&        visited_indices,
                                    PredictionsType&           predictions,
                                    const LabelType&           cluster_label,
                                    search::Searcher<Indexer>& searcher) const {
    // iterate over the samples that are assigned to the current cluster
    for (std::size_t cluster_sample_index = 0; cluster_sample_index < neighbors_indices.size();
         ++cluster_sample_index) {
        const auto neighbor_index = neighbors_indices[cluster_sample_index];
        // enter the condition if the current neighbor index hasnt been visited
        if (!visited_indices[neighbor_index]) {
            // mark the current neighbor index as visited
            visited_indices[neighbor_index] = true;

            auto ball_view = datastruct::bounds::BallView(searcher.features_range_first(neighbor_index),
                                                          searcher.features_range_last(neighbor_index),
                                                          options_.radius_);

            auto inner_neighbors_buffer = searcher(search::buffer::Unsorted(std::move(ball_view)));

            if (inner_neighbors_buffer.size() > options_.min_samples_) {
                auto inner_neighbors_indices = std::move(inner_neighbors_buffer).indices();

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

}  // namespace ffcl