#pragma once

#include "ffcl/common/Utils.hpp"

#include <cassert>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace ffcl {

template <typename Indexer>
class BoruvkasAlgorithm {
  public:
    using SampleIndexType = std::size_t;
    using SampleValueType = typename Indexer::DataType;

    using ComponentType = std::vector<SampleIndexType>;
    using ForestType    = std::vector<ComponentType>;

    using EdgeType                = std::tuple<SampleIndexType, SampleIndexType, SampleValueType>;
    using MinimumSpanningTreeType = std::vector<EdgeType>;

    struct Options {
        Options() = default;

        Options(const Options& other) = default;

        Options& k_nearest_neighbors(std::size_t k_nearest_neighbors) {
            k_nearest_neighbors_ = k_nearest_neighbors;
            return *this;
        }

        Options& operator=(const Options& options) {
            k_nearest_neighbors_ = options.k_nearest_neighbors_;
            return *this;
        }

        std::size_t k_nearest_neighbors_ = 3;
    };

    class UnionFind {
      public:
        UnionFind(std::size_t n_samples)
          : parents_{std::make_unique<SampleIndexType[]>(n_samples)}
          , ranks_{std::make_unique<SampleIndexType[]>(n_samples)} {
            std::iota(parents_.get(), parents_.get() + n_samples, static_cast<SampleIndexType>(0));
        }

        SampleIndexType find(SampleIndexType index) const {
            while (index != parents_[index]) {
                index = parents_[index];
            }
            return index;
        }

        bool merge(const SampleIndexType& index_1, const SampleIndexType& index_2) {
            const auto parent_1 = find(index_1);
            const auto parent_2 = find(index_2);

            if (parent_1 == parent_2) {
                return false;
            }
            if (ranks_[parent_1] < ranks_[parent_2]) {
                parents_[parent_1] = parent_2;

            } else if (ranks_[parent_1] > ranks_[parent_2]) {
                parents_[parent_2] = parent_1;

            } else {
                parents_[parent_2] = parent_1;
                ++ranks_[parent_1];
            }
            return true;
        }

      private:
        std::unique_ptr<SampleIndexType[]> parents_;
        std::unique_ptr<SampleIndexType[]> ranks_;
    };

    class ForestPartition {
      public:
        ForestPartition(std::size_t n_samples)
          : n_samples_{n_samples}
          , component_labels_{std::vector<SampleIndexType>(n_samples)}
          , sample_indices_{std::vector<SampleIndexType>(n_samples)}
          , component_sizes_{std::vector<SampleIndexType>(n_samples, 1)}
          , component_offsets_{std::vector<SampleIndexType>(n_samples, 0)}
          , sorting_state_{true} {
            std::iota(component_labels_.begin(), component_labels_.end(), static_cast<SampleIndexType>(0));

            std::iota(sample_indices_.begin(), sample_indices_.end(), static_cast<SampleIndexType>(0));

            update_component_offsets();
        }

        SampleIndexType n_components() const {
            return component_sizes_.size();
        }

        auto component_sizes() const {
            return component_sizes_;
        }

        auto component_indices_range(const SampleIndexType& component_index) const {
            return std::make_pair(
                sample_indices_.begin() + component_offsets_[component_index],
                sample_indices_.begin() + component_offsets_[component_index] + component_sizes_[component_index]);
        }

        void update_sample_index_to_component_label(const SampleIndexType& sample_index,
                                                    const SampleIndexType& new_component_label) {
            // decrement the number of samples in the previous component that sample_index was mapped with
            --component_sizes_[component_labels_[sample_index]];
            // remap the sample index to the new component label
            component_labels_[sample_index] = new_component_label;
            // increment the number of samples in the new component that sample_index is now mapped with
            ++component_sizes_[component_labels_[sample_index]];
            // sorting_state is now wrong
            sorting_state_ = false;
        }

        void update() {
            group_sample_indices_by_component();
            update_components();
        }

        void print() const {
            std::cout << "component_label:\n";
            for (const auto& component_label : component_labels_) {
                std::cout << component_label << ", ";
            }
            std::cout << "\n";

            std::cout << "sample_index:\n";
            for (const auto& sample_index : sample_indices_) {
                std::cout << sample_index << ", ";
            }
            std::cout << "\n";

            std::cout << "component_size:\n";
            for (const auto& component_size : component_sizes_) {
                std::cout << component_size << ", ";
            }
            std::cout << "\n";

            std::cout << "component_offset:\n";
            for (const auto& component_offset : component_offsets_) {
                std::cout << component_offset << ", ";
            }
            std::cout << "\n";
        }

      private:
        void group_sample_indices_by_component() {
            std::vector<SampleIndexType> indices(n_samples_);
            std::iota(indices.begin(), indices.end(), static_cast<SampleIndexType>(0));

            auto comparator = [this](const auto& index_1, const auto& index_2) {
                return component_labels_[index_1] < component_labels_[index_2];
            };

            std::sort(indices.begin(), indices.end(), comparator);

            auto sorted_component_labels = std::vector<SampleIndexType>(n_samples_);
            auto sorted_sample_indices   = std::vector<SampleIndexType>(n_samples_);

            for (std::size_t index = 0; index < n_samples_; ++index) {
                sorted_component_labels[index] = component_labels_[indices[index]];
                sorted_sample_indices[index]   = sample_indices_[indices[index]];
            }

            component_labels_ = std::move(sorted_component_labels);
            sample_indices_   = std::move(sorted_sample_indices);
        }

        void prune_component_sizes() {
            component_sizes_.erase(std::remove_if(component_sizes_.begin(),
                                                  component_sizes_.end(),
                                                  [](const auto& component_size) { return !component_size; }),
                                   component_sizes_.end());
        }

        void update_component_offsets() {
            if (component_offsets_.size() != component_sizes_.size()) {
                // reset the vector to a default vector with the new number of elements
                component_offsets_ = decltype(component_offsets_)(component_sizes_.size());
            }
            // recompute the offsets
            std::exclusive_scan(component_sizes_.begin(), component_sizes_.end(), component_offsets_.begin(), 0);
        }

        void update_components() {
            prune_component_sizes();
            update_component_offsets();
        }

        std::size_t n_samples_;
        // the component class/label that can range in [0, n_samples) that partitions the sample indices
        std::vector<SampleIndexType> component_labels_;
        // the sample indices that will be rearranged based on the component labels order
        std::vector<SampleIndexType> sample_indices_;
        // the component size for each label that can range in [0, n_samples)
        std::vector<SampleIndexType> component_sizes_;
        // the cumulated sum of the components sizes to retrieve the beginning of each sequence of component
        std::vector<SampleIndexType> component_offsets_;
        // whether the sample indices are sorted w.r.t. the sorted component indices
        bool sorting_state_;
    };

  public:
    BoruvkasAlgorithm() = default;

    BoruvkasAlgorithm(const Options& options);

    BoruvkasAlgorithm(const BoruvkasAlgorithm&) = delete;

    BoruvkasAlgorithm<Indexer>& set_options(const Options& options);

    template <typename NearestNeighborFunction, typename DistanceFunction, typename... Args>
    auto make_tree(const Indexer&            indexer,
                   NearestNeighborFunction&& indexer_k_nearest_neighbors,
                   DistanceFunction&&        indexer_pairwise_distance,
                   Args&&... args) const;

  private:
    Options options_;

    std::vector<std::vector<SampleValueType>> graph_;
};

template <typename Indexer>
BoruvkasAlgorithm<Indexer>::BoruvkasAlgorithm(const Options& options)
  : options_{options} {}

template <typename Indexer>
BoruvkasAlgorithm<Indexer>& BoruvkasAlgorithm<Indexer>::set_options(const Options& options) {
    options_ = options;
    return *this;
}

template <typename Indexer>
template <typename NearestNeighborFunction, typename DistanceFunction, typename... Args>
auto BoruvkasAlgorithm<Indexer>::make_tree(const Indexer&            indexer,
                                           NearestNeighborFunction&& indexer_k_nearest_neighbors,
                                           DistanceFunction&&        indexer_pairwise_distance,
                                           Args&&... args) const {
    // the indexer's function used to find the nearest vertex of a vertex query
    auto k_nearest_neighbors_lambda =
        [&indexer, indexer_k_nearest_neighbors = std::forward<NearestNeighborFunction>(indexer_k_nearest_neighbors)](
            std::size_t vertex_index, std::size_t n_neighbors = 1) mutable {
            return std::invoke(/**/ indexer_k_nearest_neighbors,
                               /**/ indexer,
                               /**/ vertex_index,
                               /**/ n_neighbors);
        };

    // the indexer's function used to evaluate the edge value (distance) between 2 vertices
    auto pairwise_distance_lambda =
        [&indexer, indexer_pairwise_distance = std::forward<DistanceFunction>(indexer_pairwise_distance)](
            /**/ std::size_t vertex_index1,
            /**/ std::size_t vertex_index2,
            /**/ std::size_t k_nearest_neighbors) mutable {
            return std::invoke(/**/ indexer_pairwise_distance,
                               /**/ indexer,
                               /**/ vertex_index1,
                               /**/ vertex_index2,
                               /**/ k_nearest_neighbors);
        };

    common::utils::ignore_parameters(k_nearest_neighbors_lambda, pairwise_distance_lambda, args...);

    ForestPartition forest(indexer.n_samples());

    // the minimum spanning tree starts with no edge
    MinimumSpanningTreeType mst;
    mst.reserve(indexer.n_samples() - 1);

    // populate the forest with the initial components
    for (std::size_t sample_index = 0; sample_index < indexer.n_samples(); ++sample_index) {
        // each initial component starts as a singleton containing a sample index
        forest.emplace_back(ComponentType{sample_index});
    }

    auto sample_index_to_component_index = std::make_unique<SampleIndexType[]>(indexer.n_samples());

    // ForestPartition forest_partition(indexer.n_samples());

    while (forest.size() > 1) {
        std::cout << forest.size() << "\n";
        // keeps track of the shortest edge thats been found w.r.t. each component
        auto closest_edges = std::make_unique<EdgeType[]>(forest.size());
        // for each component (tree) in the graph (forest), find the closest edge
        for (std::size_t component_index = 0; component_index < forest.size(); ++component_index) {
            const auto& component = forest[component_index];
            // initialize the closest edge from the current comonent to infinity
            closest_edges[component_index] = EdgeType{0, 0, common::utils::infinity<SampleValueType>()};
            // for each vertex of the current component, find its nearest neighbor thats not part of the same component
            for (const auto& sample_index : component) {
                // update the component index of the current sample index query
                sample_index_to_component_index[sample_index] = component_index;

                auto nn_buffer_with_memory =
                    NearestNeighborsBufferWithMemory<typename std::vector<SampleValueType>::iterator>(component, 1);

                // get a buffer of k nearest neighbors that are not part of the same component as the query
                indexer.buffered_k_nearest_neighbors_around_query_index(sample_index, nn_buffer_with_memory);

                const auto nearest_neighbor_index    = nn_buffer_with_memory.furthest_k_nearest_neighbor_index();
                const auto nearest_neighbor_distance = nn_buffer_with_memory.furthest_k_nearest_neighbor_distance();

                if (nearest_neighbor_distance < std::get<2>(closest_edges[component_index])) {
                    closest_edges[component_index] =
                        EdgeType{sample_index, nearest_neighbor_index, nearest_neighbor_distance};
                }
            }
        }
    }
    return mst;
}

}  // namespace ffcl