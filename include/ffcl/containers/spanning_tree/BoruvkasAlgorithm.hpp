#pragma once

#include "ffcl/common/Utils.hpp"

#include <cstddef>
#include <functional>
#include <map>
#include <unordered_set>
#include <vector>

namespace ffcl {

template <typename Indexer>
class BoruvkasAlgorithm {
  public:
    using DataType = typename Indexer::DataType;

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

    template <typename NearestNeighborFunction, typename DistanceFunction, typename... Args>
    auto make_tree_v1(const Indexer&            indexer,
                      NearestNeighborFunction&& indexer_k_nearest_neighbors,
                      DistanceFunction&&        indexer_pairwise_distance,
                      Args&&... args) const;

  private:
    Options options_;

    std::vector<std::vector<DataType>> graph_;
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

    using IndexType  = std::size_t;
    using WeightType = DataType;
    // an edge is represented by the connection between two vertices and their distance
    using EdgeType = std::tuple<IndexType, WeightType>;
    // each node component of the minimum spanning tree. The index of the node in the mst represents the source node and
    // the elements of the node the branches/vertices
    using NodeType                = std::vector<EdgeType>;
    using MinimumSpanningTreeType = std::vector<NodeType>;
    MinimumSpanningTreeType mst(indexer.n_samples());

    // a tree/component is the set of all vertices that form a cluster
    using ComponentType = std::vector<IndexType>;
    // a forest is the set of all trees/components
    using ForestType = std::vector<ComponentType>;
    ForestType forest;
    forest.reserve(indexer.n_samples());

    // populate the forest where each sample is its own cluster
    for (std::size_t sample_index = 0; sample_index < indexer.n_samples(); ++sample_index) {
        // each initial cluster starts with the first edge containing a vertex and itself as a parent.
        // Its distance with itself is obviously zero.
        forest.emplace_back(ComponentType{sample_index});
    }

    // Query each point for its nearest neighbor that is not within the component
    // for each component, add the smallest edge
    std::size_t counter = 0;
    while (forest.size() > 1) {
        // iterate over each components and find the best {Vi ∈ Si, Vj ∈ Sj} pair to connect the components
        for (std::size_t component_index = 0; component_index < forest.size(); ++component_index) {
            // the vertex of the current component that resulted to the closest distance with the best other vertex
            std::size_t best_vertex_index = 0;
            // the component that was found to be the closest to the current vertex outside of its own component
            std::size_t best_other_component_index = 0;
            // the best vertex index found in the other component
            std::size_t best_other_vertex_index = 0;
            // initial distance set to infinity to allow the closest neighbor from another component to be updated
            auto closest_distance = common::utils::infinity<DataType>();

            // iterate over each vertices from a component
            for (std::size_t vertex_index = 0; vertex_index < forest[component_index].size(); ++vertex_index) {
                // select the current vertex from the current component
                const auto sample_index_query = forest[component_index][vertex_index];

                // iterate over each other components
                for (std::size_t other_component_index = 0; other_component_index < forest.size();
                     ++other_component_index) {
                    if (component_index != other_component_index) {
                        // visit all the vertices from the other component
                        for (std::size_t other_vertex_index = 0;
                             other_vertex_index < forest[other_component_index].size();
                             ++other_vertex_index) {
                            const auto sample_index_candidate = forest[other_component_index][other_vertex_index];

                            const auto pairwise_distance = pairwise_distance_lambda(
                                sample_index_query, sample_index_candidate, options_.k_nearest_neighbors_);

                            if (pairwise_distance < closest_distance) {
                                best_vertex_index          = vertex_index;
                                best_other_component_index = other_component_index;
                                best_other_vertex_index    = other_vertex_index;
                                closest_distance           = pairwise_distance;
                            }
                        }
                    }
                }
            }

            // WEVE DONE ONLY ONE COMPONENT SO FAR

            // add the edge representing:
            // 1) the best sample index in the current version of the component
            // 2) the best sample index in the current version of the other component where the nearest neighbor was
            // found
            // 3) the weight value that was computed between both values: closest_distance

            // 1)
            const auto best_sample_index = forest[component_index][best_vertex_index];
            // 2)
            const auto best_other_sample_index = forest[best_other_component_index][best_other_vertex_index];
            mst[best_sample_index].emplace_back(EdgeType{best_other_sample_index, closest_distance});

            // merge the components
            std::move(forest[best_other_component_index].begin(),
                      forest[best_other_component_index].end(),
                      std::back_inserter(forest[component_index]));
            // erase the original location of the other best component
            forest.erase(forest.begin() + best_other_component_index);
            std::cout << forest.size() << ", ";
        }
        std::cout << "\n" << counter++ << "\n";
    }
    // TODO:
    // dst.insert(dst.end(), std::make_move_iterator(src.begin()), std::make_move_iterator(src.end()));
    // or:
    // std::move(src.begin(), src.end(), std::back_inserter(dst));
    common::utils::ignore_parameters(k_nearest_neighbors_lambda, args...);

    return mst;
}

template <typename Indexer>
template <typename NearestNeighborFunction, typename DistanceFunction, typename... Args>
auto BoruvkasAlgorithm<Indexer>::make_tree_v1(const Indexer&            indexer,
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

    using SampleIndexType = std::size_t;
    using LinkType        = std::pair<SampleIndexType, SampleIndexType>;
    using ComponentType   = std::vector<LinkType>;
    using ForestType      = std::vector<ComponentType>;
    ForestType forest;
    forest.reserve(indexer.n_samples());

    // the set of each vertex connected to other vertices with their respective distance
    // the algorithm starts with no edge
    std::map<SampleIndexType, std::vector<std::pair<SampleIndexType, DataType>>> edges_set;

    // populate the forest with each sample as its own cluster
    for (std::size_t sample_index = 0; sample_index < indexer.n_samples(); ++sample_index) {
        // each initial cluster starts with the first edge containing a vertex linked with itself.
        // Its distance with itself is obviously zero.
        forest.emplace_back(ComponentType{LinkType{sample_index, sample_index}});
    }

    while (forest.size() > 1) {
        for (std::size_t component_index = 0; component_index < forest.size(); ++component_index) {
            //
        }
    }
    return edges_set;
}

}  // namespace ffcl