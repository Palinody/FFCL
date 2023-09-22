#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/containers/spanning_tree/MinimumSpanningTree.hpp"

#include "ffcl/containers/UnionFind.hpp"

#include <cassert>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <algorithm>
#include <execution>

namespace ffcl {

template <typename Indexer>
class BoruvkasAlgorithm {
  public:
    using IndexType = std::size_t;
    using ValueType = typename Indexer::DataType;

    using ComponentType = std::unordered_set<IndexType>;
    using ForestType    = std::map<IndexType, ComponentType>;

    using EdgeType                = mst::Edge<IndexType, ValueType>;
    using MinimumSpanningTreeType = mst::MinimumSpanningTree<IndexType, ValueType>;

    using UnionFindType = UnionFind<IndexType>;

#if defined(_OPENMP) && THREADS_ENABLED == true
    static constexpr auto execution_policy = std::execution::par;
#else
    static constexpr auto execution_policy = std::execution::seq;
#endif

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

    class Forest {
      public:
        using Iterator      = typename ForestType::iterator;
        using ConstIterator = typename ForestType::const_iterator;

        Forest(std::size_t n_samples)
          : n_samples_{n_samples}
          , minimum_spanning_tree_{}
          , components_{}
          , union_find_{UnionFindType(n_samples_)} {
            minimum_spanning_tree_.reserve(n_samples_ - 1);

            for (std::size_t sample_index = 0; sample_index < n_samples_; ++sample_index) {
                components_[sample_index] = ComponentType{sample_index};
            }
        }

        auto n_elements() const {
            return n_samples_;
        }

        std::size_t n_components() const {
            return std::distance(components_.begin(), components_.end());
        }

        const auto& get_union_find_const_reference() {
            return union_find_;
        }

        Iterator begin() {
            return components_.begin();
        }

        Iterator end() {
            return components_.end();
        }

        ConstIterator begin() const {
            return components_.begin();
        }

        ConstIterator end() const {
            return components_.end();
        }

        ConstIterator cbegin() const {
            return components_.cbegin();
        }

        ConstIterator cend() const {
            return components_.cend();
        }

        auto minimum_spanning_tree() const {
            return minimum_spanning_tree_;
        }

        void merge_components(const EdgeType& edge) {
            // get the indices of the samples that form an edge
            const auto sample_index_1 = std::get<0>(edge);
            const auto sample_index_2 = std::get<1>(edge);

            // get which components they belong to
            const auto component_label_1 = union_find_.find(sample_index_1);
            const auto component_label_2 = union_find_.find(sample_index_2);

            // return if both samples belong to the same component
            if (component_label_1 == component_label_2) {
                return;
            }
            // theres nothing to merge if the components is empty
            if (components_[component_label_1].empty()) {
                components_.erase(component_label_1);
                return;
            }
            // theres nothing to merge if the components is empty
            if (components_[component_label_2].empty()) {
                components_.erase(component_label_2);
                return;
            }
            // merge the samples in the union find datastructure
            union_find_.merge(sample_index_1, sample_index_2);

            // find the component that will be the final one and the discarded one based on the union_find structure
            const auto [final_component, discarded_component] =
                (component_label_1 == union_find_.find(sample_index_1))
                    ? std::make_pair(component_label_1, component_label_2)
                    : std::make_pair(component_label_2, component_label_1);

            // move the indices from the component that will be discarded to the final one
            components_[final_component].insert(std::make_move_iterator(components_[discarded_component].begin()),
                                                std::make_move_iterator(components_[discarded_component].end()));

            // now that the old component has been merged with the final one, clear it
            components_.erase(discarded_component);

            // update the minimum spanning tree
            minimum_spanning_tree_.emplace_back(edge);
        }

        void print() const {
            std::cout << "components:\n";
            for (const auto& [component_index, component] : components_) {
                std::cout << component_index << ": ";

                for (const auto& sample_index : component) {
                    std::cout << sample_index << ", ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";

            std::cout << "Minimum Spanning Tree (MST):\n";
            for (const auto& edge : minimum_spanning_tree_) {
                std::cout << "(" << std::get<0>(edge) << ", " << std::get<1>(edge) << ", " << std::get<2>(edge)
                          << "), \n";
            }
            std::cout << "\n";
        }

      private:
        std::size_t n_samples_;
        // the container that accumulates the edges for the minimum spanning tree
        MinimumSpanningTreeType minimum_spanning_tree_;
        // the vector containing each components, represented as vectors of indices
        ForestType components_;
        // a union find data structure used to merge clusters based on sample indices from distinct clusters
        UnionFindType union_find_;
    };

  public:
    BoruvkasAlgorithm() = default;

    BoruvkasAlgorithm(const Options& options);

    BoruvkasAlgorithm(const BoruvkasAlgorithm&) = delete;

    BoruvkasAlgorithm<Indexer>& set_options(const Options& options);

    auto make_tree(const Indexer& indexer) const;

  private:
    auto make_core_distances(const Indexer& indexer, std::size_t k_nearest_neighbors) const;

    Options options_;
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
auto BoruvkasAlgorithm<Indexer>::make_core_distances(const Indexer& indexer, std::size_t k_nearest_neighbors) const {
    auto core_distances = std::make_unique<ValueType[]>(indexer.n_samples());

    for (IndexType sample_index = 0; sample_index < indexer.n_samples(); ++sample_index) {
        core_distances[sample_index] = indexer.k_nearest_neighbors_around_query_index(sample_index, k_nearest_neighbors)
                                           .furthest_k_nearest_neighbor_distance();
    }
    return core_distances;
}

template <typename Indexer>
auto BoruvkasAlgorithm<Indexer>::make_tree(const Indexer& indexer) const {
    Forest forest(indexer.n_samples());

    const bool compute_knn_reachability_distance = options_.k_nearest_neighbors_ > 1;

    // compute the core distances only if knn > 1 -> k_nearest_reachability_distance is activated
    const auto core_distances =
        compute_knn_reachability_distance ? make_core_distances(indexer, options_.k_nearest_neighbors_) : nullptr;

    while (forest.n_components() > 1) {
        // std::cout << "forest.n_components(): " << forest.n_components() << "\n";
        // keep track of the shortest edge from a component's sample index to a sample index thats not within the
        // same component
        auto closest_edges = std::map<IndexType, EdgeType>();

        std::for_each(execution_policy, forest.begin(), forest.end(), [&](const auto& forest_item) {
            const auto& [component_index, component] = forest_item;

            // initialize the closest edge from the current component to infinity
            closest_edges[component_index] = EdgeType{common::utils::infinity<IndexType>(),
                                                      common::utils::infinity<IndexType>(),
                                                      common::utils::infinity<ValueType>()};

            for (const auto& sample_index : component) {
                // initialize a nearest neighbor buffer to compare the sample_index with other sample indices from
                // other components using the UnionFind data structure
                auto nn_buffer = NearestNeighborsBufferWithUnionFind<typename std::vector<ValueType>::iterator>(
                    forest.get_union_find_const_reference(), sample_index, 1);

                indexer.buffered_k_nearest_neighbors_around_query_index(sample_index, nn_buffer);

                // the nearest neighbor buffer with memory might not find any nearest neighbor if all the candidates
                // are already within the same component
                if (!nn_buffer.empty()) {
                    // the furthest nearest neighbor is also the closest in this case since we query only 1 neighbor
                    const auto nearest_neighbor_index    = nn_buffer.furthest_k_nearest_neighbor_index();
                    const auto nearest_neighbor_distance = nn_buffer.furthest_k_nearest_neighbor_distance();

                    const auto distance =
                        compute_knn_reachability_distance
                            ? std::max(std::max(core_distances[sample_index], core_distances[nearest_neighbor_index]),
                                       nearest_neighbor_distance)
                            : nearest_neighbor_distance;

                    if (distance < std::get<2>(closest_edges[component_index])) {
                        closest_edges[component_index] = EdgeType{sample_index, nearest_neighbor_index, distance};
                    }
                }
            }
        });
        // merge components based on the best edges found in each component so far
        std::for_each(
            execution_policy, closest_edges.begin(), closest_edges.end(), [&](const auto& closest_edges_item) {
                const auto& [component_index, edge] = closest_edges_item;
                assert(std::get<2>(edge) < common::utils::infinity<ValueType>());
                common::utils::ignore_parameters(component_index);
                forest.merge_components(edge);
            });
    }
    return forest.minimum_spanning_tree();
}

}  // namespace ffcl