#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/spanning_tree/MinimumSpanningTree.hpp"

#include "ffcl/datastruct/UnionFind.hpp"

#include "ffcl/search/buffer/WithUnionFind.hpp"

#include "ffcl/search/Search.hpp"

#include <cassert>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <tuple>
#include <type_traits>  // std::remove_const_t
#include <unordered_set>
#include <vector>

#include <algorithm>
#include <execution>

namespace ffcl {

template <typename Indexer>
class BoruvkasAlgorithm {
  public:
    using IndexType           = typename Indexer::IndexType;
    using ValueType           = typename Indexer::DataType;
    using IndicesIteratorType = typename Indexer::IndicesIteratorType;
    using SamplesIteratorType = typename Indexer::SamplesIteratorType;

    using EdgeType = mst::Edge<IndexType, ValueType>;

    using CoreDistancesArray    = std::unique_ptr<ValueType[]>;
    using CoreDistancesArrayPtr = std::shared_ptr<CoreDistancesArray>;

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
        using ComponentType = std::unordered_set<IndexType>;

        using ForestType = std::map<IndexType, ComponentType>;

        using UnionFindType = datastruct::UnionFind<IndexType>;

        using MinimumSpanningTreeType = mst::MinimumSpanningTree<IndexType, ValueType>;

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

        std::size_t n_components() const {
            return std::distance(components_.begin(), components_.end());
        }

        const auto& get_union_find_const_reference() {
            return union_find_;
        }

        constexpr auto begin() {
            return components_.begin();
        }

        constexpr auto end() {
            return components_.end();
        }

        constexpr auto begin() const {
            return components_.begin();
        }

        constexpr auto end() const {
            return components_.end();
        }

        constexpr auto cbegin() const {
            return components_.cbegin();
        }

        constexpr auto cend() const {
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
            components_[final_component].insert(std::make_move_iterator(components_[discarded_component].cbegin()),
                                                std::make_move_iterator(components_[discarded_component].cend()));

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
        // the container mapping each component representative to the set of actual sample indices
        ForestType components_;
        // a union find data structure used to merge clusters based on sample indices from distinct clusters
        UnionFindType union_find_;
    };

  public:
    BoruvkasAlgorithm() = default;

    BoruvkasAlgorithm(const Options& options);

    BoruvkasAlgorithm(const BoruvkasAlgorithm&) = delete;

    BoruvkasAlgorithm<Indexer>& set_options(const Options& options);

    auto make_tree(Indexer&& indexer) const;

  private:
    auto make_core_distances_ptr(search::Searcher<Indexer>& searcher, const IndexType& k_nearest_neighbors) const;

    void step_sequential(search::Searcher<Indexer>& searcher, Forest& forest) const;

    void step_sequential(search::Searcher<Indexer>&   searcher,
                         const CoreDistancesArrayPtr& core_distances,
                         Forest&                      forest) const;

    void dual_component_step_sequential(search::Searcher<Indexer>& searcher, Forest& forest) const;

    void dual_component_step_sequential(search::Searcher<Indexer>&   searcher,
                                        const CoreDistancesArrayPtr& core_distances,
                                        Forest&                      forest) const;

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
auto BoruvkasAlgorithm<Indexer>::make_core_distances_ptr(search::Searcher<Indexer>& searcher,
                                                         const IndexType&           k_nearest_neighbors) const {
    auto core_distances_ptr = std::make_shared<CoreDistancesArray>(std::make_unique<ValueType[]>(searcher.n_samples()));

    for (std::size_t sample_index = 0; sample_index < searcher.n_samples(); ++sample_index) {
        auto nn_buffer_query = search::buffer::StaticUnsorted(searcher.features_range_first(sample_index),
                                                              searcher.features_range_last(sample_index),
                                                              k_nearest_neighbors);

        (*core_distances_ptr)[sample_index] = searcher(std::move(nn_buffer_query)).upper_bound();
    }
    return core_distances_ptr;
}

template <typename Indexer>
void BoruvkasAlgorithm<Indexer>::step_sequential(search::Searcher<Indexer>& searcher, Forest& forest) const {
    // keep track of the shortest edge from a component's sample index to a sample index thats not within the
    // same component
    auto components_closest_edge = std::map<IndexType, EdgeType>();

    for (const auto& [component_representative, component] : forest) {
        // initialize the closest edge from the current component to infinity
        components_closest_edge[component_representative] =
            EdgeType{common::infinity<IndexType>(), common::infinity<IndexType>(), common::infinity<ValueType>()};

        for (const auto& sample_index : component) {
            // initialize a nearest neighbor buffer to compare the sample_index with sample indices that don't belong to
            // the same component using the UnionFind data structure
            auto nn_buffer_query =
                searcher(search::buffer::StaticWithUnionFind(searcher.features_range_first(sample_index),
                                                             searcher.features_range_last(sample_index),
                                                             forest.get_union_find_const_reference(),
                                                             component_representative,
                                                             /*max_capacity=*/static_cast<IndexType>(1)));

            // the furthest nearest neighbor is also the closest in this case since we query only 1 neighbor
            const auto nearest_neighbor_index    = nn_buffer_query.upper_bound_index();
            const auto nearest_neighbor_distance = nn_buffer_query.upper_bound();

            const auto current_closest_edge_distance = std::get<2>(components_closest_edge[component_representative]);

            // update the current shortest edge if the nearest_neighbor_distance is indeed shortest than the current
            // shortest edge distance
            if (nearest_neighbor_distance < current_closest_edge_distance) {
                components_closest_edge[component_representative] =
                    EdgeType{sample_index, nearest_neighbor_index, nearest_neighbor_distance};
            }
        }
    }
    // merge components based on the best edges found in each component so far
    for (const auto& [component_representative, edge] : components_closest_edge) {
        assert(std::get<2>(edge) < common::infinity<ValueType>());
        common::ignore_parameters(component_representative);
        forest.merge_components(edge);
    }
}

template <typename Indexer>
void BoruvkasAlgorithm<Indexer>::step_sequential(search::Searcher<Indexer>&   searcher,
                                                 const CoreDistancesArrayPtr& core_distances,
                                                 Forest&                      forest) const {
    // keep track of the shortest edge from a component's sample index to a sample index thats not within the
    // same component
    auto components_closest_edge = std::map<IndexType, EdgeType>();

    for (const auto& [component_representative, component] : forest) {
        // initialize the closest edge from the current component to infinity
        components_closest_edge[component_representative] =
            EdgeType{common::infinity<IndexType>(), common::infinity<IndexType>(), common::infinity<ValueType>()};

        for (const auto& sample_index : component) {
            // initialize a nearest neighbor buffer to compare the sample_index with sample indices that don't belong to
            // the same component using the UnionFind data structure
            auto nn_buffer_query =
                searcher(search::buffer::StaticWithUnionFind(searcher.features_range_first(sample_index),
                                                             searcher.features_range_last(sample_index),
                                                             forest.get_union_find_const_reference(),
                                                             component_representative,
                                                             /*max_capacity=*/static_cast<IndexType>(1)));

            // the furthest nearest neighbor is also the closest in this case since we query only 1 neighbor
            const auto nearest_neighbor_index    = nn_buffer_query.upper_bound_index();
            const auto nearest_neighbor_distance = nn_buffer_query.upper_bound();

            const auto current_closest_edge_distance = std::get<2>(components_closest_edge[component_representative]);

            const auto k_mutual_reachability_distance = std::max({(*core_distances)[sample_index],
                                                                  (*core_distances)[nearest_neighbor_index],
                                                                  nearest_neighbor_distance});

            // update the current shortest edge if the k_mutual_reachability_distance is indeed shortest than the
            // current shortest edge distance
            if (k_mutual_reachability_distance < current_closest_edge_distance) {
                components_closest_edge[component_representative] =
                    EdgeType{sample_index, nearest_neighbor_index, k_mutual_reachability_distance};
            }
        }
    }
    // merge components based on the best edges found in each component so far
    for (const auto& [component_representative, edge] : components_closest_edge) {
        assert(std::get<2>(edge) < common::infinity<ValueType>());
        common::ignore_parameters(component_representative);
        forest.merge_components(edge);
    }
}

template <typename Indexer>
void BoruvkasAlgorithm<Indexer>::dual_component_step_sequential(search::Searcher<Indexer>& searcher,
                                                                Forest&                    forest) const {
    using ComponentRepresentativeType = std::remove_const_t<decltype(forest.begin()->first)>;
    using ComponentType               = std::remove_const_t<decltype(forest.begin()->second)>;

    auto smallest_component_representative = common::infinity<ComponentRepresentativeType>();
    auto smallest_component                = ComponentType{};
    auto current_smallest_component_size   = common::infinity<IndexType>();

    for (const auto& [component_representative, component] : forest) {
        if (component.size() < current_smallest_component_size) {
            smallest_component_representative = component_representative;
            smallest_component                = std::move(component);
            current_smallest_component_size   = component.size();
        }
    }
    // initialize the closest edge from the current component to infinity
    auto closest_edge =
        EdgeType{common::infinity<IndexType>(), common::infinity<IndexType>(), common::infinity<ValueType>()};

    for (const auto& sample_index : smallest_component) {
        // initialize a nearest neighbor buffer to compare the sample_index with sample indices that don't belong to
        // the same component using the UnionFind data structure
        auto nn_buffer_query =
            searcher(search::buffer::StaticWithUnionFind(searcher.features_range_first(sample_index),
                                                         searcher.features_range_last(sample_index),
                                                         forest.get_union_find_const_reference(),
                                                         smallest_component_representative,
                                                         /*max_capacity=*/static_cast<IndexType>(1)));

        // the furthest nearest neighbor is also the closest in this case since we query only 1 neighbor
        const auto nearest_neighbor_index    = nn_buffer_query.upper_bound_index();
        const auto nearest_neighbor_distance = nn_buffer_query.upper_bound();

        const auto current_closest_edge_distance = std::get<2>(closest_edge);

        // update the current shortest edge if the nearest_neighbor_distance is indeed shortest than the current
        // shortest edge distance
        if (nearest_neighbor_distance < current_closest_edge_distance) {
            closest_edge = EdgeType{sample_index, nearest_neighbor_index, nearest_neighbor_distance};
        }
    }
    // merge components based on the best edge found
    forest.merge_components(closest_edge);
}

template <typename Indexer>
void BoruvkasAlgorithm<Indexer>::dual_component_step_sequential(search::Searcher<Indexer>&   searcher,
                                                                const CoreDistancesArrayPtr& core_distances,
                                                                Forest&                      forest) const {
    using ComponentRepresentativeType = std::remove_const_t<decltype(forest.begin()->first)>;
    using ComponentType               = std::remove_const_t<decltype(forest.begin()->second)>;

    auto smallest_component_representative = common::infinity<ComponentRepresentativeType>();
    auto smallest_component                = ComponentType{};
    auto current_smallest_component_size   = common::infinity<IndexType>();

    for (const auto& [component_representative, component] : forest) {
        if (component.size() < current_smallest_component_size) {
            smallest_component_representative = component_representative;
            smallest_component                = std::move(component);
            current_smallest_component_size   = component.size();
        }
    }
    // initialize the closest edge from the current component to infinity
    auto closest_edge =
        EdgeType{common::infinity<IndexType>(), common::infinity<IndexType>(), common::infinity<ValueType>()};

    for (const auto& sample_index : smallest_component) {
        // initialize a nearest neighbor buffer to compare the sample_index with sample indices that don't belong to
        // the same component using the UnionFind data structure
        auto nn_buffer_query =
            searcher(search::buffer::StaticWithUnionFind(searcher.features_range_first(sample_index),
                                                         searcher.features_range_last(sample_index),
                                                         forest.get_union_find_const_reference(),
                                                         smallest_component_representative,
                                                         /*max_capacity=*/static_cast<IndexType>(1)));

        // the furthest nearest neighbor is also the closest in this case since we query only 1 neighbor
        const auto nearest_neighbor_index    = nn_buffer_query.upper_bound_index();
        const auto nearest_neighbor_distance = nn_buffer_query.upper_bound();

        const auto current_closest_edge_distance = std::get<2>(closest_edge);

        const auto k_mutual_reachability_distance = std::max(
            {(*core_distances)[sample_index], (*core_distances)[nearest_neighbor_index], nearest_neighbor_distance});

        // update the current shortest edge if the k_mutual_reachability_distance is indeed shortest than the current
        // shortest edge distance
        if (k_mutual_reachability_distance < current_closest_edge_distance) {
            closest_edge = EdgeType{sample_index, nearest_neighbor_index, k_mutual_reachability_distance};
        }
    }
    // merge components based on the best edge found
    forest.merge_components(closest_edge);
}

template <typename Indexer>
auto BoruvkasAlgorithm<Indexer>::make_tree(Indexer&& indexer) const {
    Forest forest(indexer.n_samples());

    auto searcher = search::Searcher(std::forward<Indexer>(indexer));

    // compute the core distances only if knn > 1 -> k_nearest_reachability_distance is activated
    const auto core_distances =
        options_.k_nearest_neighbors_ > 1 ? make_core_distances_ptr(searcher, options_.k_nearest_neighbors_) : nullptr;

    while (forest.n_components() > 1) {
        if (core_distances) {
            if (forest.n_components() == 2) {
                dual_component_step_sequential(searcher, core_distances, forest);

            } else {
                step_sequential(searcher, core_distances, forest);
            }
        } else {
            if (forest.n_components() == 2) {
                dual_component_step_sequential(searcher, forest);

            } else {
                step_sequential(searcher, forest);
            }
        }
    }
    return forest.minimum_spanning_tree();
}

}  // namespace ffcl