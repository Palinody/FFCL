#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/graph/spanning_tree/MinimumSpanningTree.hpp"

#include "ffcl/datastruct/UnionFind.hpp"

#include "ffcl/search/buffer/WithUnionFind.hpp"

#include "ffcl/datastruct/graph/spanning_tree/CoreDistances.hpp"
#include "ffcl/search/Search.hpp"

#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <algorithm>
#include <execution>

namespace ffcl {

template <typename Indexer>
class BoruvkasAlgorithm {
  public:
    using IndexType = typename Indexer::IndexType;
    using ValueType = typename Indexer::DataType;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<ValueType>, "ValueType must be trivial.");

    using IndicesIteratorType = typename Indexer::IndicesIteratorType;
    using SamplesIteratorType = typename Indexer::SamplesIteratorType;

    static_assert(common::is_iterator<IndicesIteratorType>::value, "IndicesIteratorType is not an iterator");
    static_assert(common::is_iterator<SamplesIteratorType>::value, "SamplesIteratorType is not an iterator");

    using EdgeType = datastruct::mst::Edge<IndexType, ValueType>;

    using CoreDistancesArrayType = datastruct::mst::CoreDistancesArray<ValueType>;

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

    BoruvkasAlgorithm() = default;

    BoruvkasAlgorithm(const Options& options);

    BoruvkasAlgorithm(const BoruvkasAlgorithm&) = delete;

    BoruvkasAlgorithm<Indexer>& set_options(const Options& options);

    template <typename ForwardedIndexer>
    auto make_tree(ForwardedIndexer&& indexer) const;

    template <typename ForwardedIndexer>
    auto make_tree_2(ForwardedIndexer&& indexer) const;

  private:
    class Forest {
      public:
        using RepresentativeType = IndexType;

        using ComponentType = std::vector<RepresentativeType>;

        using RepresentativeToComponentMapType = std::unordered_map<RepresentativeType, ComponentType>;

        using UnionFindType = datastruct::UnionFind<RepresentativeType>;

        using MinimumSpanningTreeType = datastruct::mst::EdgesList<RepresentativeType, ValueType>;

        Forest(std::size_t n_samples)
          : minimum_spanning_tree_{}
          , representatives_to_components_map_{}
          , union_find_{UnionFindType(n_samples)} {
            minimum_spanning_tree_.reserve(n_samples - 1);
            // each sample starts as its own component representative
            for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
                representatives_to_components_map_[sample_index] = ComponentType{sample_index};
            }
        }

        std::size_t n_components() const {
            return std::distance(representatives_to_components_map_.begin(), representatives_to_components_map_.end());
        }

        const auto& get_union_find_const_reference() const {
            return union_find_;
        }

        constexpr auto begin() {
            return representatives_to_components_map_.begin();
        }

        constexpr auto end() {
            return representatives_to_components_map_.end();
        }

        constexpr auto begin() const {
            return representatives_to_components_map_.begin();
        }

        constexpr auto end() const {
            return representatives_to_components_map_.end();
        }

        constexpr auto cbegin() const {
            return representatives_to_components_map_.cbegin();
        }

        constexpr auto cend() const {
            return representatives_to_components_map_.cend();
        }

        auto&& minimum_spanning_tree() && {
            return std::move(minimum_spanning_tree_);
        }

        void merge_components(const EdgeType& edge) {
            // get the indices of the samples that form an edge
            const auto sample_index_1 = std::get<0>(edge);
            const auto sample_index_2 = std::get<1>(edge);

            // get which component belongs to which representative before the merge
            const auto representative_1 = union_find_.find(sample_index_1);
            const auto representative_2 = union_find_.find(sample_index_2);

            // return if both samples belong to the same component
            if (representative_1 == representative_2) {
                return;
            }
            // merge the sets based on the 2 samples and return the common representative of the newly formed set
            const auto common_representative = union_find_.merge(sample_index_1, sample_index_2);

            // Determine which component will be retained and which will be discarded.
            // The Union-Find structure selects the final representative based on rank comparison,
            // ensuring that the component with the smaller size is the one that gets moved.
            // This decision is made by comparing the representatives before the merge to the common representative
            // determined by UnionFind after the merge.
            const auto [retained_representative, discarded_representative] =
                (representative_1 == common_representative) ? std::make_pair(representative_1, representative_2)
                                                            : std::make_pair(representative_2, representative_1);

            // move the indices from the component that will be discarded to the final one
            representatives_to_components_map_[retained_representative].insert(
                representatives_to_components_map_[retained_representative].end(),
                std::make_move_iterator(representatives_to_components_map_[discarded_representative].cbegin()),
                std::make_move_iterator(representatives_to_components_map_[discarded_representative].cend()));

            // now that the old component has been merged with the final one, clear it
            representatives_to_components_map_.erase(discarded_representative);

            // update the minimum spanning tree
            minimum_spanning_tree_.emplace_back(edge);
        }

        void print() const {
            std::cout << "components:\n";
            for (const auto& [component_index, component] : representatives_to_components_map_) {
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

        // the container that accumulates the edges for the minimum spanning tree
        MinimumSpanningTreeType minimum_spanning_tree_;
        // the container mapping each component representative to the set of actual sample indices
        RepresentativeToComponentMapType representatives_to_components_map_;
        // a union find data structure used to merge clusters based on sample indices from distinct clusters
        UnionFindType union_find_;
    };

    void step_sequential(const search::Searcher<Indexer>& searcher, Forest& forest) const;

    void step_sequential(const search::Searcher<Indexer>& searcher,
                         const CoreDistancesArrayType&    core_distances,
                         Forest&                          forest) const;

    void dual_component_step_sequential(const search::Searcher<Indexer>& searcher, Forest& forest) const;

    void dual_component_step_sequential(const search::Searcher<Indexer>& searcher,
                                        const CoreDistancesArrayType&    core_distances,
                                        Forest&                          forest) const;

    void step_dual_tree_sequential(const search::Searcher<Indexer>& searcher,
                                   const CoreDistancesArrayType&    core_distances,
                                   Forest&                          forest) const;

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
void BoruvkasAlgorithm<Indexer>::step_sequential(const search::Searcher<Indexer>& searcher, Forest& forest) const {
    // keep track of the shortest edge from a component's sample index to a sample index thats not within the
    // same component
    auto components_closest_edge = std::unordered_map<IndexType, EdgeType>{};

    for (const auto& [component_representative, component] : forest) {
        // initialize the closest edge from the current component to infinity
        components_closest_edge[component_representative] = datastruct::mst::make_default_edge<IndexType, ValueType>();

        for (const auto& query_index : component) {
            // initialize a nearest neighbor buffer to compare the query_index with sample indices that don't belong to
            // the same component using the UnionFind data structure
            auto nn_buffer_query = searcher(search::buffer::WithUnionFind(searcher.features_range_first(query_index),
                                                                          searcher.features_range_last(query_index),
                                                                          forest.get_union_find_const_reference(),
                                                                          component_representative,
                                                                          /*max_capacity=*/static_cast<IndexType>(1)));

            // the furthest nearest neighbor is also the closest in this case since we query only 1 neighbor
            const auto nearest_neighbor_index    = nn_buffer_query.furthest_index();
            const auto nearest_neighbor_distance = nn_buffer_query.furthest_distance();

            const auto current_closest_edge_distance = std::get<2>(components_closest_edge[component_representative]);

            // update the current shortest edge if the nearest_neighbor_distance is indeed shortest than the current
            // shortest edge distance
            if (nearest_neighbor_distance < current_closest_edge_distance) {
                components_closest_edge[component_representative] =
                    EdgeType{query_index, nearest_neighbor_index, nearest_neighbor_distance};
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
void BoruvkasAlgorithm<Indexer>::step_sequential(const search::Searcher<Indexer>& searcher,
                                                 const CoreDistancesArrayType&    core_distances,
                                                 Forest&                          forest) const {
    // keep track of the shortest edge from a component's sample index to a sample index thats not within the
    // same component
    auto components_closest_edge = std::unordered_map<IndexType, EdgeType>{};

    for (const auto& [component_representative, component] : forest) {
        // initialize the closest edge from the current component to infinity
        components_closest_edge[component_representative] = datastruct::mst::make_default_edge<IndexType, ValueType>();

        for (const auto& query_index : component) {
            // initialize a nearest neighbor buffer to compare the query_index with sample indices that don't belong to
            // the same component using the UnionFind data structure
            auto nn_buffer_query = searcher(search::buffer::WithUnionFind(searcher.features_range_first(query_index),
                                                                          searcher.features_range_last(query_index),
                                                                          forest.get_union_find_const_reference(),
                                                                          component_representative,
                                                                          /*max_capacity=*/static_cast<IndexType>(1)));

            // the furthest nearest neighbor is also the closest in this case since we query only 1 neighbor
            const auto nearest_neighbor_index    = nn_buffer_query.furthest_index();
            const auto nearest_neighbor_distance = nn_buffer_query.furthest_distance();

            const auto current_closest_edge_distance = std::get<2>(components_closest_edge[component_representative]);

            const auto k_mutual_reachability_distance = std::max(
                {core_distances[query_index], core_distances[nearest_neighbor_index], nearest_neighbor_distance});

            // update the current shortest edge if the k_mutual_reachability_distance is indeed shortest than the
            // current shortest edge distance
            if (k_mutual_reachability_distance < current_closest_edge_distance) {
                components_closest_edge[component_representative] =
                    EdgeType{query_index, nearest_neighbor_index, k_mutual_reachability_distance};
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
void BoruvkasAlgorithm<Indexer>::dual_component_step_sequential(const search::Searcher<Indexer>& searcher,
                                                                Forest&                          forest) const {
    // Lambda to find the component_representative and smallest_component pair at the smallest component size
    auto find_pair_at_smallest_component = [](const auto& key_value_iterable) {
        auto found_it = key_value_iterable.begin();

        for (auto it = key_value_iterable.begin(); it != key_value_iterable.end(); ++it) {
            if (it->second.size() < found_it->second.size()) {
                found_it = it;
            }
        }
        // Returns a key-value pair that contains the smallest value.
        return *found_it;
    };
    const auto& [smallest_component_representative, smallest_component] = find_pair_at_smallest_component(forest);

    // initialize the closest edge from the current component to infinity
    auto closest_edge = datastruct::mst::make_default_edge<IndexType, ValueType>();

    for (const auto& query_index : smallest_component) {
        // initialize a nearest neighbor buffer to compare the query_index with sample indices that don't belong to
        // the same component using the UnionFind data structure
        auto nn_buffer_query = searcher(search::buffer::WithUnionFind(searcher.features_range_first(query_index),
                                                                      searcher.features_range_last(query_index),
                                                                      forest.get_union_find_const_reference(),
                                                                      smallest_component_representative,
                                                                      /*max_capacity=*/static_cast<IndexType>(1)));

        // the furthest nearest neighbor is also the closest in this case since we query only 1 neighbor
        const auto nearest_neighbor_index    = nn_buffer_query.furthest_index();
        const auto nearest_neighbor_distance = nn_buffer_query.furthest_distance();

        const auto current_closest_edge_distance = std::get<2>(closest_edge);

        // update the current shortest edge if the nearest_neighbor_distance is indeed shortest than the current
        // shortest edge distance
        if (nearest_neighbor_distance < current_closest_edge_distance) {
            closest_edge = EdgeType{query_index, nearest_neighbor_index, nearest_neighbor_distance};
        }
    }
    // merge components based on the best edge found
    forest.merge_components(closest_edge);
}

template <typename Indexer>
void BoruvkasAlgorithm<Indexer>::dual_component_step_sequential(const search::Searcher<Indexer>& searcher,
                                                                const CoreDistancesArrayType&    core_distances,
                                                                Forest&                          forest) const {
    // Lambda to find the component_representative and smallest_component pair at the smallest component size
    auto find_pair_at_smallest_component = [](const auto& key_value_iterable) {
        auto found_it = key_value_iterable.begin();

        for (auto it = key_value_iterable.begin(); it != key_value_iterable.end(); ++it) {
            if (it->second.size() < found_it->second.size()) {
                found_it = it;
            }
        }
        // Returns a key-value pair that contains the smallest value.
        return *found_it;
    };
    const auto& [smallest_component_representative, smallest_component] = find_pair_at_smallest_component(forest);

    // initialize the closest edge from the current component to infinity
    auto closest_edge = datastruct::mst::make_default_edge<IndexType, ValueType>();

    for (const auto& query_index : smallest_component) {
        // initialize a nearest neighbor buffer to compare the query_index with sample indices that don't belong to
        // the same component using the UnionFind data structure
        auto nn_buffer_query = searcher(search::buffer::WithUnionFind(searcher.features_range_first(query_index),
                                                                      searcher.features_range_last(query_index),
                                                                      forest.get_union_find_const_reference(),
                                                                      smallest_component_representative,
                                                                      /*max_capacity=*/static_cast<IndexType>(1)));

        // the furthest nearest neighbor is also the closest in this case since we query only 1 neighbor
        const auto nearest_neighbor_index    = nn_buffer_query.furthest_index();
        const auto nearest_neighbor_distance = nn_buffer_query.furthest_distance();

        const auto current_closest_edge_distance = std::get<2>(closest_edge);

        const auto k_mutual_reachability_distance =
            std::max({core_distances[query_index], core_distances[nearest_neighbor_index], nearest_neighbor_distance});

        // update the current shortest edge if the k_mutual_reachability_distance is indeed shortest than the current
        // shortest edge distance
        if (k_mutual_reachability_distance < current_closest_edge_distance) {
            closest_edge = EdgeType{query_index, nearest_neighbor_index, k_mutual_reachability_distance};
        }
    }
    // merge components based on the best edge found
    forest.merge_components(closest_edge);
}

template <typename Indexer>
template <typename ForwardedIndexer>
auto BoruvkasAlgorithm<Indexer>::make_tree(ForwardedIndexer&& indexer) const {
    Forest forest(indexer.n_samples());

    const auto searcher = search::Searcher(std::forward<ForwardedIndexer>(indexer));

    std::size_t counter = 0;

    // compute the core distances only if knn > 1 -> k_nearest_reachability_distance is activated
    if (options_.k_nearest_neighbors_ > 1) {
        const auto core_distances =
            datastruct::mst::make_static_core_distances(searcher, options_.k_nearest_neighbors_);

        while (forest.n_components() > 1) {
            std::cout << "forest.n_components(): " << forest.n_components() << "\n";
            counter += forest.n_components();

            if (forest.n_components() == 2) {
                dual_component_step_sequential(searcher, core_distances, forest);

            } else {
                step_sequential(searcher, core_distances, forest);
            }
        }
    } else {
        while (forest.n_components() > 1) {
            std::cout << "forest.n_components(): " << forest.n_components() << "\n";
            counter += forest.n_components();

            if (forest.n_components() == 2) {
                dual_component_step_sequential(searcher, forest);

            } else {
                step_sequential(searcher, forest);
            }
        }
    }
    std::cout << "Counter: " << counter << "\n";
    return std::move(forest).minimum_spanning_tree();
}

// ---

template <typename Indexer>
void BoruvkasAlgorithm<Indexer>::step_dual_tree_sequential(const search::Searcher<Indexer>& searcher,
                                                           const CoreDistancesArrayType&    core_distances,
                                                           Forest&                          forest) const {
    common::ignore_parameters(core_distances);

    using QueryIndexerType = typename search::Searcher<Indexer>::IndexerType;

    // keep track of the shortest edge from a component's sample index to a sample index thats not within the
    // same component
    auto components_closest_edge = std::unordered_map<IndexType, EdgeType>{};

    for (const auto& [component_representative, component] : forest) {
        if (component.size() <= searcher.n_samples()) {
            auto query_indexer = QueryIndexerType();

            components_closest_edge[component_representative] =
                searcher.dual_tree_shortest_edge_with_core_distances(/**/ query_indexer,
                                                                     /**/ forest.get_union_find_const_reference(),
                                                                     /**/ component_representative,
                                                                     /**/ options_.k_nearest_neighbors_);
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
template <typename ForwardedIndexer>
auto BoruvkasAlgorithm<Indexer>::make_tree_2(ForwardedIndexer&& indexer) const {
    Forest forest(indexer.n_samples());

    const auto searcher = search::Searcher(std::forward<ForwardedIndexer>(indexer));

    std::size_t counter = 0;

    const auto core_distances = datastruct::mst::make_static_core_distances(searcher, options_.k_nearest_neighbors_);

    while (forest.n_components() > 1) {
        std::cout << "forest.n_components(): " << forest.n_components() << "\n";
        counter += forest.n_components();

        step_dual_tree_sequential(searcher, indexer, core_distances, forest);
    }
    std::cout << "Counter: " << counter << "\n";
    return std::move(forest).minimum_spanning_tree();
}

}  // namespace ffcl