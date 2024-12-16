#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/UnionFind.hpp"
#include "ffcl/datastruct/graph/spanning_tree/CoreDistances.hpp"
#include "ffcl/datastruct/graph/spanning_tree/MSTBuilder.hpp"
#include "ffcl/datastruct/graph/spanning_tree/MinimumSpanningTree.hpp"

#include "ffcl/search/buffer/WithUnionFind.hpp"

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

    using ClusteredMTSBuilderType = ClusteredMTSBuilder<IndexType, ValueType>;
    using MSTBuilderType          = MSTBuilder<IndexType, ValueType>;  // ClusteredMTSBuilderType

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
    void step_sequential(const search::Searcher<Indexer>& searcher, MSTBuilderType& mst_builder) const;

    void step_sequential(const search::Searcher<Indexer>& searcher,
                         const CoreDistancesArrayType&    core_distances,
                         MSTBuilderType&                  mst_builder) const;

    void step_dual_tree_sequential(const search::Searcher<Indexer>& searcher,
                                   const CoreDistancesArrayType&    core_distances,
                                   MSTBuilderType&                  mst_builder) const;

    /*
    void step_dual_tree_parallel(const search::Searcher<Indexer>& searcher,
                                 const CoreDistancesArrayType&    core_distances,
                                 ClusteredMTSBuilderType&         forest) const;
    */

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
void BoruvkasAlgorithm<Indexer>::step_sequential(const search::Searcher<Indexer>& searcher,
                                                 MSTBuilderType&                  mst_builder) const {
    // keep track of the shortest edge from a component's sample index to a sample index thats not within the
    // same component
    auto components_closest_edge = std::unordered_map<IndexType, EdgeType>{};

    for (std::size_t query_index = 0; query_index < searcher.n_samples(); ++query_index) {
        const auto component_representative = mst_builder.find(query_index);
        // initialize a nearest neighbor buffer to compare the query_index with sample indices that don't belong to
        // the same component using the UnionFind data structure
        auto nn_buffer_query = searcher(search::buffer::WithUnionFind(searcher.features_range_first(query_index),
                                                                      searcher.features_range_last(query_index),
                                                                      mst_builder.get_union_find_const_ref(),
                                                                      component_representative,
                                                                      static_cast<IndexType>(1)));

        // the furthest nearest neighbor is also the closest in this case since we query only 1 neighbor
        const auto nearest_neighbor_index    = nn_buffer_query.furthest_index();
        const auto nearest_neighbor_distance = nn_buffer_query.furthest_distance();

        // Set the default distance using common::infinity if no edge exists for this component
        auto current_closest_edge_distance = common::infinity<ValueType>();

        if (components_closest_edge.find(component_representative) != components_closest_edge.end()) {
            current_closest_edge_distance = std::get<2>(components_closest_edge[component_representative]);
        }
        // update the current shortest edge if the nearest_neighbor_distance is indeed shortest than the current
        // shortest edge distance
        if (nearest_neighbor_distance < current_closest_edge_distance) {
            components_closest_edge[component_representative] =
                EdgeType{query_index, nearest_neighbor_index, nearest_neighbor_distance};
        }
    }
    // merge components based on the best edges found in each component so far
    for (const auto& [component_representative, edge] : components_closest_edge) {
        assert(std::get<2>(edge) < common::infinity<ValueType>());
        common::ignore_parameters(component_representative);
        mst_builder.merge_components(edge);
    }
}

template <typename Indexer>
void BoruvkasAlgorithm<Indexer>::step_sequential(const search::Searcher<Indexer>& searcher,
                                                 const CoreDistancesArrayType&    core_distances,
                                                 MSTBuilderType&                  mst_builder) const {
    // keep track of the shortest edge from a component's sample index to a sample index thats not within the
    // same component
    auto components_closest_edge = std::unordered_map<IndexType, EdgeType>{};

    for (std::size_t query_index = 0; query_index < searcher.n_samples(); ++query_index) {
        const auto component_representative = mst_builder.find(query_index);
        // Initialize a nearest neighbor buffer to compare the query_index with sample indices that don't belong to
        // the same component using the UnionFind data structure
        auto nn_buffer_query = searcher(search::buffer::WithUnionFind(searcher.features_range_first(query_index),
                                                                      searcher.features_range_last(query_index),
                                                                      mst_builder.get_union_find_const_ref(),
                                                                      component_representative,
                                                                      static_cast<IndexType>(1)));

        // The furthest nearest neighbor is also the closest in this case since we query only 1 neighbor
        const auto nearest_neighbor_index    = nn_buffer_query.furthest_index();
        const auto nearest_neighbor_distance = nn_buffer_query.furthest_distance();

        // Set the default distance using common::infinity if no edge exists for this component
        auto current_closest_edge_distance = common::infinity<ValueType>();

        if (components_closest_edge.find(component_representative) != components_closest_edge.end()) {
            current_closest_edge_distance = std::get<2>(components_closest_edge[component_representative]);
        }
        const auto k_mutual_reachability_distance =
            std::max({core_distances[query_index], core_distances[nearest_neighbor_index], nearest_neighbor_distance});

        // Update the current shortest edge if the k_mutual_reachability_distance is indeed shorter
        if (k_mutual_reachability_distance < current_closest_edge_distance) {
            components_closest_edge[component_representative] =
                EdgeType{query_index, nearest_neighbor_index, k_mutual_reachability_distance};
        }
    }
    // merge components based on the best edges found in each component so far
    for (const auto& [component_representative, edge] : components_closest_edge) {
        assert(std::get<2>(edge) < common::infinity<ValueType>());
        common::ignore_parameters(component_representative);
        mst_builder.merge_components(edge);
    }
}

template <typename Indexer>
template <typename ForwardedIndexer>
auto BoruvkasAlgorithm<Indexer>::make_tree(ForwardedIndexer&& indexer) const {
    MSTBuilderType mst_builder(indexer.n_samples());

    const auto searcher = search::Searcher(std::forward<ForwardedIndexer>(indexer));

    std::size_t counter = 0;

    // compute the core distances only if knn > 1 -> k_nearest_reachability_distance is activated
    if (options_.k_nearest_neighbors_ > 1) {
        const auto core_distances =
            datastruct::mst::make_static_core_distances(searcher, options_.k_nearest_neighbors_);

        while (mst_builder.n_components() > 1) {
            std::cout << "mst_builder.n_components(): " << mst_builder.n_components() << "\n";
            counter += mst_builder.n_components();

            step_sequential(searcher, core_distances, mst_builder);
        }
    } else {
        while (mst_builder.n_components() > 1) {
            std::cout << "mst_builder.n_components(): " << mst_builder.n_components() << "\n";
            counter += mst_builder.n_components();

            step_sequential(searcher, mst_builder);
        }
    }
    std::cout << "Counter: " << counter << "\n";
    return std::move(mst_builder).minimum_spanning_tree();
}

// ---

template <typename Indexer>
void BoruvkasAlgorithm<Indexer>::step_dual_tree_sequential(const search::Searcher<Indexer>& searcher,
                                                           const CoreDistancesArrayType&    core_distances,
                                                           MSTBuilderType&                  mst_builder) const {
    common::ignore_parameters(core_distances);

    /*
    using IndicesIterator         = typename search::Searcher<Indexer>::IndicesIteratorType;
    using SamplesIterator         = typename search::Searcher<Indexer>::SamplesIteratorType;
    using QueryIndexerType        = typename search::Searcher<Indexer>::IndexerType;
    using QueryIndexerOptionsType = typename QueryIndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    // keep track of the shortest edge from a component's sample index to a sample index thats not within the
    // same component
    auto components_closest_edge = std::unordered_map<IndexType, EdgeType>{};
    for (auto& [component_representative, component] : mst_builder) {
        auto query_indexer =
            QueryIndexerType(component.begin(),
                             component.end(),
                             searcher.begin(),
                             searcher.end(),
                             searcher.n_features(),
                             QueryIndexerOptionsType()
                                 .bucket_size(std::max(static_cast<std::size_t>(40),
                                                       static_cast<std::size_t>(std::sqrt(component.size()))))
                                 .axis_selection_policy(AxisSelectionPolicyType{})
                                 .splitting_rule_policy(SplittingRulePolicyType{}));

        components_closest_edge[component_representative] =
            searcher.dual_tree_shortest_edge_with_core_distances(query_indexer,
                                                                 mst_builder.get_union_find_const_ref(),
                                                                 component_representative,
                                                                 options_.k_nearest_neighbors_);
    }
    // merge components based on the best edges found in each component so far
    for (const auto& [component_representative, edge] : components_closest_edge) {
        assert(std::get<2>(edge) < common::infinity<ValueType>());
        common::ignore_parameters(component_representative);
        mst_builder.merge_components(edge);
    }
    */

    const auto& component_to_k_edge_priority_queue_umap =
        searcher.dtt_shortest_edge(/**/ searcher.indexer(),
                                   /**/ mst_builder.get_union_find_const_ref(),
                                   /**/ options_.k_nearest_neighbors_);

    // merge components based on the best edges found in each component so far
    for (const auto& [component_representative, edge_priority_queue] : component_to_k_edge_priority_queue_umap) {
        assert(std::get<2>(edge_priority_queue.top()) < common::infinity<ValueType>());
        common::ignore_parameters(component_representative);
        mst_builder.merge_components(edge_priority_queue.top());
    }
}

/*
template <typename Indexer>
void BoruvkasAlgorithm<Indexer>::step_dual_tree_parallel(const search::Searcher<Indexer>& searcher,
                                                         const CoreDistancesArrayType&    core_distances,
                                                         ClusteredMTSBuilderType&         forest) const {
    common::ignore_parameters(core_distances);

    using IndicesIterator         = typename search::Searcher<Indexer>::IndicesIteratorType;
    using SamplesIterator         = typename search::Searcher<Indexer>::SamplesIteratorType;
    using QueryIndexerType        = typename search::Searcher<Indexer>::IndexerType;
    using QueryIndexerOptionsType = typename QueryIndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    // Map to store each component's closest edge
    auto components_closest_edge = std::unordered_map<IndexType, EdgeType>{};

    // The forest needs to be cast to a contiguous container.
    auto forest_vector = std::vector(forest.begin(), forest.end());

#pragma omp parallel for
    for (std::size_t index = 0; index < forest_vector.size(); ++index) {
        auto& component_representative = forest_vector[index].first;
        auto& component                = forest_vector[index].second;

        auto query_indexer =
            QueryIndexerType(component.begin(),
                             component.end(),
                             searcher.begin(),
                             searcher.end(),
                             searcher.n_features(),
                             QueryIndexerOptionsType()
                                 .bucket_size(std::max(static_cast<std::size_t>(40),
                                                       static_cast<std::size_t>(std::sqrt(component.size()))))
                                 .max_depth(component.size())
                                 .axis_selection_policy(AxisSelectionPolicyType{})
                                 .splitting_rule_policy(SplittingRulePolicyType{}));

        auto closest_edge = searcher.dual_tree_shortest_edge_with_core_distances(
            query_indexer, forest.get_union_find_const_ref(), component_representative, options_.k_nearest_neighbors_);

        // Ensure thread-safe access to shared resource
#pragma omp critical
        { components_closest_edge[component_representative] = closest_edge; }
    }

    // Sequentially merge components based on the best edges found in each component
    for (const auto& [component_representative, edge] : components_closest_edge) {
        assert(std::get<2>(edge) < common::infinity<ValueType>());
        common::ignore_parameters(component_representative);
        forest.merge_components(edge);
    }
}
*/

template <typename Indexer>
template <typename ForwardedIndexer>
auto BoruvkasAlgorithm<Indexer>::make_tree_2(ForwardedIndexer&& indexer) const {
    MSTBuilderType forest(indexer.n_samples());

    const auto searcher = search::Searcher(std::forward<ForwardedIndexer>(indexer));

    std::size_t counter = 0;

    const auto core_distances = datastruct::mst::make_static_core_distances(searcher, options_.k_nearest_neighbors_);

    while (forest.n_components() > 1) {
        std::cout << "forest.n_components(): " << forest.n_components() << "\n";
        counter += forest.n_components();

        step_dual_tree_sequential(searcher, core_distances, forest);
    }
    std::cout << "Counter: " << counter << "\n";
    return std::move(forest).minimum_spanning_tree();
}

}  // namespace ffcl