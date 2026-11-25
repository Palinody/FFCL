#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/UnionFind.hpp"
#include "ffcl/datastruct/graph/spanning_tree/CoreDistances.hpp"
#include "ffcl/datastruct/graph/spanning_tree/MSTBuilder.hpp"
#include "ffcl/datastruct/graph/spanning_tree/MinimumSpanningTree.hpp"

#include "ffcl/search/buffer/EdgesBuffer.hpp"
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
#include <cmath>
#include <memory>
#include <unordered_map>

#if defined(_OPENMP) && THREADS_ENABLED == true
#include <omp.h>
#endif

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
    // The dual-tree pass works on squared distances and only takes the square root of the final edges.
    using SquaredCoreDistancesArrayType = datastruct::mst::CoreDistancesArray<ValueType>;

    using MSTBuilderType = MSTBuilder<IndexType, ValueType>;

    struct Options {
        Options() = default;

        Options(const Options& other) = default;

        Options& k_nearest_neighbors(std::size_t k_nearest_neighbors) {
            k_nearest_neighbors_ = k_nearest_neighbors;
            return *this;
        }

        Options& parallel_execution(bool parallel_execution) {
            parallel_execution_ = parallel_execution;
            return *this;
        }

        Options& operator=(const Options& options) {
            k_nearest_neighbors_ = options.k_nearest_neighbors_;
            parallel_execution_  = options.parallel_execution_;
            return *this;
        }

        std::size_t k_nearest_neighbors_ = 3;
        // Only meaningful when the library is compiled with `-DTHREADS_ENABLED=true`.
        bool parallel_execution_ =
#if defined(_OPENMP) && THREADS_ENABLED == true
            true;
#else
            false;
#endif
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

    using NodePtr        = typename Indexer::NodePtr;
    using EdgeBufferType = search::buffer::EdgesBuffer<Indexer, Indexer>;

    // Merge the per-component shortest edges of one Boruvka step into the builder. The buffer works in
    // squared space, so the weight is converted back once per merged edge.
    void merge_shortest_edges(const std::vector<EdgeType>& component_to_shortest_edges,
                              MSTBuilderType&              mst_builder) const;

    void step_dual_tree_sequential(const search::Searcher<Indexer>&     searcher,
                                   const SquaredCoreDistancesArrayType& squared_core_distances,
                                   MSTBuilderType&                      mst_builder) const;

    void step_dual_tree_parallel(const search::Searcher<Indexer>&     searcher,
                                 const SquaredCoreDistancesArrayType& squared_core_distances,
                                 MSTBuilderType&                      mst_builder) const;

    static void collect_leaves(const NodePtr& node, std::vector<NodePtr>& leaves);

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
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    MSTBuilderType mst_builder(indexer.n_samples());

    const auto searcher = search::Searcher(std::forward<ForwardedIndexer>(indexer));

    // compute the core distances only if knn > 1 -> k_nearest_reachability_distance is activated
    if (options_.k_nearest_neighbors_ > 1) {
        const auto core_distances =
            datastruct::mst::make_static_core_distances(searcher, options_.k_nearest_neighbors_);

        while (mst_builder.n_components() > 1) {
            step_sequential(searcher, core_distances, mst_builder);
        }
    } else {
        while (mst_builder.n_components() > 1) {
            step_sequential(searcher, mst_builder);
        }
    }
    return std::move(mst_builder).minimum_spanning_tree();
}

// ---

template <typename Indexer>
void BoruvkasAlgorithm<Indexer>::step_dual_tree_sequential(const search::Searcher<Indexer>&     searcher,
                                                           const SquaredCoreDistancesArrayType& squared_core_distances,
                                                           MSTBuilderType&                      mst_builder) const {
    // The MST has to be built on the mutual reachability distances, not on the raw ones, otherwise
    // `k_nearest_neighbors > 1` has no effect on the dual-tree path.
    const auto& component_to_shortest_edges = searcher.dtt_shortest_edge(searcher.indexer(),
                                                                         mst_builder.get_union_find_const_ref(),
                                                                         squared_core_distances.data());

    merge_shortest_edges(component_to_shortest_edges, mst_builder);
}

template <typename Indexer>
void BoruvkasAlgorithm<Indexer>::merge_shortest_edges(const std::vector<EdgeType>& component_to_shortest_edges,
                                                      MSTBuilderType&              mst_builder) const {
    for (const auto& edge : component_to_shortest_edges) {
        // The buffer works in squared space. Floating point cancellation can leave a tiny negative value for
        // near-coincident samples, so the argument is clamped before the square root.
        const auto squared_weight = std::max(ValueType{}, std::get<2>(edge));

        mst_builder.merge_components(
            ffcl::datastruct::mst::make_edge(std::get<0>(edge), std::get<1>(edge), std::sqrt(squared_weight)));
    }
}

template <typename Indexer>
void BoruvkasAlgorithm<Indexer>::collect_leaves(const NodePtr& node, std::vector<NodePtr>& leaves) {
    if (node->is_leaf()) {
        leaves.emplace_back(node);
        return;
    }
    collect_leaves(node->left_, leaves);
    collect_leaves(node->right_, leaves);
}

template <typename Indexer>
void BoruvkasAlgorithm<Indexer>::step_dual_tree_parallel(const search::Searcher<Indexer>&     searcher,
                                                         const SquaredCoreDistancesArrayType& squared_core_distances,
                                                         MSTBuilderType&                      mst_builder) const {
#if !defined(_OPENMP) || THREADS_ENABLED != true
    // No OpenMP available: the parallel step degrades to the sequential one.
    step_dual_tree_sequential(searcher, squared_core_distances, mst_builder);
#else
    // The dual-tree descent has no shared mutable state on the reference side: the candidate of a sample is
    // only ever touched by the thread that owns the query leaf containing it, and the node bounds are per
    // query node. So the step is parallelised over disjoint query leaves, each thread running the reference
    // descent for its own leaves with its own buffer. The per-component shortest edges are reduced and merged
    // sequentially afterwards, which keeps the union-find single threaded.
    const auto query_indexer = searcher.indexer();

    auto leaves = std::vector<NodePtr>{};
    leaves.reserve(query_indexer.n_nodes() / 2 + 1);
    collect_leaves(query_indexer.root(), leaves);

    const auto n_threads = static_cast<std::size_t>(std::max(1, omp_get_max_threads()));

    auto edge_buffers = std::vector<std::unique_ptr<EdgeBufferType>>{};
    edge_buffers.reserve(n_threads);

    for (std::size_t thread_index = 0; thread_index < n_threads; ++thread_index) {
        edge_buffers.emplace_back(
            std::make_unique<EdgeBufferType>(search::buffer::make_edge_buffer(query_indexer,
                                                                              query_indexer,
                                                                              mst_builder.get_union_find_const_ref(),
                                                                              squared_core_distances.data())));
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t leaf_index = 0; leaf_index < leaves.size(); ++leaf_index) {
        const auto thread_index = static_cast<std::size_t>(omp_get_thread_num());

        searcher.dtt_shortest_edge(query_indexer, {leaves[leaf_index]}, *edge_buffers[thread_index]);
    }

    // Reduce the per-thread component -> shortest edge maps into a single one.
    auto component_to_edge = std::unordered_map<IndexType, EdgeType>{};
    component_to_edge.reserve(leaves.size());

    for (const auto& edge_buffer : edge_buffers) {
        for (const auto& edge : edge_buffer->component_to_shortest_edge()) {
            const auto component = mst_builder.find(std::get<0>(edge));

            const auto existing_it = component_to_edge.find(component);

            if (existing_it == component_to_edge.end() || std::get<2>(edge) < std::get<2>(existing_it->second)) {
                component_to_edge[component] = edge;
            }
        }
    }
    auto shortest_edges = std::vector<EdgeType>{};
    shortest_edges.reserve(component_to_edge.size());

    for (const auto& [component, edge] : component_to_edge) {
        common::ignore_parameters(component);
        shortest_edges.emplace_back(edge);
    }
    merge_shortest_edges(shortest_edges, mst_builder);
#endif
}

template <typename Indexer>
template <typename ForwardedIndexer>
auto BoruvkasAlgorithm<Indexer>::make_tree_2(ForwardedIndexer&& indexer) const {
    MSTBuilderType forest(indexer.n_samples());

    const auto searcher = search::Searcher(std::forward<ForwardedIndexer>(indexer));

    const auto squared_core_distances =
        datastruct::mst::make_squared_core_distances(searcher, options_.k_nearest_neighbors_);

#if defined(_OPENMP) && THREADS_ENABLED == true
    const bool parallel_execution = options_.parallel_execution_;
#else
    const bool parallel_execution = false;
#endif
    while (forest.n_components() > 1) {
        if (parallel_execution) {
            step_dual_tree_parallel(searcher, squared_core_distances, forest);

        } else {
            step_dual_tree_sequential(searcher, squared_core_distances, forest);
        }
    }
    return std::move(forest).minimum_spanning_tree();
}

}  // namespace ffcl