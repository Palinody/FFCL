#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"
#include "ffcl/datastruct/bounds/distances/MinDistance.hpp"
#include "ffcl/datastruct/graph/spanning_tree/MinimumSpanningTree.hpp"  // for ffcl::datastruct::mst::Edge

#include "ffcl/datastruct/UnionFind.hpp"

#include "ffcl/search/buffer/IndicesToBuffersMap.hpp"  // Just for custom hash and nodes combination datastruct etc

// #include "ffcl/search/buffer/Unsorted.hpp"
// #include "ffcl/search/buffer/WithMemory.hpp"
#include "ffcl/search/buffer/WithUnionFind.hpp"

#include <cstddef>
#include <optional>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "ffcl/common/Timer.hpp"

namespace ffcl::search::buffer {


template <typename QueryIndexer, typename ReferenceIndexer>
class EdgesBuffer {
  public:
    using IndexType    = std::common_type_t<typename QueryIndexer::IndexType, typename ReferenceIndexer::IndexType>;
    using DistanceType = std::common_type_t<typename QueryIndexer::DataType, typename ReferenceIndexer::DataType>;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DistanceType>, "DistanceType must be trivial.");

    using QueryNodePtr     = typename QueryIndexer::NodePtr;
    using ReferenceNodePtr = typename ReferenceIndexer::NodePtr;

    using FeaturesIteratorType =
        std::common_type_t<typename QueryIndexer::SamplesIteratorType, typename ReferenceIndexer::SamplesIteratorType>;

    using EdgeType = datastruct::mst::Edge<IndexType, DistanceType>;

  public:
    EdgesBuffer(const QueryIndexer&                     query_indexer,
                const ReferenceIndexer&                 reference_indexer,
                const datastruct::UnionFind<IndexType>& union_find_const_ref,
                std::size_t                             buffer_size        = 1,
                const DistanceType*                     core_distances_ptr = nullptr);

    auto tightest_edge() const;

    const auto& component_to_k_shortest_edges() const;

    auto mark_combination_as_visited(const QueryNodePtr& query_node, const ReferenceNodePtr& reference_node)
        -> std::pair<typename std::unordered_set<NodesCombinationKey<QueryNodePtr, ReferenceNodePtr>>::iterator, bool>;

    // The trailing arguments are the ones the generic `IndicesToBuffersMap` buffer needs to construct its
    // per-query buffers. `EdgesBuffer` builds its buffers itself and only accepts them for signature
    // compatibility with the shared dual-tree traversal.
    template <typename... BufferArgs>
    void base_case(const QueryNodePtr& query_node, const ReferenceNodePtr& reference_node, BufferArgs&&...);

    // Component membership of a node, i.e. the representative shared by every sample of the
    // node (and of all its descendants). std::nullopt when the node spans several components.
    // Computed lazily bottom-up and cached for the duration of one Boruvka step.
    auto query_node_component_membership(const QueryNodePtr& query_node) -> std::optional<IndexType>;

    auto reference_node_component_membership(const ReferenceNodePtr& reference_node) -> std::optional<IndexType>;

    // Smallest edge of every component. Boruvka needs exactly one outgoing edge per component,
    // so this is the edge that has to be merged (the priority queue is a max-heap and its top()
    // is the *furthest* of the k candidates).
    auto component_to_shortest_edge() const -> std::vector<EdgeType>;

    auto cost(const QueryNodePtr& query_node, const ReferenceNodePtr& reference_node) -> std::optional<DistanceType>;

    auto update_cost(const QueryNodePtr& query_node, const ReferenceNodePtr&, const DistanceType& cost)
        -> std::optional<DistanceType>;

  private:
    // The indexer containing the queries but that could also contain reference indices.
    const QueryIndexer& query_indexer_const_ref_;
    // The indexer containing the references but that could also contain query indices.
    const ReferenceIndexer& reference_indexer_const_ref_;
    // Data structure that helps to determine which nodes are worth descending into.
    const datastruct::UnionFind<IndexType>& union_find_const_ref_;
    // Keeps track of the nodes combination that have already been visited so far.
    std::unordered_set<NodesCombinationKey<QueryNodePtr, ReferenceNodePtr>> visited_nodes_combinations_uset_;

    // Keeps track of the shortest edge found w.r.t. each component.
    std::unordered_map<IndexType, EdgeType> component_to_k_shortest_edge_umap_;
    // Keeps track of the current best edges found w.r.t. each component and place them in a priority queue.
    using EdgePriorityQueueElementType = EdgeType;
    using EdgePriorityQueueType        = std::priority_queue<EdgePriorityQueueElementType,
                                                      std::vector<EdgePriorityQueueElementType>,
                                                      std::less<EdgePriorityQueueElementType>>;
    // The edges priority queues are mapped with the component they belong to.
    using ComponentToEdgePriorityQueueUMapType = std::unordered_map<IndexType, EdgePriorityQueueType>;
    ComponentToEdgePriorityQueueUMapType component_to_k_edge_priority_queue_umap_;
    // Keeps track of the shortest edge found so far w.r.t. each component (the merge candidate). Indexed by
    // the component representative, i.e. a sample index, so no hashing is needed on the hot path.
    std::vector<EdgeType>  component_to_shortest_edge_vec_;
    std::vector<char>      component_has_edge_vec_;
    std::vector<IndexType> components_with_edge_vec_;
    // Keeps track of the component this node and all its descendants belong to.
    // Possible states (current node is included in the 'descendants'):
    //    std::nullopt: if any descendant sample belongs to a different component.
    //    Integer [0, n_samples-1]: the representative of all the descendant samples in the current node.
    // Flat per-node caches, indexed by `node_id()`. Both trees share the same ids when the query and the
    // reference indexer are the same object (the usual Boruvka case), which is the only case the traversal
    // supports.
    std::vector<char>          node_component_is_known_vec_;
    std::vector<IndexType>     node_component_vec_;
    std::vector<char>          node_component_is_homogeneous_vec_;

    using IndexToBufferUMapType         = std::unordered_map<IndexType, buffer::WithUnionFind<FeaturesIteratorType>>;
    using IndexToBufferMapIterator      = typename IndexToBufferUMapType::iterator;
    using IndexToBufferMapConstIterator = typename IndexToBufferUMapType::const_iterator;

    IndexToBufferUMapType query_to_buffer_umap_;
    // Flat lookup table from a sample index to its buffer. `update_bounds_limits` and `find_or_emplace_buffer`
    // are on the hottest path of the traversal and a hash lookup per sample per node combination is a large
    // part of the runtime, so the buffers are also reachable in O(1) without hashing.
    std::vector<buffer::WithUnionFind<FeaturesIteratorType>*> index_to_buffer_ptr_vec_;
    std::size_t                                               buffer_size_;
    // Optional core distances (HDBSCAN*): when set, every buffered distance is a mutual reachability distance.
    const DistanceType* core_distances_ptr_;

    auto find_or_emplace_buffer(const IndexType& index) -> buffer::WithUnionFind<FeaturesIteratorType>*;

    void update_component_to_edge_priority_queue(const IndexType& query_index,
                                                 const buffer::WithUnionFind<FeaturesIteratorType>& query_buffer);

    // ---

    struct BoundsLimits {
        void try_update_closest_limit(const DistanceType& closest_limit_candidate) {
            closest_limit = std::min(closest_limit, closest_limit_candidate);
        }

        void try_update_furthest_limit(const DistanceType& furthest_limit_candidate) {
            furthest_limit = std::max(furthest_limit, furthest_limit_candidate);
        }

        void try_update_limits(const DistanceType& bound_candidate) {
            try_update_closest_limit(bound_candidate);
            try_update_furthest_limit(bound_candidate);
        }

        auto compute_adjusted_bound(const DistanceType& node_diameter) const -> DistanceType {
            return (closest_limit < common::infinity<DistanceType>() - node_diameter)
                       ? std::min(furthest_limit, closest_limit + node_diameter)
                       : common::infinity<DistanceType>();
        }

        DistanceType closest_limit  = common::infinity<DistanceType>();
        DistanceType furthest_limit = 0;
    };
    // Maps each query nodes to their respective bound limits, updated as the query tree is traversed and used to
    // determine whether a query-reference nodes combination is worth exploring.
    std::vector<BoundsLimits> node_to_bounds_limits_vec_;
    // A node's limits only have to be recomputed when one of its samples improved its buffer.
    std::vector<char> node_bounds_limits_dirty_vec_;
    // Cached minimum core distance per node (only used when core distances are provided).
    std::vector<DistanceType> node_min_core_distance_vec_;
    std::vector<char>         node_min_core_distance_is_known_vec_;
    // Union-find representatives, computed once per step (the union-find is fixed for the whole step).
    std::vector<IndexType>    representatives_vec_;

    // Function that updates, as we traverse the query tree, the samples that are associated with the closest and
    // furthest representatives so as to find the lowest and upper bound of the node.
    auto query_node_furthest_bound(const QueryNodePtr& query_node) -> DistanceType;

    auto update_bounds_limits(const QueryNodePtr& query_node) -> BoundsLimits&;

    template <typename NodePtr>
    auto compute_node_component_membership(const NodePtr& node) -> std::optional<IndexType>;

    // Smallest core distance among all the samples of a node. It is a valid lower bound of the mutual
    // reachability distance between any sample of the node and any other sample.
    template <typename NodePtr>
    auto node_min_core_distance(const NodePtr& node) -> DistanceType;
};

template <typename QueryIndexer, typename ReferenceIndexer>
auto make_edge_buffer(const QueryIndexer&                                            query_indexer,
                      const ReferenceIndexer&                                        reference_indexer,
                      const datastruct::UnionFind<typename QueryIndexer::IndexType>& union_find_const_ref,
                      std::size_t                                                    buffer_size        = 1,
                      const typename QueryIndexer::DataType*                         core_distances_ptr = nullptr)
    -> EdgesBuffer<QueryIndexer, ReferenceIndexer> {
    return EdgesBuffer<QueryIndexer, ReferenceIndexer>(/**/ query_indexer,
                                                       /**/ reference_indexer,
                                                       /**/ union_find_const_ref,
                                                       /**/ buffer_size,
                                                       /**/ core_distances_ptr);
}

template <typename QueryIndexer, typename ReferenceIndexer>
EdgesBuffer<QueryIndexer, ReferenceIndexer>::EdgesBuffer(const QueryIndexer&                     query_indexer,
                                                         const ReferenceIndexer&                 reference_indexer,
                                                         const datastruct::UnionFind<IndexType>& union_find_const_ref,
                                                         std::size_t                             buffer_size,
                                                         const DistanceType*                     core_distances_ptr)
  : query_indexer_const_ref_{query_indexer}
  , reference_indexer_const_ref_{reference_indexer}
  , union_find_const_ref_{union_find_const_ref}
  , visited_nodes_combinations_uset_{}
  , component_to_k_shortest_edge_umap_{}
  , component_to_k_edge_priority_queue_umap_{}
  , component_to_shortest_edge_vec_(query_indexer.n_samples(), datastruct::mst::make_infinity_edge<IndexType, DistanceType>())
  , component_has_edge_vec_(query_indexer.n_samples(), 0)
  , components_with_edge_vec_{}
  , node_component_is_known_vec_(query_indexer.n_nodes(), 0)
  , node_component_vec_(query_indexer.n_nodes(), 0)
  , node_component_is_homogeneous_vec_(query_indexer.n_nodes(), 0)
  , query_to_buffer_umap_{}
  , index_to_buffer_ptr_vec_(query_indexer.n_samples(), nullptr)
  , buffer_size_{buffer_size}
  , core_distances_ptr_{core_distances_ptr}
  , node_to_bounds_limits_vec_(query_indexer.n_nodes())
  , node_bounds_limits_dirty_vec_(query_indexer.n_nodes(), 1)
  , node_min_core_distance_vec_(query_indexer.n_nodes(), DistanceType{})
  , node_min_core_distance_is_known_vec_(query_indexer.n_nodes(), 0)
  , representatives_vec_(query_indexer.n_samples()) {
    for (std::size_t index = 0; index < query_indexer.n_samples(); ++index) {
        representatives_vec_[index] = union_find_const_ref_.find(static_cast<IndexType>(index));
    }
    // visited_nodes_combinations_uset_.reserve(n_components);
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::tightest_edge() const {
    return datastruct::mst::make_infinity_edge<IndexType, DistanceType>();
}

template <typename QueryIndexer, typename ReferenceIndexer>
const auto& EdgesBuffer<QueryIndexer, ReferenceIndexer>::component_to_k_shortest_edges() const {
    return component_to_k_edge_priority_queue_umap_;
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::mark_combination_as_visited(const QueryNodePtr&     query_node,
                                                                              const ReferenceNodePtr& reference_node)
    -> std::pair<typename std::unordered_set<NodesCombinationKey<QueryNodePtr, ReferenceNodePtr>>::iterator, bool> {
    auto nodes_combination_key = NodesCombinationKey{query_node, reference_node};
    // Returns a pair consisting of an iterator to the inserted element (or to the element that prevented the
    // insertion) and a bool value set to true if and only if the insertion took place.
    // We are only interested in the boolean value.
    return visited_nodes_combinations_uset_.emplace(nodes_combination_key);
}

template <typename QueryIndexer, typename ReferenceIndexer>
template <typename... BufferArgs>
void EdgesBuffer<QueryIndexer, ReferenceIndexer>::base_case(const QueryNodePtr&     query_node,
                                                            const ReferenceNodePtr& reference_node,
                                                            BufferArgs&&...) {
    // Component membership of the whole reference node. When every reference sample belongs to the same
    // component, a query sample from that component has no candidate to find in this node and the search
    // can be skipped entirely.
    const auto references_component_membership = reference_node_component_membership(reference_node);
    // Constant for the whole base case: hoisted out of the per-query loop.
    const auto reference_node_min_core_distance =
        core_distances_ptr_ ? node_min_core_distance(reference_node) : DistanceType{};

    // Iterate through all query indices within the specified range of the query node.
    for (auto query_index_it = query_node->indices_range_.first; query_index_it != query_node->indices_range_.second;
         ++query_index_it) {
        const auto query_index = static_cast<IndexType>(*query_index_it);

        // Attempts to find the buffer corresponding to the current query if it exists. Otherwise it makes and returns
        // it.
        auto&      query_buffer    = *this->find_or_emplace_buffer(query_index);
        const auto query_component = union_find_const_ref_.find(query_index);

        const auto previous_closest_distance = query_buffer.closest_distance();

        // Perform a search through the references if at least one of the query_component is different than the
        // reference node component.
        if (!references_component_membership || *references_component_membership != query_component) {
            // Point-to-node prune: when the buffer is already full, its furthest candidate is the current
            // upper bound of the query sample. If the closest point of the reference node is further away than
            // that bound, no sample of the reference node can improve the buffer and the whole node is skipped.
            // This is the `referenceNode.MinDistance(queryPoint)` prune of the reference dual-tree Boruvka
            // implementation and it is what keeps the base cases cheap.
            const bool buffer_is_saturated = query_buffer.remaining_capacity() == 0;

            // Lower bound of the mutual reachability distance between the query sample and the reference
            // node: max(raw point-to-node distance, core(query), min core of the reference node).
            auto reference_node_lower_bound = datastruct::bounds::min_distance(
                query_buffer.centroid_begin(), query_buffer.centroid_end(), reference_node->bound_);

            if (core_distances_ptr_) {
                reference_node_lower_bound =
                    std::max(reference_node_lower_bound,
                             std::max(core_distances_ptr_[static_cast<std::size_t>(query_index)],
                                      reference_node_min_core_distance));
            }
            const bool reference_node_can_improve =
                !buffer_is_saturated || reference_node_lower_bound < query_buffer.furthest_distance();

            if (reference_node_can_improve) {
                query_buffer.partial_search(reference_node->indices_range_.first,
                                            reference_node->indices_range_.second,
                                            reference_indexer_const_ref_.begin(),
                                            reference_indexer_const_ref_.end(),
                                            reference_indexer_const_ref_.n_features());
            }
        }
        // The per-component bookkeeping is only worth updating when this base case actually improved the
        // buffer of the query sample.
        if (query_buffer.closest_distance() < previous_closest_distance) {
            update_component_to_edge_priority_queue(query_index, query_buffer);
            node_bounds_limits_dirty_vec_[query_node->node_id()] = 1;
        }
    }
    // Note: the component membership of the nodes is *not* cached here. A node only owns its pivot when it is
    // internal, so the membership derived from `indices_range_` alone would be too optimistic and would make
    // `compute_node_component_membership` prune node combinations that still contain useful edges.
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::find_or_emplace_buffer(const IndexType& index)
    -> buffer::WithUnionFind<FeaturesIteratorType>* {
    // Fast path: the buffer already exists, no hashing is needed.
    if (auto* buffer_ptr = index_to_buffer_ptr_vec_[index]) {
        return buffer_ptr;
    }
    auto buffer = buffer::WithUnionFind<FeaturesIteratorType>(
        query_indexer_const_ref_.begin() + index * query_indexer_const_ref_.n_features(),
        query_indexer_const_ref_.begin() + (index + 1) * query_indexer_const_ref_.n_features(),
        union_find_const_ref_,
        /*query_representative=*/union_find_const_ref_.find(index),
        /*max_capacity=*/buffer_size_);

    auto& buffer_ref = query_to_buffer_umap_.try_emplace(index, std::move(buffer)).first->second;

    index_to_buffer_ptr_vec_[index] = &buffer_ref;

    if (core_distances_ptr_) {
        buffer_ref.set_core_distances(core_distances_ptr_, index);
    }
    return &buffer_ref;
}

template <typename QueryIndexer, typename ReferenceIndexer>
void EdgesBuffer<QueryIndexer, ReferenceIndexer>::update_component_to_edge_priority_queue(
    const IndexType&                                  query_index,
    const buffer::WithUnionFind<FeaturesIteratorType>& buffer) {
    // Nothing was found for this query sample yet: do not record an edge. Recording one would push an
    // edge with an infinite distance and a default (0) reference index into the component's queue, which
    // then gets merged into the spanning tree.
    if (buffer.empty()) {
        return;
    }

    const auto query_component = union_find_const_ref_.find(query_index);
    const auto candidate_edge =
        ffcl::datastruct::mst::make_edge(query_index, buffer.closest_index(), buffer.closest_distance());

    // Boruvka only needs the single shortest outgoing edge per component, so the flat table is only touched
    // when the candidate actually improves it. This is the hottest bookkeeping of the traversal.
    if (component_has_edge_vec_[query_component] &&
        std::get<2>(candidate_edge) >= std::get<2>(component_to_shortest_edge_vec_[query_component])) {
        return;
    }

    if (!component_has_edge_vec_[query_component]) {
        component_has_edge_vec_[query_component] = 1;
        components_with_edge_vec_.emplace_back(query_component);
    }
    component_to_shortest_edge_vec_[query_component] = candidate_edge;

    // The k-best queue is only needed for the approximate variant; keep it for compatibility but do not
    // maintain it for the default k == 1.
    if (buffer_size_ > 1) {
        auto& edge_priority_queue = component_to_k_edge_priority_queue_umap_[query_component];

        if (edge_priority_queue.size() < buffer_size_) {
            edge_priority_queue.emplace(candidate_edge);

        } else if (std::get<2>(candidate_edge) < std::get<2>(edge_priority_queue.top())) {
            edge_priority_queue.pop();
            edge_priority_queue.emplace(candidate_edge);
        }
    }
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::component_to_shortest_edge() const -> std::vector<EdgeType> {
    auto edges = std::vector<EdgeType>{};
    edges.reserve(components_with_edge_vec_.size());

    for (const auto& component : components_with_edge_vec_) {
        edges.emplace_back(component_to_shortest_edge_vec_[component]);
    }
    return edges;
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::cost(const QueryNodePtr&     query_node,
                                                       const ReferenceNodePtr& reference_node)
    -> std::optional<DistanceType> {
    const auto optional_query_component     = query_node_component_membership(query_node);
    const auto optional_reference_component = reference_node_component_membership(reference_node);

    // If the query is not nullopt and the query is in the same component as the reference, we prune this combination.
    if (optional_query_component && optional_query_component == optional_reference_component) {
        return std::nullopt;
    }
    const auto min_distance = datastruct::bounds::min_distance(query_node->bound_, reference_node->bound_);

    // Bound prune: if the worst current edge of the query node is already shorter than the closest possible
    // distance between the two nodes, no query sample of the node can be improved by this reference node.
    if (query_node_furthest_bound(query_node) < min_distance) {
        return std::nullopt;
    }
    return std::make_optional(min_distance);
}

// Component membership of a node: the representative shared by all of its samples and of all of its
// descendants, or std::nullopt when the node spans several components. It is computed bottom-up on
// demand and cached, so that component based pruning is available at every level of the trees and not
// only on the node combinations that happened to be used as a base case.
template <typename QueryIndexer, typename ReferenceIndexer>
template <typename NodePtr>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::compute_node_component_membership(const NodePtr& node)
    -> std::optional<IndexType> {
    const auto node_id = node->node_id();

    if (node_component_is_known_vec_[node_id]) {
        return node_component_is_homogeneous_vec_[node_id] ? std::make_optional(node_component_vec_[node_id])
                                                           : std::optional<IndexType>{std::nullopt};
    }
    auto optional_component = std::optional<IndexType>{std::nullopt};
    bool is_first_sample    = true;
    bool is_consistent      = true;

    const auto consider_sample = [&](const IndexType& sample_index) {
        const auto sample_component = union_find_const_ref_.find(sample_index);

        if (is_first_sample) {
            optional_component = sample_component;
            is_first_sample    = false;

        } else if (!optional_component || *optional_component != sample_component) {
            optional_component = std::nullopt;
            is_consistent      = false;
        }
    };

    for (auto index_it = node->indices_range_.first; index_it != node->indices_range_.second; ++index_it) {
        consider_sample(static_cast<IndexType>(*index_it));

        if (!is_consistent) {
            break;
        }
    }
    // An internal node is only homogeneous when both of its children are homogeneous and agree.
    if (is_consistent && !node->is_leaf()) {
        for (const auto& child_node : {node->left_, node->right_}) {
            const auto optional_child_component = compute_node_component_membership(child_node);

            if (!optional_child_component || (optional_component && *optional_component != *optional_child_component)) {
                optional_component = std::nullopt;
                is_consistent      = false;
                break;
            }
            if (!optional_component) {
                optional_component = optional_child_component;
            }
        }
    }
    node_component_is_known_vec_[node_id]       = 1;
    node_component_is_homogeneous_vec_[node_id] = optional_component.has_value() ? 1 : 0;
    node_component_vec_[node_id]                = optional_component.value_or(0);

    return optional_component;
}

template <typename QueryIndexer, typename ReferenceIndexer>
template <typename NodePtr>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::node_min_core_distance(const NodePtr& node) -> DistanceType {
    if (!core_distances_ptr_) {
        return DistanceType{};
    }
    const auto node_id = node->node_id();

    if (node_min_core_distance_is_known_vec_[node_id]) {
        return node_min_core_distance_vec_[node_id];
    }
    auto min_core_distance = common::infinity<DistanceType>();

    for (auto index_it = node->indices_range_.first; index_it != node->indices_range_.second; ++index_it) {
        min_core_distance = std::min(min_core_distance, core_distances_ptr_[static_cast<std::size_t>(*index_it)]);
    }
    if (!node->is_leaf()) {
        for (const auto& child_node : {node->left_, node->right_}) {
            min_core_distance = std::min(min_core_distance, node_min_core_distance(child_node));
        }
    }
    node_min_core_distance_vec_[node_id]          = min_core_distance;
    node_min_core_distance_is_known_vec_[node_id] = 1;

    return min_core_distance;
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::query_node_component_membership(const QueryNodePtr& query_node)
    -> std::optional<IndexType> {
    return compute_node_component_membership(query_node);
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::reference_node_component_membership(
    const ReferenceNodePtr& reference_node) -> std::optional<IndexType> {
    return compute_node_component_membership(reference_node);
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::query_node_furthest_bound(const QueryNodePtr& query_node)
    -> DistanceType {
    auto& query_node_bounds_limits = update_bounds_limits(query_node);

    if (!query_node->is_leaf()) {
        // Fold in the cached children limits. They are monotone (best decreases, worst increases), so
        // re-applying them on every call is idempotent and keeps the bound valid as the buffers improve.
        for (const auto& child_node : {query_node->left_, query_node->right_}) {
            const auto& child_node_bounds_limits = node_to_bounds_limits_vec_[child_node->node_id()];

            query_node_bounds_limits.try_update_closest_limit(child_node_bounds_limits.closest_limit);
            query_node_bounds_limits.try_update_furthest_limit(child_node_bounds_limits.furthest_limit);
        }
    }
    return query_node_bounds_limits.compute_adjusted_bound(query_node->diameter());
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::update_bounds_limits(const QueryNodePtr& query_node)
    -> BoundsLimits& {
    static constexpr auto infinity = common::infinity<DistanceType>();

    const auto node_id = query_node->node_id();

    auto& query_node_bounds_limits = node_to_bounds_limits_vec_[node_id];

    // Nothing changed for this node since the last computation: reuse the cached limits.
    if (!node_bounds_limits_dirty_vec_[node_id]) {
        return query_node_bounds_limits;
    }
    node_bounds_limits_dirty_vec_[node_id] = 0;

    for (const auto& query_index : *query_node) {
        const auto* query_buffer_ptr = index_to_buffer_ptr_vec_[query_index];
        // If the buffer at the current index wasn't initialized, then its furthest distance is infinity by
        // default. We also don't need to iterate further since the next nodes will never be greater than
        // infinity.
        if (!query_buffer_ptr) {
            query_node_bounds_limits.furthest_limit = infinity;
        }
        // If there's remaining space in the buffers, then candidates might potentially be further than the
        // buffer's current furthest distance.
        else if (query_buffer_ptr->remaining_capacity()) {
            query_node_bounds_limits.furthest_limit = infinity;

        } else {
            query_node_bounds_limits.try_update_limits(query_buffer_ptr->furthest_distance());
        }
    }
    return query_node_bounds_limits;
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::update_cost(const QueryNodePtr& query_node,
                                                              const ReferenceNodePtr&,
                                                              const DistanceType& cost) -> std::optional<DistanceType> {
    return (query_node_furthest_bound(query_node) < cost) ? std::nullopt : std::make_optional(cost);
}

}  // namespace ffcl::search::buffer