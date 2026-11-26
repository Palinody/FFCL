#pragma once

// Edge buffer of the dual-tree Boruvka traversal.
//
// This is the hot path of HDBSCAN*, so it is not polymorphic: everything lives in flat arrays indexed by
// sample index, component representative or dense node id. The generic `StaticBuffer` strategies remain
// available for the kNN / radius queries of `Searcher::operator()`, which is where a client selects a buffer
// strategy.
//
// All distances handled here are **squared** mutual reachability distances. The caller squares the core
// distances once and takes the square root of the resulting spanning-tree edges once.

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"
#include "ffcl/datastruct/bounds/distances/MinDistance.hpp"
#include "ffcl/datastruct/graph/spanning_tree/MinimumSpanningTree.hpp"  // for ffcl::datastruct::mst::Edge

#include "ffcl/datastruct/UnionFind.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>

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

    using EdgeType = datastruct::mst::Edge<IndexType, DistanceType>;

  public:
    // `squared_core_distances` is indexed by sample and holds the squared core distances.
    EdgesBuffer(const QueryIndexer&                     query_indexer,
                const ReferenceIndexer&                 reference_indexer,
                const datastruct::UnionFind<IndexType>& union_find_const_ref,
                const DistanceType*                     squared_core_distances);

    // Shortest outgoing edge of every component that found one (squared mutual reachability weights).
    auto component_to_shortest_edge() const -> std::vector<EdgeType>;

    auto tightest_edge() const -> EdgeType;

    template <typename... BufferArgs>
    void base_case(const QueryNodePtr& query_node, const ReferenceNodePtr& reference_node, BufferArgs&&...);

    auto cost(const QueryNodePtr& query_node, const ReferenceNodePtr& reference_node) -> std::optional<DistanceType>;

    auto update_cost(const QueryNodePtr& query_node, const ReferenceNodePtr& reference_node, const DistanceType& cost)
        -> std::optional<DistanceType>;

  private:
    // Squared distances are always finite and non-negative, so the sentinel is the squared infinity.
    static constexpr auto kInfinitySquared = common::infinity<DistanceType>();

    // The indexer containing the queries but that could also contain reference indices.
    const QueryIndexer& query_indexer_const_ref_;
    // The indexer containing the references but that could also contain query indices.
    const ReferenceIndexer& reference_indexer_const_ref_;
    // Data structure that tells which samples already belong to the same component.
    const datastruct::UnionFind<IndexType>& union_find_const_ref_;
    // Squared core distances, indexed by sample.
    const DistanceType* squared_core_distances_ptr_;

    // Union-find representatives, resolved once because the union-find is fixed for a whole Boruvka step.
    std::vector<IndexType> representative_vec_;

    // Shortest outgoing edge found so far for every component, indexed by the component representative.
    // The candidate is kept per component rather than per sample: a node's bound is then the maximum over
    // the distinct components it contains, which is tighter than the maximum over its samples and keeps
    // pruning effective once the components grow.
    std::vector<EdgeType>  component_best_edge_vec_;
    std::vector<char>      component_has_edge_vec_;
    std::vector<IndexType> components_with_edge_vec_;

    // Maximum of `sample_best_distance_vec_` over the subtree of every node. It is the bound used to prune a
    // node combination and it is maintained incrementally: after a base case the leaf is recomputed and the
    // value is propagated up the parent chain (O(depth)) instead of being recomputed on every `cost()` call.
    std::vector<DistanceType> node_max_bound_vec_;

    // Component membership of every node: the representative shared by all of its samples, or nullopt.
    std::vector<char>      node_component_is_known_vec_;
    std::vector<char>      node_component_is_homogeneous_vec_;
    std::vector<IndexType> node_component_vec_;
    // Smallest squared core distance of the subtree of every node.
    std::vector<char>         node_min_squared_core_is_known_vec_;
    std::vector<DistanceType> node_min_squared_core_vec_;

    auto compute_node_component_membership(const QueryNodePtr& node) -> std::optional<IndexType>;

    auto node_min_squared_core_distance(const QueryNodePtr& node) -> DistanceType;

    void refresh_node_max_bound(const QueryNodePtr& leaf_node);
};

template <typename QueryIndexer, typename ReferenceIndexer>
auto make_edge_buffer(const QueryIndexer&                                            query_indexer,
                      const ReferenceIndexer&                                        reference_indexer,
                      const datastruct::UnionFind<typename QueryIndexer::IndexType>& union_find_const_ref,
                      const typename QueryIndexer::DataType*                         squared_core_distances)
    -> EdgesBuffer<QueryIndexer, ReferenceIndexer> {
    return EdgesBuffer<QueryIndexer, ReferenceIndexer>(query_indexer,
                                                       reference_indexer,
                                                       union_find_const_ref,
                                                       squared_core_distances);
}

template <typename QueryIndexer, typename ReferenceIndexer>
EdgesBuffer<QueryIndexer, ReferenceIndexer>::EdgesBuffer(const QueryIndexer&                     query_indexer,
                                                         const ReferenceIndexer&                 reference_indexer,
                                                         const datastruct::UnionFind<IndexType>& union_find_const_ref,
                                                         const DistanceType*                     squared_core_distances)
  : query_indexer_const_ref_{query_indexer}
  , reference_indexer_const_ref_{reference_indexer}
  , union_find_const_ref_{union_find_const_ref}
  , squared_core_distances_ptr_{squared_core_distances}
  , representative_vec_(query_indexer.n_samples())
  , component_best_edge_vec_(query_indexer.n_samples(), datastruct::mst::make_infinity_edge<IndexType, DistanceType>())
  , component_has_edge_vec_(query_indexer.n_samples(), 0)
  , components_with_edge_vec_{}
  , node_max_bound_vec_(query_indexer.n_nodes(), kInfinitySquared)
  , node_component_is_known_vec_(query_indexer.n_nodes(), 0)
  , node_component_is_homogeneous_vec_(query_indexer.n_nodes(), 0)
  , node_component_vec_(query_indexer.n_nodes(), 0)
  , node_min_squared_core_is_known_vec_(query_indexer.n_nodes(), 0)
  , node_min_squared_core_vec_(query_indexer.n_nodes(), kInfinitySquared) {
    for (std::size_t index = 0; index < query_indexer.n_samples(); ++index) {
        representative_vec_[index] = union_find_const_ref_.find(static_cast<IndexType>(index));
    }
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::tightest_edge() const -> EdgeType {
    return datastruct::mst::make_infinity_edge<IndexType, DistanceType>();
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::component_to_shortest_edge() const -> std::vector<EdgeType> {
    auto edges = std::vector<EdgeType>{};
    edges.reserve(components_with_edge_vec_.size());

    for (const auto& component : components_with_edge_vec_) {
        edges.emplace_back(component_best_edge_vec_[component]);
    }
    return edges;
}

template <typename QueryIndexer, typename ReferenceIndexer>
template <typename... BufferArgs>
void EdgesBuffer<QueryIndexer, ReferenceIndexer>::base_case(const QueryNodePtr&     query_node,
                                                            const ReferenceNodePtr& reference_node,
                                                            BufferArgs&&...) {
    // Component membership of the whole reference node: when every reference sample belongs to the same
    // component, a query sample from that component has nothing to find here.
    const auto optional_reference_component = compute_node_component_membership(reference_node);
    const auto reference_min_squared_core   = node_min_squared_core_distance(reference_node);

    const auto reference_range_first = reference_node->indices_range_.first;
    const auto reference_range_last  = reference_node->indices_range_.second;

    const auto n_features = reference_indexer_const_ref_.n_features();

    for (auto query_it = query_node->indices_range_.first; query_it != query_node->indices_range_.second; ++query_it) {
        const auto query_index     = static_cast<IndexType>(*query_it);
        const auto query_component = representative_vec_[query_index];

        if (optional_reference_component && *optional_reference_component == query_component) {
            continue;
        }
        const auto query_squared_core = squared_core_distances_ptr_[static_cast<std::size_t>(query_index)];

        // Point-to-node prune: the smallest squared mutual reachability distance between this sample and any
        // sample of the reference node. If it cannot beat the component's current candidate the node is
        // skipped.
        if (component_has_edge_vec_[query_component]) {
            auto squared_lower_bound = datastruct::bounds::squared_min_distance(
                query_indexer_const_ref_.begin() + static_cast<std::ptrdiff_t>(query_index) * n_features,
                query_indexer_const_ref_.begin() + static_cast<std::ptrdiff_t>(query_index + 1) * n_features,
                reference_node->bound_);

            squared_lower_bound =
                std::max(squared_lower_bound, std::max(query_squared_core, reference_min_squared_core));

            if (squared_lower_bound >= std::get<2>(component_best_edge_vec_[query_component])) {
                continue;
            }
        }
        for (auto reference_it = reference_range_first; reference_it != reference_range_last; ++reference_it) {
            const auto reference_index = static_cast<IndexType>(*reference_it);

            if (query_component == representative_vec_[reference_index]) {
                continue;
            }
            const auto reference_squared_core = squared_core_distances_ptr_[static_cast<std::size_t>(reference_index)];

            // Cheap per-candidate prune: if the reference sample's own core distance already exceeds the
            // component's current candidate there is nothing to gain.
            if (component_has_edge_vec_[query_component] &&
                reference_squared_core >= std::get<2>(component_best_edge_vec_[query_component])) {
                continue;
            }
            const auto squared_distance = common::math::heuristics::squared_euclidean_distance(
                query_indexer_const_ref_.begin() + static_cast<std::ptrdiff_t>(query_index) * n_features,
                query_indexer_const_ref_.begin() + static_cast<std::ptrdiff_t>(query_index + 1) * n_features,
                reference_indexer_const_ref_.begin() + static_cast<std::ptrdiff_t>(reference_index) * n_features,
                reference_indexer_const_ref_.begin() + static_cast<std::ptrdiff_t>(reference_index + 1) * n_features);

            const auto squared_mutual_reachability =
                std::max(squared_distance, std::max(query_squared_core, reference_squared_core));

            if (squared_mutual_reachability < std::get<2>(component_best_edge_vec_[query_component]) ||
                !component_has_edge_vec_[query_component]) {
                if (!component_has_edge_vec_[query_component]) {
                    component_has_edge_vec_[query_component] = 1;
                    components_with_edge_vec_.emplace_back(query_component);
                }
                component_best_edge_vec_[query_component] =
                    ffcl::datastruct::mst::make_edge(query_index, reference_index, squared_mutual_reachability);
            }
        }
    }
    refresh_node_max_bound(query_node);
}

template <typename QueryIndexer, typename ReferenceIndexer>
void EdgesBuffer<QueryIndexer, ReferenceIndexer>::refresh_node_max_bound(const QueryNodePtr& leaf_node) {
    auto max_bound = DistanceType{};

    for (auto index_it = leaf_node->indices_range_.first; index_it != leaf_node->indices_range_.second; ++index_it) {
        const auto component = representative_vec_[static_cast<IndexType>(*index_it)];

        max_bound = std::max(
            max_bound,
            component_has_edge_vec_[component] ? std::get<2>(component_best_edge_vec_[component]) : kInfinitySquared);
    }
    if (max_bound >= node_max_bound_vec_[leaf_node->node_id()]) {
        return;
    }
    node_max_bound_vec_[leaf_node->node_id()] = max_bound;

    // Propagate upwards while the parent's bound actually improves.
    auto parent_node = leaf_node->parent_;

    while (parent_node) {
        const auto parent_id = parent_node->node_id();
        const auto new_bound = std::max(node_max_bound_vec_[parent_node->left_->node_id()],
                                        node_max_bound_vec_[parent_node->right_->node_id()]);

        if (new_bound >= node_max_bound_vec_[parent_id]) {
            break;
        }
        node_max_bound_vec_[parent_id] = new_bound;
        parent_node                    = parent_node->parent_;
    }
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::compute_node_component_membership(const QueryNodePtr& node)
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
        const auto sample_component = representative_vec_[sample_index];

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
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::node_min_squared_core_distance(const QueryNodePtr& node)
    -> DistanceType {
    const auto node_id = node->node_id();

    if (node_min_squared_core_is_known_vec_[node_id]) {
        return node_min_squared_core_vec_[node_id];
    }
    auto min_squared_core = kInfinitySquared;

    for (auto index_it = node->indices_range_.first; index_it != node->indices_range_.second; ++index_it) {
        min_squared_core = std::min(min_squared_core, squared_core_distances_ptr_[static_cast<std::size_t>(*index_it)]);
    }
    if (!node->is_leaf()) {
        min_squared_core = std::min(min_squared_core, node_min_squared_core_distance(node->left_));
        min_squared_core = std::min(min_squared_core, node_min_squared_core_distance(node->right_));
    }
    node_min_squared_core_vec_[node_id]          = min_squared_core;
    node_min_squared_core_is_known_vec_[node_id] = 1;

    return min_squared_core;
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::cost(const QueryNodePtr&     query_node,
                                                       const ReferenceNodePtr& reference_node)
    -> std::optional<DistanceType> {
    // Component prune: nothing to find when both nodes are entirely inside the same component.
    const auto optional_query_component     = compute_node_component_membership(query_node);
    const auto optional_reference_component = compute_node_component_membership(reference_node);

    if (optional_query_component && optional_query_component == optional_reference_component) {
        return std::nullopt;
    }
    // Bound prune: the smallest possible squared mutual reachability distance between the two nodes.
    auto squared_min_distance = datastruct::bounds::squared_min_distance(query_node->bound_, reference_node->bound_);

    const auto min_squared_core =
        std::max(node_min_squared_core_distance(query_node), node_min_squared_core_distance(reference_node));

    squared_min_distance = std::max(squared_min_distance, min_squared_core);

    return node_max_bound_vec_[query_node->node_id()] <= squared_min_distance
               ? std::nullopt
               : std::make_optional(squared_min_distance);
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::update_cost(const QueryNodePtr& query_node,
                                                              const ReferenceNodePtr&,
                                                              const DistanceType& cost) -> std::optional<DistanceType> {
    return node_max_bound_vec_[query_node->node_id()] <= cost ? std::nullopt : std::make_optional(cost);
}

}  // namespace ffcl::search::buffer
