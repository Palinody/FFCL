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

  public:
    EdgesBuffer(const QueryIndexer&                     query_indexer,
                const ReferenceIndexer&                 reference_indexer,
                const datastruct::UnionFind<IndexType>& union_find_const_ref,
                std::size_t                             buffer_size = 1);

    auto tightest_edge() const;

    const auto& component_to_k_edge_priority_queue_umap() const;

    auto emplace(const QueryNodePtr& query_node, const ReferenceNodePtr& reference_node)
        -> std::pair<typename std::unordered_set<NodesCombinationKey<QueryNodePtr, ReferenceNodePtr>>::iterator, bool>;

    template <typename... BufferArgs>
    void base_case(const QueryNodePtr& query_node, const ReferenceNodePtr& reference_node, BufferArgs&&... buffer_args);

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

    using EdgeType = datastruct::mst::Edge<IndexType, DistanceType>;
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
    // Keeps track of the component this node and all its descendants belong to.
    // Possible states (current node is included in the 'descendants'):
    //    std::nullopt: if any descendant sample belongs to a different component.
    //    Integer [0, n_samples-1]: the representative of all the descendant samples in the current node.
    std::unordered_map<QueryNodePtr, std::optional<IndexType>> query_node_to_descendants_component_umap_;
    // Same as for the queries.
    std::unordered_map<ReferenceNodePtr, std::optional<IndexType>> reference_node_to_descendants_component_umap_;

    using IndexToBufferUMapType         = std::unordered_map<IndexType, buffer::WithUnionFind<FeaturesIteratorType>>;
    using IndexToBufferMapIterator      = typename IndexToBufferUMapType::iterator;
    using IndexToBufferMapConstIterator = typename IndexToBufferUMapType::const_iterator;

    IndexToBufferUMapType query_to_buffer_umap_;
    std::size_t           buffer_size_;

    auto find_or_emplace_buffer(const IndexType& index) -> IndexToBufferMapIterator;

    void update_component_to_edge_priority_queue(const IndexToBufferMapIterator& query_to_buffer_it);

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
    std::unordered_map<QueryNodePtr, BoundsLimits> query_nodes_to_bounds_limits_umap_;

    // Function that updates, as we traverse the query tree, the samples that are associated with the closest and
    // furthest representatives so as to find the lowest and upper bound of the node.
    auto query_node_furthest_bound(const QueryNodePtr& query_node) -> DistanceType;

    auto update_bounds_limits(const QueryNodePtr& query_node) ->
        typename std::unordered_map<QueryNodePtr, BoundsLimits>::iterator;

    auto update_and_find_query_node_descendants_component(const QueryNodePtr& query_node) -> std::optional<IndexType>;

    auto update_and_find_reference_node_descendants_component(const ReferenceNodePtr& reference_node)
        -> std::optional<IndexType>;
};

template <typename QueryIndexer, typename ReferenceIndexer>
auto make_edge_buffer(const QueryIndexer&                                            query_indexer,
                      const ReferenceIndexer&                                        reference_indexer,
                      const datastruct::UnionFind<typename QueryIndexer::IndexType>& union_find_const_ref,
                      std::size_t buffer_size = 1) -> EdgesBuffer<QueryIndexer, ReferenceIndexer> {
    return EdgesBuffer<QueryIndexer, ReferenceIndexer>(/**/ query_indexer,
                                                       /**/ reference_indexer,
                                                       /**/ union_find_const_ref,
                                                       /**/ buffer_size);
}

template <typename QueryIndexer, typename ReferenceIndexer>
EdgesBuffer<QueryIndexer, ReferenceIndexer>::EdgesBuffer(const QueryIndexer&                     query_indexer,
                                                         const ReferenceIndexer&                 reference_indexer,
                                                         const datastruct::UnionFind<IndexType>& union_find_const_ref,
                                                         std::size_t                             buffer_size)
  : query_indexer_const_ref_{query_indexer}
  , reference_indexer_const_ref_{reference_indexer}
  , union_find_const_ref_{union_find_const_ref}
  , visited_nodes_combinations_uset_{}
  , component_to_k_shortest_edge_umap_{}
  , component_to_k_edge_priority_queue_umap_{}
  , query_node_to_descendants_component_umap_{}
  , reference_node_to_descendants_component_umap_{}
  , query_to_buffer_umap_{}
  , buffer_size_{buffer_size}
  , query_nodes_to_bounds_limits_umap_{} {
    // visited_nodes_combinations_uset_.reserve(n_components);
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::tightest_edge() const {
    return datastruct::mst::make_infinity_edge<IndexType, DistanceType>();
}

template <typename QueryIndexer, typename ReferenceIndexer>
const auto& EdgesBuffer<QueryIndexer, ReferenceIndexer>::component_to_k_edge_priority_queue_umap() const {
    return component_to_k_edge_priority_queue_umap_;
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::emplace(const QueryNodePtr&     query_node,
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
                                                            BufferArgs&&... buffer_args) {
    common::ignore_parameters(std::forward(buffer_args)...);
    // Keeps track of the component membership of the query node. Evaluates to std::nullopt if any of the query
    // samples are different.
    auto queries_component_membership = std::optional<IndexType>{std::nullopt};
    // Keeps track of the component membership of the reference node. Evaluates to std::nullopt if any of the reference
    // samples are different.
    auto references_component_membership = std::optional<IndexType>{std::nullopt};

    // Iterate through all query indices within the specified range of the query node.
    for (auto query_index_it = query_node->indices_range_.first; query_index_it != query_node->indices_range_.second;
         ++query_index_it) {
        // Attempts to find the buffer corresponding to the current query if it exists. Otherwise it makes and returns
        // it.
        auto query_to_buffer_it = this->find_or_emplace_buffer(*query_index_it);

        const auto query_component = union_find_const_ref_.find(*query_index_it);

        // Do the following for the first iteration of the loop.
        if (query_index_it == query_node->indices_range_.first) {
            // references_component_membership will be updated inplace in this function.
            query_to_buffer_it->second.partial_search(reference_node->indices_range_.first,
                                                      reference_node->indices_range_.second,
                                                      reference_indexer_const_ref_.begin(),
                                                      reference_indexer_const_ref_.end(),
                                                      reference_indexer_const_ref_.n_features(),
                                                      references_component_membership);

            queries_component_membership = query_component;

            // Insert references_component_membership in the container only if the reference_node key doesnt already
            // exist, creating a new key/value pair and not overriding if it already exists.
            reference_node_to_descendants_component_umap_.try_emplace(reference_node, references_component_membership);
        }
        // Perform a search through the references if at least one of the query_component is different than the
        // reference node component.
        else if (query_component != references_component_membership) {
            query_to_buffer_it->second.partial_search(reference_node->indices_range_.first,
                                                      reference_node->indices_range_.second,
                                                      reference_indexer_const_ref_.begin(),
                                                      reference_indexer_const_ref_.end(),
                                                      reference_indexer_const_ref_.n_features());
        }
        // Set as std::nullopt if any of the query samples dont belong to the same component.
        if (query_component != queries_component_membership) {
            queries_component_membership = std::nullopt;
        }
        update_component_to_edge_priority_queue(query_to_buffer_it);
    }
    // Insert queries_component_membership in the container only if the query_node key doesnt already
    // exist, creating a new key/value pair and not overriding if it already exists.
    query_node_to_descendants_component_umap_.try_emplace(query_node, queries_component_membership);
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::find_or_emplace_buffer(const IndexType& index)
    -> IndexToBufferMapIterator {
    // Attempt to find the buffer associated with the current index in the buffer map.
    auto index_to_buffer_it = query_to_buffer_umap_.find(index);
    // If the current index does not have an associated buffer in the map,
    if (index_to_buffer_it == query_to_buffer_umap_.end()) {
        auto buffer = buffer::WithUnionFind<FeaturesIteratorType>(
            query_indexer_const_ref_.begin() + index * query_indexer_const_ref_.n_features(),
            query_indexer_const_ref_.begin() + (index + 1) * query_indexer_const_ref_.n_features(),
            union_find_const_ref_,
            /*query_representative=*/union_find_const_ref_.find(index),
            /*max_capacity=*/buffer_size_);

        // Attempt to insert the newly created buffer into the map. If an element with the same
        // index already exists, try_emplace does nothing. Otherwise, it inserts the new element.
        // The method returns a pair, where the first element is an iterator to the inserted element
        // (or to the element that prevented the insertion) and the second element is a boolean
        // indicating whether the insertion took place.
        // We are only interested in the first element of the pair.
        index_to_buffer_it = query_to_buffer_umap_.try_emplace(index, std::move(buffer)).first;
    }
    return index_to_buffer_it;
}

template <typename QueryIndexer, typename ReferenceIndexer>
void EdgesBuffer<QueryIndexer, ReferenceIndexer>::update_component_to_edge_priority_queue(
    const IndexToBufferMapIterator& query_to_buffer_it) {
    const auto& [query_index, buffer] = *query_to_buffer_it;

    const auto query_component = union_find_const_ref_.find(query_index);

    // Attempt to try_emplace a default edge if the key doesnt exist yet.
    const auto& [component_to_edge_priority_queue_it, is_inserted] =
        component_to_k_edge_priority_queue_umap_.try_emplace(query_component, EdgePriorityQueueType{});

    // If the umap didnt contain a priority queue for the specified query_component or if the max capacity isnt reached
    // yet.
    if (is_inserted || component_to_edge_priority_queue_it->second.size() < buffer_size_) {
        // We can add the edge directly.
        component_to_edge_priority_queue_it->second.emplace(
            ffcl::datastruct::mst::make_edge(/**/ query_index,
                                             /**/ buffer.closest_index(),
                                             /**/ buffer.closest_distance()));

    } else {
        // Finds the edge with the lowest priority in the queue and gets its distance.
        const auto& edge_priority_queue_furthest_distance =
            std::get<2>(component_to_edge_priority_queue_it->second.top());

        // The edge with the lowest priority will get replaced if the new edge's distance is inferior.
        if (buffer.closest_distance() < edge_priority_queue_furthest_distance) {
            component_to_edge_priority_queue_it->second.pop();

            component_to_edge_priority_queue_it->second.emplace(
                ffcl::datastruct::mst::make_edge(/**/ query_index,
                                                 /**/ buffer.closest_index(),
                                                 /**/ buffer.closest_distance()));
        }
    }
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::cost(const QueryNodePtr&     query_node,
                                                       const ReferenceNodePtr& reference_node)
    -> std::optional<DistanceType> {
    const auto optional_query_component     = update_and_find_query_node_descendants_component(query_node);
    const auto optional_reference_component = update_and_find_reference_node_descendants_component(reference_node);

    // static std::size_t n_prune = 0;

    // If the query is not nullopt and the query is in the same component as the reference, we prune this combination.
    if (optional_query_component && optional_query_component == optional_reference_component) {
        // std::cout << "Pruning: " << (++n_prune) << "\n";
        return std::nullopt;
    }
    const auto min_distance = datastruct::bounds::min_distance(query_node->bound_, reference_node->bound_);

    return (query_node_furthest_bound(query_node) < min_distance) ? std::nullopt : std::make_optional(min_distance);
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::update_and_find_query_node_descendants_component(
    const QueryNodePtr& query_node) -> std::optional<IndexType> {
    const auto& node_to_descendants_component_it = query_node_to_descendants_component_umap_.find(query_node);

    // Check if query_node exists in the map and if its component is not nullopt.
    if (node_to_descendants_component_it != query_node_to_descendants_component_umap_.end() &&
        node_to_descendants_component_it->second != std::nullopt) {
        if (!query_node->is_leaf()) {
            // Iterate over children if query_node is not a leaf.
            for (const auto& child_node : {query_node->left_, query_node->right_}) {
                const auto child_node_to_descendants_component_it =
                    query_node_to_descendants_component_umap_.find(child_node);

                // If child query_node exists in the map.
                if (child_node_to_descendants_component_it != query_node_to_descendants_component_umap_.end()) {
                    const auto& optional_component = child_node_to_descendants_component_it->second;

                    // If child has no component, set the parent query_node component to nullopt.
                    if (optional_component == std::nullopt) {
                        node_to_descendants_component_it->second = std::nullopt;
                        // No need to check the other children.
                        return std::nullopt;
                    }
                    // If child's component differs from the parent's, set the parent's component to nullopt.
                    else if (optional_component != node_to_descendants_component_it->second) {
                        node_to_descendants_component_it->second = std::nullopt;
                        // No need to check the other children.
                        return std::nullopt;
                    }
                }
            }
        }
        return node_to_descendants_component_it->second;
    }
    return std::nullopt;
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::update_and_find_reference_node_descendants_component(
    const ReferenceNodePtr& reference_node) -> std::optional<IndexType> {
    const auto& node_to_descendants_component_it = reference_node_to_descendants_component_umap_.find(reference_node);

    // Check if reference_node exists in the map and if its component is not nullopt.
    if (node_to_descendants_component_it != reference_node_to_descendants_component_umap_.end() &&
        node_to_descendants_component_it->second != std::nullopt) {
        if (!reference_node->is_leaf()) {
            // Iterate over children if reference_node is not a leaf.
            for (const auto& child_node : {reference_node->left_, reference_node->right_}) {
                const auto child_node_to_descendants_component_it =
                    reference_node_to_descendants_component_umap_.find(child_node);

                // If child reference_node exists in the map.
                if (child_node_to_descendants_component_it != reference_node_to_descendants_component_umap_.end()) {
                    const auto& optional_component = child_node_to_descendants_component_it->second;

                    // If child has no component, set the parent reference_node component to nullopt.
                    if (optional_component == std::nullopt) {
                        node_to_descendants_component_it->second = std::nullopt;
                        // No need to check the other children.
                        return std::nullopt;
                    }
                    // If child's component differs from the parent's, set the parent's component to nullopt.
                    else if (optional_component != node_to_descendants_component_it->second) {
                        node_to_descendants_component_it->second = std::nullopt;
                        // No need to check the other children.
                        return std::nullopt;
                    }
                }
            }
        }
        return node_to_descendants_component_it->second;
    }
    return std::nullopt;
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::query_node_furthest_bound(const QueryNodePtr& query_node)
    -> DistanceType {
    auto query_node_to_bound_limits_it = update_bounds_limits(query_node);

    if (!query_node->is_leaf()) {
        // Update the current node's limits based on the cached children limits.
        for (const auto& child_node : {query_node->left_, query_node->right_}) {
            const auto [child_node_to_bound_limits_it, is_emplaced] =
                query_nodes_to_bounds_limits_umap_.try_emplace(child_node, BoundsLimits{});

            if (!is_emplaced) {
                query_node_to_bound_limits_it->second.try_update_closest_limit(
                    child_node_to_bound_limits_it->second.closest_limit);

                query_node_to_bound_limits_it->second.try_update_furthest_limit(
                    child_node_to_bound_limits_it->second.furthest_limit);
            }
        }
    }
    return query_node_to_bound_limits_it->second.compute_adjusted_bound(query_node->diameter());
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::update_bounds_limits(const QueryNodePtr& query_node) ->
    typename std::unordered_map<QueryNodePtr, BoundsLimits>::iterator {
    static constexpr auto infinity = common::infinity<DistanceType>();

    // Try to emplace if the 'query_node' key is not already present. Return 'first' to discard the bool (not) inserted.
    auto query_node_to_bound_limits_it =
        query_nodes_to_bounds_limits_umap_.try_emplace(query_node, BoundsLimits{}).first;

    for (const auto& query_index : *query_node) {
        const auto index_to_buffer_it = query_to_buffer_umap_.find(query_index);
        // If the buffer at the current index wasn't initialized, then its furthest distance is infinity by
        // default. We also don't need to iterate further since the next nodes will never be greater than
        // infinity.
        if (index_to_buffer_it == query_to_buffer_umap_.cend()) {
            query_node_to_bound_limits_it->second.furthest_limit = infinity;
        }
        // If there's remaining space in the buffers, then candidates might potentially be further than the
        // buffer's current furthest distance.
        else if (index_to_buffer_it->second.remaining_capacity()) {
            query_node_to_bound_limits_it->second.furthest_limit = infinity;

        } else {
            const auto query_buffer_furthest_distance = index_to_buffer_it->second.furthest_distance();

            query_node_to_bound_limits_it->second.try_update_limits(query_buffer_furthest_distance);
        }
    }
    return query_node_to_bound_limits_it;
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::update_cost(const QueryNodePtr& query_node,
                                                              const ReferenceNodePtr&,
                                                              const DistanceType& cost) -> std::optional<DistanceType> {
    return (query_node_furthest_bound(query_node) < cost) ? std::nullopt : std::make_optional(cost);
}

}  // namespace ffcl::search::buffer