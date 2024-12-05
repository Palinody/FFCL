#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"
#include "ffcl/datastruct/bounds/distances/MinDistance.hpp"
#include "ffcl/datastruct/graph/spanning_tree/MinimumSpanningTree.hpp"  // for ffcl::datastruct::mst::Edge

#include "ffcl/datastruct/UnionFind.hpp"

#include "ffcl/search/buffer/IndicesToBuffersMap.hpp"  // Just for custom hash and nodes combination datastruct etc

#include "ffcl/search/buffer/Unsorted.hpp"
#include "ffcl/search/buffer/WithMemory.hpp"
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
    using IndexType    = typename QueryIndexer::IndexType;
    using DistanceType = typename QueryIndexer::DataType;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DistanceType>, "DistanceType must be trivial.");

    using QueryNodePtr     = typename QueryIndexer::NodePtr;
    using ReferenceNodePtr = typename ReferenceIndexer::NodePtr;

  public:
    EdgesBuffer(const QueryIndexer&                     query_indexer,
                const ReferenceIndexer&                 reference_indexer,
                const datastruct::UnionFind<IndexType>& union_find_const_ref,
                std::size_t                             buffer_size = 1);

    auto tightest_edge() const;

    auto component_to_shortest_edge_map() const;

    auto emplace(const QueryNodePtr& query_node, const ReferenceNodePtr& reference_node)
        -> std::pair<typename std::unordered_set<NodesCombinationKey<QueryNodePtr, ReferenceNodePtr>>::iterator, bool>;

    void base_case(const QueryNodePtr& query_node, const ReferenceNodePtr& reference_node);

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
    std::unordered_set<NodesCombinationKey<QueryNodePtr, ReferenceNodePtr>> visited_nodes_combinations_;
    // Keeps track of the shortest edge found w.r.t. each component.
    std::unordered_map<IndexType, datastruct::mst::Edge<IndexType, DistanceType>> component_to_shortest_edge_map_;
    // Keeps track of the component this node and all its descendants belong to.
    // Possible states (current node is included in the 'descendants'):
    //    std::nullopt: if any descendant sample belongs to a different component.
    //    Integer [0, n_samples-1]: the representative of all the descendant samples in the current node.
    std::unordered_map<QueryNodePtr, std::optional<IndexType>> query_node_to_descendants_component_;
    // Same as for the queries.
    std::unordered_map<ReferenceNodePtr, std::optional<IndexType>> reference_node_to_descendants_component_;

    using FeaturesIteratorType =
        std::common_type_t<typename QueryIndexer::SamplesIteratorType, typename ReferenceIndexer::SamplesIteratorType>;

    using IndexToBufferMapType          = std::unordered_map<IndexType, buffer::Unsorted<FeaturesIteratorType>>;
    using IndexToBufferMapIterator      = typename IndexToBufferMapType::iterator;
    using IndexToBufferMapConstIterator = typename IndexToBufferMapType::const_iterator;

    IndexToBufferMapType query_to_buffer_map_;
    std::size_t          buffer_size_;

    auto find_or_emplace_buffer(const IndexType& index) -> IndexToBufferMapIterator;
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
  , visited_nodes_combinations_{}
  , component_to_shortest_edge_map_{}
  , query_node_to_descendants_component_{}
  , reference_node_to_descendants_component_{}
  , query_to_buffer_map_{}
  , buffer_size_{buffer_size} {
    // visited_nodes_combinations_.reserve(n_components);
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::tightest_edge() const {
    return datastruct::mst::make_infinity_edge<IndexType, DistanceType>();
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::component_to_shortest_edge_map() const {
    using EdgeType = datastruct::mst::Edge<IndexType, DistanceType>;

    return std::unordered_map<IndexType, EdgeType>{};
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::emplace(const QueryNodePtr&     query_node,
                                                          const ReferenceNodePtr& reference_node)
    -> std::pair<typename std::unordered_set<NodesCombinationKey<QueryNodePtr, ReferenceNodePtr>>::iterator, bool> {
    auto nodes_combination_key = NodesCombinationKey{query_node, reference_node};
    // Returns a pair consisting of an iterator to the inserted element (or to the element that prevented the
    // insertion) and a bool value set to true if and only if the insertion took place.
    // We are only interested in the boolean value.
    return visited_nodes_combinations_.emplace(nodes_combination_key).second;
}

template <typename QueryIndexer, typename ReferenceIndexer>
void EdgesBuffer<QueryIndexer, ReferenceIndexer>::base_case(const QueryNodePtr&     query_node,
                                                            const ReferenceNodePtr& reference_node) {
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

        if (query_index_it = query_node->indices_range_.first) {
            // references_component_membership will be updated inplace in this function.
            query_to_buffer_it->second.partial_search(reference_node->indices_range_.first,
                                                      reference_node->indices_range_.second,
                                                      reference_indexer_const_ref_.first(),
                                                      reference_indexer_const_ref_.end(),
                                                      reference_indexer_const_ref_.n_features(),
                                                      references_component_membership);

            queries_component_membership = query_component;

            // Insert references_component_membership in the container only if the reference_node key doesnt already
            // exist, creating a new key/value pair and not overriding if it already exists.
            reference_node_to_descendants_component_.emplace(reference_node, references_component_membership);

        }
        // Perform a search through the references if at least one of the reference samples dont belong to the same
        // component. Or, if its the case, if the query_component is different than the reference node component.
        else if (!references_component_membership || query_component != references_component_membership) {
            query_to_buffer_it->second.partial_search(reference_node->indices_range_.first,
                                                      reference_node->indices_range_.second,
                                                      reference_indexer_const_ref_.first(),
                                                      reference_indexer_const_ref_.end(),
                                                      reference_indexer_const_ref_.n_features());
        }
        // Set as std::nullopt if any of the query samples dont belong to the same component.
        if (query_component != queries_component_membership) {
            queries_component_membership = std::nullopt;
        }

        // update_priority_queue(query_to_buffer_it);
    }
    // Insert queries_component_membership in the container only if the query_node key doesnt already
    // exist, creating a new key/value pair and not overriding if it already exists.
    query_node_to_descendants_component_.emplace(query_node, queries_component_membership);
}

template <typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<QueryIndexer, ReferenceIndexer>::find_or_emplace_buffer(const IndexType& index)
    -> IndexToBufferMapIterator {
    // Attempt to find the buffer associated with the current index in the buffer map.
    auto index_to_buffer_it = query_to_buffer_map_.find(index);
    // If the current index does not have an associated buffer in the map,
    if (index_to_buffer_it == query_to_buffer_map_.end()) {
        auto buffer = buffer::WithUnionFind<FeaturesIteratorType>(
            query_indexer_const_ref_.first() + index * query_indexer_const_ref_.n_features(),
            query_indexer_const_ref_.first() + index * query_indexer_const_ref_.n_features() +
                query_indexer_const_ref_.n_features(),
            union_find_const_ref_,
            /*query_representative=*/union_find_const_ref_.find(index),
            /*max_capacity=*/buffer_size_);

        // Attempt to insert the newly created buffer into the map. If an element with the same
        // index already exists, emplace does nothing. Otherwise, it inserts the new element.
        // The method returns a pair, where the first element is an iterator to the inserted element
        // (or to the element that prevented the insertion) and the second element is a boolean
        // indicating whether the insertion took place.
        // We are only interested in the first element of the pair.
        index_to_buffer_it = query_to_buffer_map_.emplace(index, std::move(buffer)).first;
    }
    return index_to_buffer_it;
}

}  // namespace ffcl::search::buffer