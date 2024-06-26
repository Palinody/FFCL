#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/StaticBuffer.hpp"

#include "ffcl/search/buffer/Unsorted.hpp"
#include "ffcl/search/buffer/WithMemory.hpp"
#include "ffcl/search/buffer/WithUnionFind.hpp"

#include "ffcl/search/buffer/IndicesToBuffersMap.hpp"

#include "ffcl/search/ClosestPairOfSamples.hpp"

#include <deque>
#include <iterator>
#include <queue>
#include <vector>

namespace ffcl::search {

template <typename ReferenceIndexer>
class TreeTraverser {
  public:
    using IndexType = typename ReferenceIndexer::IndexType;
    using DataType  = typename ReferenceIndexer::DataType;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    using IndicesIteratorType = typename ReferenceIndexer::IndicesIteratorType;
    using SamplesIteratorType = typename ReferenceIndexer::SamplesIteratorType;

    static_assert(common::is_iterator<IndicesIteratorType>::value, "IndicesIteratorType is not an iterator");
    static_assert(common::is_iterator<SamplesIteratorType>::value, "SamplesIteratorType is not an iterator");

    using ReferenceNodePtr = typename ReferenceIndexer::NodePtr;

    static_assert(common::is_raw_or_smart_ptr<ReferenceNodePtr>, "ReferenceNodePtr is not a raw or smart pointer");

    explicit TreeTraverser(const ReferenceIndexer& reference_indexer);

    explicit TreeTraverser(ReferenceIndexer&& reference_indexer) noexcept;

    std::size_t n_samples() const;

    constexpr auto features_range_first(std::size_t sample_index) const;

    constexpr auto features_range_last(std::size_t sample_index) const;

    template <typename ForwardedBuffer,
              typename std::enable_if_t<!common::is_iterable_v<ForwardedBuffer> &&
                                            common::is_crtp_of<ForwardedBuffer, buffer::StaticBuffer>::value,
                                        bool> = true>
    ForwardedBuffer operator()(ForwardedBuffer&& forwarded_buffer) const;

    template <
        typename ForwardedBufferBatch,
        typename std::enable_if_t<common::is_iterable_of_static_base<ForwardedBufferBatch, buffer::StaticBuffer>::value,
                                  bool> = true>
    ForwardedBufferBatch operator()(ForwardedBufferBatch&& forwarded_buffer_batch) const;

    template <typename ForwardedQueryIndexer,
              typename... BufferArgs,
              typename std::enable_if_t<std::is_same_v<ForwardedQueryIndexer, ReferenceIndexer>, bool> = true>
    auto dual_tree_shortest_edge(ForwardedQueryIndexer&& forwarded_query_indexer, BufferArgs&&... buffer_args) const;

    // template <typename... BufferArgs>
    // auto dual_tree_shortest_edge(BufferArgs&&... buffer_args) const;

  private:
    template <typename QueryNodePtr, typename ReferenceNodePtr, typename Cost>
    using DualNodePriorityQueueElementType = std::tuple<QueryNodePtr, ReferenceNodePtr, Cost>;

    static constexpr auto dual_node_less_comparator_ = [](const auto& left_tuple, const auto& right_tuple) {
        return std::get<2>(left_tuple) > std::get<2>(right_tuple);
    };

    template <typename QueryNodePtr, typename ReferenceNodePtr, typename Cost>
    using DualNodePriorityQueueType =
        std::priority_queue<DualNodePriorityQueueElementType<QueryNodePtr, ReferenceNodePtr, Cost>,
                            std::vector<DualNodePriorityQueueElementType<QueryNodePtr, ReferenceNodePtr, Cost>>,
                            decltype(dual_node_less_comparator_)>;

    template <typename Buffer>
    void single_tree_traversal(ReferenceNodePtr node, Buffer& buffer) const;

    template <typename Buffer>
    auto recursive_search_to_leaf_node(ReferenceNodePtr node, Buffer& buffer) const -> ReferenceNodePtr;

    template <typename Buffer>
    auto get_parent_node_after_sibling_traversal(const ReferenceNodePtr& node, Buffer& buffer) const
        -> ReferenceNodePtr;

    template <typename QueryNodePtr, typename QueriesToBuffersMap, typename... BufferArgs>
    void dual_tree_traversal(const QueryNodePtr&     query_node,
                             const ReferenceNodePtr& reference_node,
                             QueriesToBuffersMap&    queries_to_buffers_map,
                             std::optional<DataType> optional_cost,
                             BufferArgs&&... buffer_args) const;

    template <typename QueryNodePtr, typename QueriesToBuffersMap, typename... BufferArgs>
    void process_non_leaf_query(const QueryNodePtr&     query_node,
                                const ReferenceNodePtr& reference_node,
                                QueriesToBuffersMap&    queries_to_buffers_map,
                                BufferArgs&&... buffer_args) const;

    template <typename QueryNodePtr, typename QueriesToBuffersMap, typename... BufferArgs>
    void process_non_leaf_reference(const QueryNodePtr&     query_node,
                                    const ReferenceNodePtr& reference_node,
                                    QueriesToBuffersMap&    queries_to_buffers_map,
                                    BufferArgs&&... buffer_args) const;

    template <typename QueryNodePtr, typename QueriesToBuffersMap, typename... BufferArgs>
    void process_non_leaf_both(const QueryNodePtr&     query_node,
                               const ReferenceNodePtr& reference_node,
                               QueriesToBuffersMap&    queries_to_buffers_map,
                               BufferArgs&&... buffer_args) const;

    template <typename Node1Ptr, typename Node2Ptr, typename QueriesToBuffersMap, typename... BufferArgs>
    void process_child_nodes(const Node1Ptr&      node1,
                             const Node2Ptr&      node2,
                             QueriesToBuffersMap& queries_to_buffers_map,
                             BufferArgs&&... buffer_args) const;

    ReferenceIndexer reference_indexer_;
};

template <typename ReferenceIndexer>
TreeTraverser(ReferenceIndexer) -> TreeTraverser<ReferenceIndexer>;

template <typename ReferenceIndexer>
TreeTraverser<ReferenceIndexer>::TreeTraverser(const ReferenceIndexer& reference_indexer)
  : reference_indexer_{reference_indexer} {}

template <typename ReferenceIndexer>
TreeTraverser<ReferenceIndexer>::TreeTraverser(ReferenceIndexer&& reference_indexer) noexcept
  : reference_indexer_{std::forward<ReferenceIndexer>(reference_indexer)} {}

template <typename ReferenceIndexer>
std::size_t TreeTraverser<ReferenceIndexer>::n_samples() const {
    return reference_indexer_.n_samples();
}

template <typename ReferenceIndexer>
constexpr auto TreeTraverser<ReferenceIndexer>::features_range_first(std::size_t sample_index) const {
    return reference_indexer_.features_range_first(sample_index);
}

template <typename ReferenceIndexer>
constexpr auto TreeTraverser<ReferenceIndexer>::features_range_last(std::size_t sample_index) const {
    return reference_indexer_.features_range_last(sample_index);
}

template <typename ReferenceIndexer>
template <typename ForwardedBuffer,
          typename std::enable_if_t<!common::is_iterable_v<ForwardedBuffer> &&
                                        common::is_crtp_of<ForwardedBuffer, buffer::StaticBuffer>::value,
                                    bool>>
ForwardedBuffer TreeTraverser<ReferenceIndexer>::operator()(ForwardedBuffer&& forwarded_buffer) const {
    auto buffer = std::forward<ForwardedBuffer>(forwarded_buffer);

    single_tree_traversal(reference_indexer_.root(), buffer);

    return buffer;
}

template <typename ReferenceIndexer>
template <typename ForwardedBufferBatch,
          typename std::
              enable_if_t<common::is_iterable_of_static_base<ForwardedBufferBatch, buffer::StaticBuffer>::value, bool>>
ForwardedBufferBatch TreeTraverser<ReferenceIndexer>::operator()(ForwardedBufferBatch&& forwarded_buffer_batch) const {
    auto buffer_batch = std::forward<ForwardedBufferBatch>(forwarded_buffer_batch);

    std::for_each(
        buffer_batch.begin(), buffer_batch.end(), [this](auto& buffer) { buffer = (*this)(std::move(buffer)); });

    return buffer_batch;
}

template <typename ReferenceIndexer>
template <typename Buffer>
void TreeTraverser<ReferenceIndexer>::single_tree_traversal(ReferenceNodePtr reference_node, Buffer& buffer) const {
    // current_node is a leaf node (and root in the special case where the entire tree is in a single node)
    auto current_reference_node = recursive_search_to_leaf_node(reference_node, buffer);
    // performs a partial search one step at a time from the leaf node until the input node is reached if
    // node parameter is a subtree
    while (current_reference_node != reference_node) {
        // performs a partial search starting from the specified node then returns its parent if it exists
        // (nullptr otherwise)
        current_reference_node = get_parent_node_after_sibling_traversal(current_reference_node, buffer);
    }
}

template <typename ReferenceIndexer>
template <typename Buffer>
auto TreeTraverser<ReferenceIndexer>::recursive_search_to_leaf_node(ReferenceNodePtr reference_node,
                                                                    Buffer&          buffer) const -> ReferenceNodePtr {
    buffer.partial_search(reference_node->indices_range_.first,
                          reference_node->indices_range_.second,
                          reference_indexer_.begin(),
                          reference_indexer_.end(),
                          reference_indexer_.n_features());
    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!reference_node->is_leaf()) {
        reference_node =
            recursive_search_to_leaf_node(reference_node->select_closest_child(reference_indexer_.begin(),
                                                                               reference_indexer_.end(),
                                                                               reference_indexer_.n_features(),
                                                                               buffer.centroid_begin(),
                                                                               buffer.centroid_end()),
                                          buffer);
    }
    return reference_node;
}

template <typename ReferenceIndexer>
template <typename Buffer>
auto TreeTraverser<ReferenceIndexer>::get_parent_node_after_sibling_traversal(const ReferenceNodePtr& reference_node,
                                                                              Buffer&                 buffer) const
    -> ReferenceNodePtr {
    // if a sibling node has been selected (is not nullptr)
    if (auto sibling_node = reference_node->select_sibling_node(/**/ reference_indexer_.begin(),
                                                                /**/ reference_indexer_.end(),
                                                                /**/ reference_indexer_.n_features(),
                                                                /**/ buffer)) {
        // search for other candidates from the sibling node
        single_tree_traversal(sibling_node, buffer);
    }
    // returns nullptr if node doesnt have parent (or is root)
    return reference_node->parent_.lock();
}

template <typename ReferenceIndexer>
template <typename ForwardedQueryIndexer,
          typename... BufferArgs,
          typename std::enable_if_t<std::is_same_v<ForwardedQueryIndexer, ReferenceIndexer>, bool>>
auto TreeTraverser<ReferenceIndexer>::dual_tree_shortest_edge(ForwardedQueryIndexer&& forwarded_query_indexer,
                                                              BufferArgs&&... buffer_args) const {
    using BuffersFeaturesIteratorType = decltype(std::declval<ForwardedQueryIndexer>().features_range_first(0));

    using DeducedBufferType = typename common::select_constructible_type<
        buffer::Unsorted<BuffersFeaturesIteratorType>,
        buffer::WithMemory<BuffersFeaturesIteratorType>,
        buffer::WithUnionFind<BuffersFeaturesIteratorType>>::from_signature</**/ BuffersFeaturesIteratorType,
                                                                            /**/ BuffersFeaturesIteratorType,
                                                                            /**/ BufferArgs...>::type;

    static_assert(!std::is_same_v<DeducedBufferType, void>,
                  "Deduced DeducedBufferType: void. Buffer type couldn't be deduced from 'BufferArgs&&...'.");

    const auto query_indexer = std::forward<ForwardedQueryIndexer>(forwarded_query_indexer);

    auto queries_to_buffers_map =
        buffer::make_indices_to_buffers_map<DeducedBufferType>(query_indexer.begin(),
                                                               query_indexer.end(),
                                                               query_indexer.n_features(),
                                                               reference_indexer_.begin(),
                                                               reference_indexer_.end(),
                                                               reference_indexer_.n_features());

    dual_tree_traversal(query_indexer.root(),
                        reference_indexer_.root(),
                        queries_to_buffers_map,
                        std::nullopt,
                        std::forward<BufferArgs>(buffer_args)...);

    return queries_to_buffers_map.tightest_edge();
}

/*
template <typename ReferenceIndexer>
template <typename... BufferArgs>
auto TreeTraverser<ReferenceIndexer>::dual_tree_shortest_edge(BufferArgs&&... buffer_args) const {
    auto queries_to_buffers_map = buffer::IndicesToBuffersMap<buffer::WithUnionFind<SamplesIteratorType>>{};

    throw;

    dual_tree_traversal(reference_indexer_.root(),
                        reference_indexer_.root(),
                        queries_to_buffers_map,
                        std::forward<BufferArgs>(buffer_args)...);

    return queries_to_buffers_map.tightest_edge();
}
*/

/*
template <typename ReferenceIndexer>
template <typename QueryNodePtr, typename QueriesToBuffersMap, typename... BufferArgs>
void TreeTraverser<ReferenceIndexer>::dual_tree_traversal(const QueryNodePtr&     query_node,
                                                          const ReferenceNodePtr& reference_node,
                                                          QueriesToBuffersMap&    queries_to_buffers_map,
                                                          BufferArgs&&... buffer_args) const {
    // updates the query buffers with the reference set while keeping track of the global shortest edge
    queries_to_buffers_map.partial_search_for_each_query(query_node->indices_range_.first,
                                                         query_node->indices_range_.second,
                                                         reference_node->indices_range_.first,
                                                         reference_node->indices_range_.second,
                                                         std::forward<BufferArgs>(buffer_args)...);

    auto dual_node_priority_queue =
        DualNodePriorityQueueType<QueryNodePtr, ReferenceNodePtr, DataType>{dual_node_less_comparator_};

    auto try_enqueue_combinations_from_cost = [&](const QueryNodePtr& q_node, const ReferenceNodePtr& r_node) {
        const auto optional_cost = queries_to_buffers_map.cost(q_node, r_node);
        if (optional_cost) {
            dual_node_priority_queue.emplace(std::make_tuple(q_node, r_node, *optional_cost));
        }
    };
    if (!query_node->is_leaf()) {
        try_enqueue_combinations_from_cost(query_node->left_, reference_node);
        try_enqueue_combinations_from_cost(query_node->right_, reference_node);
    }
    if (!reference_node->is_leaf()) {
        try_enqueue_combinations_from_cost(query_node, reference_node->left_);
        try_enqueue_combinations_from_cost(query_node, reference_node->right_);
    }
    for (; !dual_node_priority_queue.empty();) {
        const auto& [pq_query_node, pq_reference_node, cost] = dual_node_priority_queue.top();

        dual_tree_traversal(
            pq_query_node, pq_reference_node, queries_to_buffers_map, std::forward<BufferArgs>(buffer_args)...);

        dual_node_priority_queue.pop();

        if (!queries_to_buffers_map.update_cost(pq_query_node, pq_reference_node, cost)) {
            //
        }
    }
}
*/

/*
template <typename ReferenceIndexer>
template <typename QueryNodePtr, typename QueriesToBuffersMap, typename... BufferArgs>
void TreeTraverser<ReferenceIndexer>::dual_tree_traversal(const QueryNodePtr&     query_node,
                                                          const ReferenceNodePtr& reference_node,
                                                          QueriesToBuffersMap&    queries_to_buffers_map,
                                                          BufferArgs&&... buffer_args) const {
    // Updates the query buffers with the reference set while keeping track of the global shortest edge
    queries_to_buffers_map.partial_search_for_each_query(query_node->indices_range_.first,
                                                         query_node->indices_range_.second,
                                                         reference_node->indices_range_.first,
                                                         reference_node->indices_range_.second,
                                                         std::forward<BufferArgs>(buffer_args)...);

    auto dual_node_priority_queue = DualNodePriorityQueueType<QueryNodePtr, ReferenceNodePtr, DataType>{};

    auto try_enqueue_combinations_from_cost = [&](const QueryNodePtr& q_node, const ReferenceNodePtr& r_node) {
        const auto optional_cost = queries_to_buffers_map.cost(q_node, r_node);
        if (optional_cost) {
            dual_node_priority_queue.emplace(std::make_tuple(q_node, r_node, *optional_cost));
        }
    };

    if (!query_node->is_leaf()) {
        try_enqueue_combinations_from_cost(query_node->left_, reference_node);
        try_enqueue_combinations_from_cost(query_node->right_, reference_node);
    }
    if (!reference_node->is_leaf()) {
        try_enqueue_combinations_from_cost(query_node, reference_node->left_);
        try_enqueue_combinations_from_cost(query_node, reference_node->right_);
    }
    while (!dual_node_priority_queue.empty()) {
        const auto& [query_node, reference_node, cost] = dual_node_priority_queue.top();

        if (!queries_to_buffers_map.update_cost(query_node, reference_node, cost)) {
            dual_tree_traversal(
                query_node, reference_node, queries_to_buffers_map, std::forward<BufferArgs>(buffer_args)...);
        }
        dual_node_priority_queue.pop();
    }
}
*/

template <typename ReferenceIndexer>
template <typename QueryNodePtr, typename QueriesToBuffersMap, typename... BufferArgs>
void TreeTraverser<ReferenceIndexer>::dual_tree_traversal(const QueryNodePtr&     query_node,
                                                          const ReferenceNodePtr& reference_node,
                                                          QueriesToBuffersMap&    queries_to_buffers_map,
                                                          std::optional<DataType> optional_cost,
                                                          BufferArgs&&... buffer_args) const {
    if (!queries_to_buffers_map.is_nodes_combination_visited(query_node, reference_node)) {
        queries_to_buffers_map.mark_nodes_combination_as_vitited(query_node, reference_node);

        optional_cost = optional_cost ? queries_to_buffers_map.update_cost(query_node, reference_node, *optional_cost)
                                      : queries_to_buffers_map.cost(query_node, reference_node);

        if (optional_cost) {
            // updates the query buffers with the reference set while keeping track of the global shortest edge
            queries_to_buffers_map.partial_search_for_each_query(query_node->indices_range_.first,
                                                                 query_node->indices_range_.second,
                                                                 reference_node->indices_range_.first,
                                                                 reference_node->indices_range_.second,
                                                                 std::forward<BufferArgs>(buffer_args)...);
        } else {
            return;
        }
        // The order of traversal doesnt matter for the query node.
        if (!query_node->is_leaf()) {
            dual_tree_traversal(/**/ query_node->left_,
                                /**/ reference_node,
                                /**/ queries_to_buffers_map,
                                std::nullopt,
                                /**/ std::forward<BufferArgs>(buffer_args)...);

            dual_tree_traversal(/**/ query_node->right_,
                                /**/ reference_node,
                                /**/ queries_to_buffers_map,
                                std::nullopt,
                                /**/ std::forward<BufferArgs>(buffer_args)...);
        }
        if (!reference_node->is_leaf()) {
            // The order of traversal does matter in this case.
            const auto left_cost =
                queries_to_buffers_map.cost(query_node, reference_node->left_).value_or(common::infinity<DataType>());
            const auto right_cost =
                queries_to_buffers_map.cost(query_node, reference_node->right_).value_or(common::infinity<DataType>());

            if (left_cost < right_cost) {
                dual_tree_traversal(/**/ query_node,
                                    /**/ reference_node->left_,
                                    /**/ queries_to_buffers_map,
                                    std::nullopt,
                                    /**/ std::forward<BufferArgs>(buffer_args)...);

                dual_tree_traversal(/**/ query_node,
                                    /**/ reference_node->right_,
                                    /**/ queries_to_buffers_map,
                                    std::make_optional(right_cost),
                                    /**/ std::forward<BufferArgs>(buffer_args)...);

            } else if (left_cost > right_cost) {
                dual_tree_traversal(/**/ query_node,
                                    /**/ reference_node->right_,
                                    /**/ queries_to_buffers_map,
                                    std::nullopt,
                                    /**/ std::forward<BufferArgs>(buffer_args)...);

                dual_tree_traversal(/**/ query_node,
                                    /**/ reference_node->left_,
                                    /**/ queries_to_buffers_map,
                                    std::make_optional(left_cost),
                                    /**/ std::forward<BufferArgs>(buffer_args)...);
            } else if (common::equality(left_cost, right_cost) && (left_cost < common::infinity<DataType>())) {
                dual_tree_traversal(/**/ query_node,
                                    /**/ reference_node->left_,
                                    /**/ queries_to_buffers_map,
                                    std::nullopt,
                                    /**/ std::forward<BufferArgs>(buffer_args)...);

                dual_tree_traversal(/**/ query_node,
                                    /**/ reference_node->right_,
                                    /**/ queries_to_buffers_map,
                                    std::make_optional(right_cost),
                                    /**/ std::forward<BufferArgs>(buffer_args)...);
            }
        }
    }
}

}  // namespace ffcl::search