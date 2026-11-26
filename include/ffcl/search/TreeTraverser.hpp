#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/StaticBuffer.hpp"

#include "ffcl/search/buffer/Unsorted.hpp"
#include "ffcl/search/buffer/WithMemory.hpp"
#include "ffcl/search/buffer/WithUnionFind.hpp"

#include "ffcl/search/buffer/EdgesBuffer.hpp"

#include "ffcl/search/ClosestPairOfSamples.hpp"

#include "ffcl/datastruct/UnionFind.hpp"

#include <array>
#include <deque>
#include <iterator>
#include <optional>
#include <queue>
#include <tuple>
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

    std::size_t n_features() const;

    constexpr auto begin() const;

    constexpr auto end() const;

    constexpr auto cbegin() const;

    constexpr auto cend() const;

    constexpr auto root() const;

    constexpr auto indexer() const {
        return reference_indexer_;
    }

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

    template <typename QueryIndexer>
    auto dtt_shortest_edge(const QueryIndexer&                     query_indexer,
                           const datastruct::UnionFind<IndexType>& union_find_const_ref,
                           const DataType*                         squared_core_distances_ptr) const;

    // Same traversal but restricted to a caller-provided set of query nodes and writing into a
    // caller-provided buffer. This is what lets the Boruvka step be parallelised over disjoint query
    // subtrees: each thread owns its buffer, so no state is shared.
    template <typename QueryIndexer, typename EdgesBuffer>
    void dtt_shortest_edge(const QueryIndexer&                  query_indexer,
                           const std::vector<ReferenceNodePtr>& query_nodes,
                           EdgesBuffer&                         edge_buffer) const;

  private:
    template <typename QueryNodePtr, typename ReferenceNodePtr, typename Cost>
    using DualNodePriorityQueueElementType = std::tuple<QueryNodePtr, ReferenceNodePtr, Cost>;

    static constexpr auto dual_node_greater_than_comparator_ = [](const auto& left_tuple, const auto& right_tuple) {
        return std::get<2>(left_tuple) > std::get<2>(right_tuple);
    };

    template <typename QueryNodePtr, typename ReferenceNodePtr, typename Cost>
    using DualNodePriorityQueueType =
        std::priority_queue<DualNodePriorityQueueElementType<QueryNodePtr, ReferenceNodePtr, Cost>,
                            std::vector<DualNodePriorityQueueElementType<QueryNodePtr, ReferenceNodePtr, Cost>>,
                            decltype(dual_node_greater_than_comparator_)>;

    template <typename Buffer>
    void single_tree_traversal(ReferenceNodePtr node, Buffer& buffer) const;

    template <typename QueryNodePtr, typename EdgesBuffer, typename... BufferArgs>
    void descend_query_children(const QueryNodePtr&     query_node,
                                const ReferenceNodePtr& reference_node,
                                EdgesBuffer&            edge_buffer,
                                BufferArgs&&... buffer_args) const;

    template <typename QueryNodePtr, typename EdgesBuffer, typename... BufferArgs>
    void descend_reference_children(const QueryNodePtr&     query_node,
                                    const ReferenceNodePtr& reference_node,
                                    EdgesBuffer&            edge_buffer,
                                    BufferArgs&&... buffer_args) const;

    template <typename QueryNodePtr, typename EdgesBuffer, typename... BufferArgs>
    void dual_tree_traversal_with_core_distances(const QueryNodePtr&     query_node,
                                                 const ReferenceNodePtr& reference_node,
                                                 EdgesBuffer&            edge_buffer,
                                                 std::optional<DataType> optional_cost,
                                                 bool                    bypass_cost_calculation,
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
std::size_t TreeTraverser<ReferenceIndexer>::n_features() const {
    return reference_indexer_.n_features();
}

template <typename ReferenceIndexer>
constexpr auto TreeTraverser<ReferenceIndexer>::begin() const {
    return reference_indexer_.cbegin();
}

template <typename ReferenceIndexer>
constexpr auto TreeTraverser<ReferenceIndexer>::end() const {
    return reference_indexer_.cend();
}

template <typename ReferenceIndexer>
constexpr auto TreeTraverser<ReferenceIndexer>::cbegin() const {
    return reference_indexer_.cbegin();
}

template <typename ReferenceIndexer>
constexpr auto TreeTraverser<ReferenceIndexer>::cend() const {
    return reference_indexer_.cend();
}

template <typename ReferenceIndexer>
constexpr auto TreeTraverser<ReferenceIndexer>::root() const {
    return reference_indexer_.root();
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

    std::for_each(buffer_batch.begin(), buffer_batch.end(), [this](auto& buffer) {
        buffer = (*this)(std::move(buffer));
    });

    return buffer_batch;
}

template <typename ReferenceIndexer>
template <typename Buffer>
void TreeTraverser<ReferenceIndexer>::single_tree_traversal(ReferenceNodePtr reference_node, Buffer& buffer) const {
    // Scan what this node owns. An internal node owns no sample (its pivot lives in a leaf), so this only
    // does work at the leaves.
    buffer.partial_search(reference_node->indices_range_.first,
                          reference_node->indices_range_.second,
                          reference_indexer_.begin(),
                          reference_indexer_.end(),
                          reference_indexer_.n_features());

    if (reference_node->is_leaf()) {
        return;
    }
    // Visit the child on the query's side first, then the other one only if the buffer's current bound can
    // still reach it. Same visiting order and same pruning test as the descend-then-climb traversal this
    // replaces, without the climb: the recursion returns through the call stack instead of through parent
    // links.
    const auto closest_child = reference_node->select_closest_child(reference_indexer_.begin(),
                                                                    reference_indexer_.end(),
                                                                    reference_indexer_.n_features(),
                                                                    buffer.centroid_begin(),
                                                                    buffer.centroid_end());

    single_tree_traversal(closest_child, buffer);

    // the sibling of the child just visited, not of this node: `select_sibling_node` answers for the node
    // it is called on
    if (const auto sibling_node = closest_child->select_sibling_node(reference_indexer_.begin(),
                                                                     reference_indexer_.end(),
                                                                     reference_indexer_.n_features(),
                                                                     buffer)) {
        single_tree_traversal(sibling_node, buffer);
    }
}

template <typename ReferenceIndexer>
template <typename QueryNodePtr, typename EdgesBuffer, typename... BufferArgs>
void TreeTraverser<ReferenceIndexer>::dual_tree_traversal_with_core_distances(const QueryNodePtr&     query_node,
                                                                              const ReferenceNodePtr& reference_node,
                                                                              EdgesBuffer&            edge_buffer,
                                                                              std::optional<DataType> optional_cost,
                                                                              bool bypass_cost_calculation,
                                                                              BufferArgs&&... buffer_args) const {
    // Calculate or update the cost if necessary. The caller already scored the combination when it built its
    // priority queue, in which case `bypass_cost_calculation` is set and the score is reused.
    if (!bypass_cost_calculation) {
        optional_cost = optional_cost ? edge_buffer.update_cost(query_node, reference_node, *optional_cost)
                                      : edge_buffer.cost(query_node, reference_node);

        if (!optional_cost) {
            return;
        }
    }
    // Base case: only leaf/leaf combinations are brute forced. The leaves of the KDTree cover every sample
    // exactly once (an internal node owns no sample, its pivot is stored at the end of its left child range),
    // so the leaf/leaf base cases alone are a complete enumeration of the point pairs.
    if (query_node->is_leaf() && reference_node->is_leaf()) {
        edge_buffer.base_case(query_node, reference_node, std::forward<BufferArgs>(buffer_args)...);
        return;
    }
    // Descend a single side per call, always the larger one. This is the size balanced descent of the
    // reference dual-tree Boruvka implementations: it reaches every leaf/leaf combination exactly once, so no
    // visited-combination bookkeeping is needed and the recursion depth stays logarithmic.
    if (query_node->is_leaf() || (!reference_node->is_leaf() && reference_node->diameter() > query_node->diameter())) {
        this->descend_reference_children(query_node,
                                         reference_node,
                                         edge_buffer,
                                         std::forward<BufferArgs>(buffer_args)...);

    } else {
        this->descend_query_children(query_node, reference_node, edge_buffer, std::forward<BufferArgs>(buffer_args)...);
    }
}

template <typename ReferenceIndexer>
template <typename QueryNodePtr, typename EdgesBuffer, typename... BufferArgs>
void TreeTraverser<ReferenceIndexer>::descend_query_children(const QueryNodePtr&     query_node,
                                                             const ReferenceNodePtr& reference_node,
                                                             EdgesBuffer&            edge_buffer,
                                                             BufferArgs&&... buffer_args) const {
    // Fixed two-element ordering: a std::priority_queue here would heap-allocate once per node combination,
    // which dominates at large sample counts.
    auto children = std::array<std::tuple<QueryNodePtr, DataType>, 2>{};

    for (std::size_t child_index = 0; child_index < 2; ++child_index) {
        const auto& child_query_node = (child_index == 0) ? query_node->left_ : query_node->right_;

        children[child_index] =
            std::make_tuple(child_query_node,
                            edge_buffer.cost(child_query_node, reference_node).value_or(common::infinity<DataType>()));
    }
    if (std::get<1>(children[1]) < std::get<1>(children[0])) {
        std::swap(children[0], children[1]);
    }
    // The closest child is descended first so that the reference node is scored against tightened buffers as
    // early as possible; the second one is re-scored because the first descent may have made it prunable.
    for (std::size_t child_index = 0; child_index < 2; ++child_index) {
        const auto& [child_query_node, child_cost] = children[child_index];

        if (child_cost >= common::infinity<DataType>()) {
            continue;
        }
        dual_tree_traversal_with_core_distances(child_query_node,
                                                reference_node,
                                                edge_buffer,
                                                child_index == 0 ? std::nullopt : std::make_optional(child_cost),
                                                child_index == 0,
                                                std::forward<BufferArgs>(buffer_args)...);
    }
}

template <typename ReferenceIndexer>
template <typename QueryNodePtr, typename EdgesBuffer, typename... BufferArgs>
void TreeTraverser<ReferenceIndexer>::descend_reference_children(const QueryNodePtr&     query_node,
                                                                 const ReferenceNodePtr& reference_node,
                                                                 EdgesBuffer&            edge_buffer,
                                                                 BufferArgs&&... buffer_args) const {
    // Same fixed two-element ordering as `descend_query_children`.
    auto children = std::array<std::tuple<ReferenceNodePtr, DataType>, 2>{};

    for (std::size_t child_index = 0; child_index < 2; ++child_index) {
        const auto& child_reference_node = (child_index == 0) ? reference_node->left_ : reference_node->right_;

        children[child_index] =
            std::make_tuple(child_reference_node,
                            edge_buffer.cost(query_node, child_reference_node).value_or(common::infinity<DataType>()));
    }
    if (std::get<1>(children[1]) < std::get<1>(children[0])) {
        std::swap(children[0], children[1]);
    }
    for (std::size_t child_index = 0; child_index < 2; ++child_index) {
        const auto& [child_reference_node, child_cost] = children[child_index];

        if (child_cost >= common::infinity<DataType>()) {
            continue;
        }
        dual_tree_traversal_with_core_distances(query_node,
                                                child_reference_node,
                                                edge_buffer,
                                                child_index == 0 ? std::nullopt : std::make_optional(child_cost),
                                                child_index == 0,
                                                std::forward<BufferArgs>(buffer_args)...);
    }
}

template <typename ReferenceIndexer>
template <typename QueryIndexer, typename EdgesBuffer>
void TreeTraverser<ReferenceIndexer>::dtt_shortest_edge(const QueryIndexer&                  query_indexer,
                                                        const std::vector<ReferenceNodePtr>& query_nodes,
                                                        EdgesBuffer&                         edge_buffer) const {
    common::ignore_parameters(query_indexer);

    for (const auto& query_node : query_nodes) {
        dual_tree_traversal_with_core_distances(query_node,
                                                reference_indexer_.root(),
                                                edge_buffer,
                                                std::nullopt,
                                                false);
    }
}

template <typename ReferenceIndexer>
template <typename QueryIndexer>
auto TreeTraverser<ReferenceIndexer>::dtt_shortest_edge(const QueryIndexer&                     query_indexer,
                                                        const datastruct::UnionFind<IndexType>& union_find_const_ref,
                                                        const DataType* squared_core_distances_ptr) const {
    // using BuffersFeaturesIteratorType = decltype(std::declval<QueryIndexer>().features_range_first(0));
    // using BufferType                  = buffer::Unsorted<BuffersFeaturesIteratorType>;

    auto edge_buffer =
        buffer::make_edge_buffer(query_indexer, reference_indexer_, union_find_const_ref, squared_core_distances_ptr);

    dual_tree_traversal_with_core_distances(query_indexer.root(),
                                            reference_indexer_.root(),
                                            edge_buffer,
                                            std::nullopt,
                                            false);

    return edge_buffer.component_to_shortest_edge();
}

}  // namespace ffcl::search
