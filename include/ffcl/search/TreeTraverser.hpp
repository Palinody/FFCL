#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/StaticBuffer.hpp"

#include "ffcl/search/buffer/Unsorted.hpp"
#include "ffcl/search/buffer/WithMemory.hpp"
#include "ffcl/search/buffer/WithUnionFind.hpp"

#include "ffcl/search/buffer/IndicesToBuffersMap.hpp"

#include <iterator>

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

    explicit TreeTraverser(ReferenceIndexer&& reference_indexer);

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

    template <typename... BufferArgs>
    auto dual_tree_shortest_edge(BufferArgs&&... buffer_args) const;

  private:
    template <typename Buffer>
    void single_tree_traversal(ReferenceNodePtr node, Buffer& buffer) const;

    template <typename Buffer>
    auto recursive_search_to_leaf_node(ReferenceNodePtr node, Buffer& buffer) const -> ReferenceNodePtr;

    template <typename Buffer>
    auto get_parent_node_after_sibling_traversal(const ReferenceNodePtr& node, Buffer& buffer) const
        -> ReferenceNodePtr;

    template <typename QueryNodePtr,
              typename QuerySamplesIterator,
              typename QueriesToBuffersMap,
              typename... BufferArgs>
    auto dual_tree_traversal(const QueryNodePtr&                   query_node,
                             const QuerySamplesIterator&           query_samples_range_first,
                             const QuerySamplesIterator&           query_samples_range_last,
                             std::size_t                           query_n_features,
                             const ReferenceNodePtr&               reference_node,
                             QueriesToBuffersMap&                  queries_to_buffers_map,
                             const std::tuple<DataType, DataType>& cache,
                             BufferArgs&&... buffer_args) const -> std::tuple<DataType, DataType>;

    ReferenceIndexer reference_indexer_;
};

template <typename ReferenceIndexer>
TreeTraverser<ReferenceIndexer>::TreeTraverser(ReferenceIndexer&& reference_indexer)
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

    auto queries_to_buffers_map = buffer::IndicesToBuffersMap<DeducedBufferType>{};

    auto shortest_edge_distance = common::infinity<DataType>();

    auto queries_max_upper_bound = DataType{0};

    auto cache = std::make_tuple(std::move(shortest_edge_distance), std::move(queries_max_upper_bound));

    dual_tree_traversal(query_indexer.root(),
                        query_indexer.begin(),
                        query_indexer.end(),
                        query_indexer.n_features(),
                        reference_indexer_.root(),
                        queries_to_buffers_map,
                        cache,
                        std::forward<BufferArgs>(buffer_args)...);

    return std::move(queries_to_buffers_map).tightest_edge();
}

template <typename ReferenceIndexer>
template <typename... BufferArgs>
auto TreeTraverser<ReferenceIndexer>::dual_tree_shortest_edge(BufferArgs&&... buffer_args) const {
    auto queries_to_buffers_map = buffer::IndicesToBuffersMap<buffer::WithUnionFind<SamplesIteratorType>>{};

    auto shortest_edge_distance = common::infinity<DataType>();

    auto queries_max_upper_bound = DataType{0};

    auto cache = std::make_tuple(std::move(shortest_edge_distance), std::move(queries_max_upper_bound));

    throw;

    dual_tree_traversal(reference_indexer_.root(),
                        reference_indexer_.begin(),
                        reference_indexer_.end(),
                        reference_indexer_.n_features(),
                        reference_indexer_.root(),
                        queries_to_buffers_map,
                        cache,
                        std::forward<BufferArgs>(buffer_args)...);

    return std::move(queries_to_buffers_map).tightest_edge();
}

template <typename ReferenceIndexer>
template <typename QueryNodePtr, typename QuerySamplesIterator, typename QueriesToBuffersMap, typename... BufferArgs>
auto TreeTraverser<ReferenceIndexer>::dual_tree_traversal(const QueryNodePtr&         query_node,
                                                          const QuerySamplesIterator& query_samples_range_first,
                                                          const QuerySamplesIterator& query_samples_range_last,
                                                          std::size_t                 query_n_features,
                                                          const ReferenceNodePtr&     reference_node,
                                                          QueriesToBuffersMap&        queries_to_buffers_map,
                                                          const std::tuple<DataType, DataType>& cache,
                                                          BufferArgs&&... buffer_args) const
    -> std::tuple<DataType, DataType> {
    // updates the query buffers with the reference set while keeping track of the global shortest edge
    auto updated_cache = queries_to_buffers_map.partial_search_for_each_query(query_node->indices_range_.first,
                                                                              query_node->indices_range_.second,
                                                                              query_samples_range_first,
                                                                              query_samples_range_last,
                                                                              query_n_features,
                                                                              reference_node->indices_range_.first,
                                                                              reference_node->indices_range_.second,
                                                                              reference_indexer_.begin(),
                                                                              reference_indexer_.end(),
                                                                              reference_indexer_.n_features(),
                                                                              cache,
                                                                              std::forward<BufferArgs>(buffer_args)...);

    const auto& [shortest_edge_distance, queries_max_upper_bound] = updated_cache;

    if (shortest_edge_distance <= queries_max_upper_bound) {
        // If at least one of the nodes is not a leaf, continue traversal
        if (!query_node->is_leaf()) {
            // Traverse left and right children of the query node
            updated_cache = dual_tree_traversal(query_node->left_,
                                                query_samples_range_first,
                                                query_samples_range_last,
                                                query_n_features,
                                                reference_node,
                                                queries_to_buffers_map,
                                                updated_cache,
                                                std::forward<BufferArgs>(buffer_args)...);

            updated_cache = dual_tree_traversal(query_node->right_,
                                                query_samples_range_first,
                                                query_samples_range_last,
                                                query_n_features,
                                                reference_node,
                                                queries_to_buffers_map,
                                                updated_cache,
                                                std::forward<BufferArgs>(buffer_args)...);
        }
        if (!reference_node->is_leaf()) {
            // Traverse left and right children of the reference node
            updated_cache = dual_tree_traversal(query_node,
                                                query_samples_range_first,
                                                query_samples_range_last,
                                                query_n_features,
                                                reference_node->left_,
                                                queries_to_buffers_map,
                                                updated_cache,
                                                std::forward<BufferArgs>(buffer_args)...);

            updated_cache = dual_tree_traversal(query_node,
                                                query_samples_range_first,
                                                query_samples_range_last,
                                                query_n_features,
                                                reference_node->right_,
                                                queries_to_buffers_map,
                                                updated_cache,
                                                std::forward<BufferArgs>(buffer_args)...);
        }
    }
    return updated_cache;
}

}  // namespace ffcl::search