#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/StaticBuffer.hpp"

#include "ffcl/search/DualTreeTraverser.hpp"
#include "ffcl/search/TreeTraverser.hpp"

namespace ffcl::search {

template <typename ReferenceIndexer>
class Searcher {
  public:
    using IndexerType = ReferenceIndexer;

    using IndexType = typename ReferenceIndexer::IndexType;
    using DataType  = typename ReferenceIndexer::DataType;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    using IndicesIteratorType = typename ReferenceIndexer::IndicesIteratorType;
    using SamplesIteratorType = typename ReferenceIndexer::SamplesIteratorType;

    static_assert(common::is_iterator<IndicesIteratorType>::value, "IndicesIteratorType is not an iterator");
    static_assert(common::is_iterator<SamplesIteratorType>::value, "SamplesIteratorType is not an iterator");

    using NodePtr = typename ReferenceIndexer::NodePtr;

    static_assert(common::is_raw_or_smart_ptr<NodePtr>, "NodePtr is not a raw or smart pointer");

    explicit Searcher(const ReferenceIndexer& reference_indexer);

    explicit Searcher(ReferenceIndexer&& reference_indexer) noexcept;

    std::size_t n_samples() const;

    std::size_t n_features() const;

    constexpr auto begin() const;

    constexpr auto end() const;

    constexpr auto cbegin() const;

    constexpr auto cend() const;

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

    template <typename QueryIndexer,
              typename... BufferArgs,
              typename std::enable_if_t<std::is_same_v<QueryIndexer, ReferenceIndexer>, bool> = true>
    auto dual_tree_shortest_edge(const QueryIndexer& forwarded_query_indexer, BufferArgs&&... buffer_args) const;

    template <typename ForwardedQueryIndexer,
              typename... BufferArgs,
              typename std::enable_if_t<std::is_same_v<ForwardedQueryIndexer, ReferenceIndexer>, bool> = true>
    auto dual_tree_shortest_edge(ForwardedQueryIndexer&& forwarded_query_indexer, BufferArgs&&... buffer_args) const;

    // template <typename... BufferArgs>
    // auto dual_tree_shortest_edge(BufferArgs&&... buffer_args) const;

    template <typename QueryIndexer,
              typename... BufferArgs,
              typename std::enable_if_t<std::is_same_v<QueryIndexer, ReferenceIndexer>, bool> = true>
    auto dual_tree_shortest_edge_with_core_distances(const QueryIndexer& forwarded_query_indexer,
                                                     BufferArgs&&... buffer_args) const;

    template <typename ForwardedQueryIndexer,
              typename... BufferArgs,
              typename std::enable_if_t<std::is_same_v<ForwardedQueryIndexer, ReferenceIndexer>, bool> = true>
    auto dual_tree_shortest_edge_with_core_distances(ForwardedQueryIndexer&& forwarded_query_indexer,
                                                     BufferArgs&&... buffer_args) const;

  private:
    TreeTraverser<ReferenceIndexer> tree_traverser_;
};

template <typename ReferenceIndexer>
Searcher(ReferenceIndexer) -> Searcher<ReferenceIndexer>;

template <typename ReferenceIndexer>
Searcher<ReferenceIndexer>::Searcher(const ReferenceIndexer& reference_indexer)
  : tree_traverser_{reference_indexer} {}

template <typename ReferenceIndexer>
Searcher<ReferenceIndexer>::Searcher(ReferenceIndexer&& reference_indexer) noexcept
  : tree_traverser_{std::move(reference_indexer)} {}

template <typename ReferenceIndexer>
std::size_t Searcher<ReferenceIndexer>::n_samples() const {
    return tree_traverser_.n_samples();
}

template <typename ReferenceIndexer>
std::size_t Searcher<ReferenceIndexer>::n_features() const {
    return tree_traverser_.n_features();
}

template <typename ReferenceIndexer>
constexpr auto Searcher<ReferenceIndexer>::begin() const {
    return tree_traverser_.cbegin();
}

template <typename ReferenceIndexer>
constexpr auto Searcher<ReferenceIndexer>::end() const {
    return tree_traverser_.cend();
}

template <typename ReferenceIndexer>
constexpr auto Searcher<ReferenceIndexer>::cbegin() const {
    return tree_traverser_.cbegin();
}

template <typename ReferenceIndexer>
constexpr auto Searcher<ReferenceIndexer>::cend() const {
    return tree_traverser_.cend();
}

template <typename ReferenceIndexer>
constexpr auto Searcher<ReferenceIndexer>::features_range_first(std::size_t sample_index) const {
    return tree_traverser_.features_range_first(sample_index);
}

template <typename ReferenceIndexer>
constexpr auto Searcher<ReferenceIndexer>::features_range_last(std::size_t sample_index) const {
    return tree_traverser_.features_range_last(sample_index);
}

template <typename ReferenceIndexer>
template <typename ForwardedBuffer,
          typename std::enable_if_t<!common::is_iterable_v<ForwardedBuffer> &&
                                        common::is_crtp_of<ForwardedBuffer, buffer::StaticBuffer>::value,
                                    bool>>
ForwardedBuffer Searcher<ReferenceIndexer>::operator()(ForwardedBuffer&& forwarded_buffer) const {
    return tree_traverser_(std::forward<ForwardedBuffer>(forwarded_buffer));
}

template <typename ReferenceIndexer>
template <typename ForwardedBufferBatch,
          typename std::
              enable_if_t<common::is_iterable_of_static_base<ForwardedBufferBatch, buffer::StaticBuffer>::value, bool>>
ForwardedBufferBatch Searcher<ReferenceIndexer>::operator()(ForwardedBufferBatch&& forwarded_buffer_batch) const {
    auto buffer_batch = std::forward<ForwardedBufferBatch>(forwarded_buffer_batch);

    std::for_each(buffer_batch.begin(),
                  buffer_batch.end(),
                  // update each buffer inplace
                  [this](auto& buffer) { buffer = (*this)(std::move(buffer)); });

    return buffer_batch;
}

template <typename ReferenceIndexer>
template <typename QueryIndexer,
          typename... BufferArgs,
          typename std::enable_if_t<std::is_same_v<QueryIndexer, ReferenceIndexer>, bool>>
auto Searcher<ReferenceIndexer>::dual_tree_shortest_edge(const QueryIndexer& query_indexer,
                                                         BufferArgs&&... buffer_args) const {
    return tree_traverser_.dual_tree_shortest_edge(query_indexer, std::forward<BufferArgs>(buffer_args)...);
}

template <typename ReferenceIndexer>
template <typename ForwardedQueryIndexer,
          typename... BufferArgs,
          typename std::enable_if_t<std::is_same_v<ForwardedQueryIndexer, ReferenceIndexer>, bool>>
auto Searcher<ReferenceIndexer>::dual_tree_shortest_edge(ForwardedQueryIndexer&& forwarded_query_indexer,
                                                         BufferArgs&&... buffer_args) const {
    return tree_traverser_.dual_tree_shortest_edge(std::forward<ForwardedQueryIndexer>(forwarded_query_indexer),
                                                   std::forward<BufferArgs>(buffer_args)...);
}

/*
template <typename ReferenceIndexer>
template <typename... BufferArgs>
auto Searcher<ReferenceIndexer>::dual_tree_shortest_edge(BufferArgs&&... buffer_args) const {
    return tree_traverser_.dual_tree_shortest_edge(std::forward<BufferArgs>(buffer_args)...);
}
*/

template <typename ReferenceIndexer>
template <typename QueryIndexer,
          typename... BufferArgs,
          typename std::enable_if_t<std::is_same_v<QueryIndexer, ReferenceIndexer>, bool>>
auto Searcher<ReferenceIndexer>::dual_tree_shortest_edge_with_core_distances(const QueryIndexer& query_indexer,
                                                                             BufferArgs&&... buffer_args) const {
    return tree_traverser_.dual_tree_shortest_edge_with_core_distances(query_indexer,
                                                                       std::forward<BufferArgs>(buffer_args)...);
}

template <typename ReferenceIndexer>
template <typename ForwardedQueryIndexer,
          typename... BufferArgs,
          typename std::enable_if_t<std::is_same_v<ForwardedQueryIndexer, ReferenceIndexer>, bool>>
auto Searcher<ReferenceIndexer>::dual_tree_shortest_edge_with_core_distances(
    ForwardedQueryIndexer&& forwarded_query_indexer,
    BufferArgs&&... buffer_args) const {
    return tree_traverser_.dual_tree_shortest_edge_with_core_distances(
        std::forward<ForwardedQueryIndexer>(forwarded_query_indexer), std::forward<BufferArgs>(buffer_args)...);
}

}  // namespace ffcl::search