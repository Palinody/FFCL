#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/StaticBase.hpp"
#include "ffcl/search/count/StaticBase.hpp"

#include "ffcl/search/DualTreeTraverser.hpp"
#include "ffcl/search/SingleTreeTraverser.hpp"

namespace ffcl::search {

template <typename ReferenceIndexer>
class Searcher {
  public:
    using IndexType = typename ReferenceIndexer::IndexType;
    using DataType  = typename ReferenceIndexer::DataType;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    using IndicesIteratorType = typename ReferenceIndexer::IndicesIteratorType;
    using SamplesIteratorType = typename ReferenceIndexer::SamplesIteratorType;

    static_assert(common::is_iterator<IndicesIteratorType>::value, "IndicesIteratorType is not an iterator");
    static_assert(common::is_iterator<SamplesIteratorType>::value, "SamplesIteratorType is not an iterator");

    using NodePtr = typename ReferenceIndexer::NodePtr;

    static_assert(common::is_raw_or_smart_ptr<NodePtr>(), "NodePtr is not a raw or smart pointer");

    explicit Searcher(ReferenceIndexer&& reference_indexer);

    template <typename ForwardedBuffer, typename std::enable_if_t<!common::is_iterable_v<ForwardedBuffer>, int> = 0>
    ForwardedBuffer operator()(ForwardedBuffer&& forwarded_buffer) const;

    template <typename ForwardedBufferBatch,
              typename std::enable_if_t<common::is_iterable_v<ForwardedBufferBatch>, int> = 0>
    ForwardedBufferBatch operator()(ForwardedBufferBatch&& forwarded_buffer_batch) const;

    std::size_t n_samples() const;

    constexpr auto features_range_first(std::size_t sample_index) const;

    constexpr auto features_range_last(std::size_t sample_index) const;

  private:
    SingleTreeTraverser<ReferenceIndexer> single_tree_traverser_;
};

template <typename ReferenceIndexer>
Searcher(ReferenceIndexer) -> Searcher<ReferenceIndexer>;

template <typename ReferenceIndexer>
Searcher<ReferenceIndexer>::Searcher(ReferenceIndexer&& reference_indexer)
  : single_tree_traverser_{std::forward<ReferenceIndexer>(reference_indexer)} {}

template <typename ReferenceIndexer>
template <typename ForwardedBuffer, typename std::enable_if_t<!common::is_iterable_v<ForwardedBuffer>, int>>
ForwardedBuffer Searcher<ReferenceIndexer>::operator()(ForwardedBuffer&& forwarded_buffer) const {
    static_assert(common::is_crtp_of<ForwardedBuffer, buffer::StaticBase>::value,
                  "Provided a ForwardedBuffer that does not inherit from StaticBase<Derived>");

    return single_tree_traverser_(std::forward<ForwardedBuffer>(forwarded_buffer));
}

template <typename ReferenceIndexer>
template <typename ForwardedBufferBatch, typename std::enable_if_t<common::is_iterable_v<ForwardedBufferBatch>, int>>
ForwardedBufferBatch Searcher<ReferenceIndexer>::operator()(ForwardedBufferBatch&& forwarded_buffer_batch) const {
    static_assert(common::is_iterable_v<ForwardedBufferBatch>, "ForwardedBufferBatch must be an iterable container");

    using BufferType = typename ForwardedBufferBatch::value_type;

    static_assert(common::is_crtp_of<BufferType, buffer::StaticBase>::value,
                  "Elements of ForwardedBufferBatch must inherit from StaticBase<Derived>");

    auto buffer_batch = std::forward<ForwardedBufferBatch>(forwarded_buffer_batch);

    for (auto& buffer : buffer_batch) {
        buffer = (*this)(std::move(buffer));
    }
    return buffer_batch;
}

template <typename ReferenceIndexer>
std::size_t Searcher<ReferenceIndexer>::n_samples() const {
    return single_tree_traverser_.n_samples();
}

template <typename ReferenceIndexer>
constexpr auto Searcher<ReferenceIndexer>::features_range_first(std::size_t sample_index) const {
    return single_tree_traverser_.features_range_first(sample_index);
}

template <typename ReferenceIndexer>
constexpr auto Searcher<ReferenceIndexer>::features_range_last(std::size_t sample_index) const {
    return single_tree_traverser_.features_range_last(sample_index);
}

}  // namespace ffcl::search