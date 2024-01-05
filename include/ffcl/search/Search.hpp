#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/StaticBase.hpp"
#include "ffcl/search/count/StaticBase.hpp"

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

    using KDNodeViewPtr = typename ReferenceIndexer::KDNodeViewPtr;

    static_assert(common::is_raw_or_smart_ptr<KDNodeViewPtr>(), "KDNodeViewPtr is not a row or smart pointer");

  public:
    explicit Searcher(ReferenceIndexer&& reference_indexer);

    template <typename Buffer>
    Buffer operator()(Buffer&& buffer) const;

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
template <typename Buffer>
Buffer Searcher<ReferenceIndexer>::operator()(Buffer&& buffer) const {
    static_assert(common::is_crtp_of<Buffer, buffer::StaticBase>::value,
                  "Provided a Buffer that does not inherit from StaticBase<Derived>");

    return single_tree_traverser_(std::forward<Buffer>(buffer));
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