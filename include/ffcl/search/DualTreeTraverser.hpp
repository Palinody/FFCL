#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/StaticBase.hpp"
#include "ffcl/search/count/StaticBase.hpp"

namespace ffcl::search {

template <typename ReferenceIndexer>
class DualTreeTraverser {
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

    explicit DualTreeTraverser(ReferenceIndexer&& reference_indexer);

    std::size_t n_samples() const;

    constexpr auto features_range_first(std::size_t sample_index) const;

    constexpr auto features_range_last(std::size_t sample_index) const;

    template <typename ForwardedBuffer>
    ForwardedBuffer operator()(ForwardedBuffer&& forwarded_buffer) const;

  private:
    template <typename QueryNodePtr>
    void dual_tree_traversal(ReferenceNodePtr reference_node, QueryNodePtr query_node) const;

    ReferenceIndexer reference_indexer_;
};

template <typename ReferenceIndexer>
DualTreeTraverser<ReferenceIndexer>::DualTreeTraverser(ReferenceIndexer&& reference_indexer)
  : reference_indexer_{std::forward<ReferenceIndexer>(reference_indexer)} {}

template <typename ReferenceIndexer>
std::size_t DualTreeTraverser<ReferenceIndexer>::n_samples() const {
    return reference_indexer_.n_samples();
}

template <typename ReferenceIndexer>
constexpr auto DualTreeTraverser<ReferenceIndexer>::features_range_first(std::size_t sample_index) const {
    return reference_indexer_.features_range_first(sample_index);
}

template <typename ReferenceIndexer>
constexpr auto DualTreeTraverser<ReferenceIndexer>::features_range_last(std::size_t sample_index) const {
    return reference_indexer_.features_range_last(sample_index);
}

template <typename ReferenceIndexer>
template <typename ForwardedBuffer>
ForwardedBuffer DualTreeTraverser<ReferenceIndexer>::operator()(ForwardedBuffer&& forwarded_buffer) const {
    static_assert(common::is_crtp_of<ForwardedBuffer, buffer::StaticBase>::value,
                  "Provided a ForwardedBuffer that does not inherit from StaticBase<Derived>");

    auto processed_buffer = std::forward<ForwardedBuffer>(forwarded_buffer);
    // single_tree_traversal(reference_indexer_.root(), processed_buffer);
    return processed_buffer;
}

template <typename ReferenceIndexer>
template <typename QueryNodePtr>
void DualTreeTraverser<ReferenceIndexer>::dual_tree_traversal(ReferenceNodePtr reference_node,
                                                              QueryNodePtr     query_node) const {}

}  // namespace ffcl::search