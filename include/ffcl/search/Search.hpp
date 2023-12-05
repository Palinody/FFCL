#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/Base.hpp"
#include "ffcl/search/count/Base.hpp"

#include "ffcl/search/SingleTreeTraverser.hpp"

namespace ffcl::search {

template <typename IndexerPtr, typename Buffer>
class Searcher {
  public:
    using IndexType           = typename IndexerPtr::element_type::IndexType;
    using DataType            = typename IndexerPtr::element_type::DataType;
    using IndicesIteratorType = typename IndexerPtr::element_type::IndicesIteratorType;
    using SamplesIteratorType = typename IndexerPtr::element_type::SamplesIteratorType;

  private:
    static_assert(common::is_crtp_of<Buffer, buffer::StaticBase>::value,
                  "Derived class does not inherit from StaticBase<Derived>");

  public:
    Searcher(IndexerPtr query_indexer_ptr, const Buffer& buffer)
      : single_tree_traverser_{SingleTreeTraverser(query_indexer_ptr)}
      , buffer_{buffer} {}

    Buffer operator()(std::size_t query_index) {
        return single_tree_traverser_(query_index, buffer_);
    }

    Buffer operator()(const SamplesIteratorType& query_feature_first, const SamplesIteratorType& query_feature_last) {
        return single_tree_traverser_(query_feature_first, query_feature_last, buffer_);
    }

  private:
    SingleTreeTraverser<IndexerPtr> single_tree_traverser_;
    Buffer                          buffer_;
};

}  // namespace ffcl::search