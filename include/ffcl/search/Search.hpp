#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/Base.hpp"
#include "ffcl/search/count/Base.hpp"

#include "ffcl/search/SingleTreeTraverser.hpp"

namespace ffcl::search {

template <typename IndexerPtr>
class Searcher {
  public:
    static_assert(common::is_raw_or_smart_ptr<IndexerPtr>());

    using IndexType           = typename IndexerPtr::element_type::IndexType;
    using DataType            = typename IndexerPtr::element_type::DataType;
    using IndicesIteratorType = typename IndexerPtr::element_type::IndicesIteratorType;
    using SamplesIteratorType = typename IndexerPtr::element_type::SamplesIteratorType;

  public:
    Searcher(IndexerPtr query_indexer_ptr)
      : single_tree_traverser_{SingleTreeTraverser(query_indexer_ptr)} {}

    template <typename Buffer>
    Buffer operator()(Buffer&& buffer) {
        static_assert(common::is_crtp_of<Buffer, buffer::StaticBase>::value,
                      "Derived class does not inherit from StaticBase<Derived>");

        return single_tree_traverser_(std::forward<Buffer>(buffer));
    }

  private:
    SingleTreeTraverser<IndexerPtr> single_tree_traverser_;
};

template <typename IndexerPtr>
Searcher(IndexerPtr) -> Searcher<IndexerPtr>;

}  // namespace ffcl::search