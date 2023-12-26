#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/StaticBase.hpp"
#include "ffcl/search/count/StaticBase.hpp"

#include "ffcl/search/SingleTreeTraverser.hpp"

namespace ffcl::search {

template <typename Indexer>
class Searcher {
  public:
    using IndexType = typename Indexer::IndexType;
    using DataType  = typename Indexer::DataType;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    using IndicesIteratorType = typename Indexer::IndicesIteratorType;
    using SamplesIteratorType = typename Indexer::SamplesIteratorType;

    static_assert(common::is_iterator<IndicesIteratorType>::value, "IndicesIteratorType is not an iterator");
    static_assert(common::is_iterator<SamplesIteratorType>::value, "SamplesIteratorType is not an iterator");

  public:
    Searcher(Indexer&& query_indexer)
      : single_tree_traverser_{SingleTreeTraverser(std::forward<Indexer>(query_indexer))} {}

    template <typename Buffer>
    Buffer operator()(Buffer&& buffer) {
        static_assert(common::is_crtp_of<Buffer, buffer::StaticBase>::value,
                      "Provided a Buffer that does not inherit from StaticBase<Derived>");

        return single_tree_traverser_(std::forward<Buffer>(buffer));
    }

    std::size_t n_samples() const {
        return single_tree_traverser_.n_samples();
    }

    constexpr auto features_range_first(std::size_t sample_index) const {
        return single_tree_traverser_.features_range_first(sample_index);
    }

    constexpr auto features_range_last(std::size_t sample_index) const {
        return single_tree_traverser_.features_range_last(sample_index);
    }

  private:
    SingleTreeTraverser<Indexer> single_tree_traverser_;
};

template <typename Indexer>
Searcher(Indexer) -> Searcher<Indexer>;

}  // namespace ffcl::search