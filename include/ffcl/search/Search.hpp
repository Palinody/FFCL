#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/Base.hpp"
#include "ffcl/search/count/Base.hpp"

#include "ffcl/search/SingleTreeTraverser.hpp"

namespace ffcl::search {

template <typename Indexer>
class Searcher {
  public:
    // static_assert(common::is_raw_or_smart_ptr<IndexerPtr>());

    // using IndexType           = typename Indexer::element_type::IndexType;
    // using DataType            = typename Indexer::element_type::DataType;
    // using IndicesIteratorType = typename Indexer::element_type::IndicesIteratorType;
    // using SamplesIteratorType = typename Indexer::element_type::SamplesIteratorType;

    using IndexType           = typename Indexer::IndexType;
    using DataType            = typename Indexer::DataType;
    using IndicesIteratorType = typename Indexer::IndicesIteratorType;
    using SamplesIteratorType = typename Indexer::SamplesIteratorType;

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