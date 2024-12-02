#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"
#include "ffcl/datastruct/bounds/distances/MinDistance.hpp"
#include "ffcl/datastruct/graph/spanning_tree/MinimumSpanningTree.hpp"  // for ffcl::datastruct::mst::Edge

#include "ffcl/search/buffer/EdgesBuffer.hpp"  // Just for custom hash and nodes combination datastruct etc

#include "ffcl/search/buffer/Unsorted.hpp"
#include "ffcl/search/buffer/WithMemory.hpp"
#include "ffcl/search/buffer/WithUnionFind.hpp"

#include <cstddef>
#include <optional>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "ffcl/common/Timer.hpp"

namespace ffcl::search::buffer {

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
class EdgesBuffer {
  public:
    using IndexType    = typename QueryIndexer::IndexType;
    using DistanceType = typename QueryIndexer::DataType;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DistanceType>, "DistanceType must be trivial.");

    using IndicesToBuffersMapType          = std::unordered_map<IndexType, Buffer>;
    using IndicesToBuffersMapIterator      = typename IndicesToBuffersMapType::iterator;
    using IndicesToBuffersMapConstIterator = typename IndicesToBuffersMapType::const_iterator;

    using QueryNodePtr     = typename QueryIndexer::NodePtr;
    using ReferenceNodePtr = typename ReferenceIndexer::NodePtr;

    using QuerySamplesIteratorType     = typename QueryIndexer::SamplesIteratorType;
    using ReferenceSamplesIteratorType = typename ReferenceIndexer::SamplesIteratorType;

  public:
    EdgesBuffer(const QueryIndexer& query_indexer, const ReferenceIndexer& reference_indexer);

    auto tightest_edge() const;

    auto component_to_shortest_edge_map() const;

  private:
    QuerySamplesIteratorType query_samples_range_first_;
    QuerySamplesIteratorType query_samples_range_last_;
    std::size_t              query_n_features_;

    ReferenceSamplesIteratorType reference_samples_range_first_;
    ReferenceSamplesIteratorType reference_samples_range_last_;
    std::size_t                  reference_n_features_;
};

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
auto make_edge_buffer(const QueryIndexer& query_indexer, const ReferenceIndexer& reference_indexer)
    -> EdgesBuffer<Buffer, QueryIndexer, ReferenceIndexer> {
    return EdgesBuffer<Buffer, QueryIndexer, ReferenceIndexer>(query_indexer, reference_indexer);
}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
EdgesBuffer<Buffer, QueryIndexer, ReferenceIndexer>::EdgesBuffer(const QueryIndexer&     query_indexer,
                                                                 const ReferenceIndexer& reference_indexer)
  : query_samples_range_first_{query_indexer.begin()}
  , query_samples_range_last_{query_indexer.end()}
  , query_n_features_{query_indexer.n_features()}
  , reference_samples_range_first_{reference_indexer.begin()}
  , reference_samples_range_last_{reference_indexer.end()}
  , reference_n_features_{reference_indexer.n_features()} {}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<Buffer, QueryIndexer, ReferenceIndexer>::tightest_edge() const {
    return datastruct::mst::make_infinity_edge<IndexType, DistanceType>();
}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
auto EdgesBuffer<Buffer, QueryIndexer, ReferenceIndexer>::component_to_shortest_edge_map() const {
    using EdgeType = datastruct::mst::Edge<IndexType, DistanceType>;

    return std::unordered_map<IndexType, EdgeType>{};
}

}  // namespace ffcl::search::buffer