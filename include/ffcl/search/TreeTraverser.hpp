#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/StaticBase.hpp"
#include "ffcl/search/count/StaticBase.hpp"

#include "ffcl/search/buffer/Unsorted.hpp"

#include "ffcl/search/ClosestPairOfSamples.hpp"
#include "ffcl/search/MaxUpperBound.hpp"

#include <iterator>
#include <memory>
#include <optional>
#include <unordered_map>

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
                                            common::is_crtp_of<ForwardedBuffer, buffer::StaticBase>::value,
                                        bool> = true>
    ForwardedBuffer operator()(ForwardedBuffer&& forwarded_buffer) const;

    template <
        typename ForwardedBufferBatch,
        typename std::enable_if_t<common::is_iterable_of_static_base<ForwardedBufferBatch, buffer::StaticBase>::value,
                                  bool> = true>
    ForwardedBufferBatch operator()(ForwardedBufferBatch&& forwarded_buffer_batch) const;

    template <typename ForwardedQueryIndexer,
              typename std::enable_if_t<std::is_same_v<ForwardedQueryIndexer, ReferenceIndexer>, bool> = true>
    auto dual_tree_closest_edge(ForwardedQueryIndexer&& forwarded_query_indexer) const;

  private:
    template <typename Buffer>
    class IndicesToBuffersMap {
      public:
        using IndexType    = typename Buffer::IndexType;
        using DistanceType = typename Buffer::DistanceType;

        static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
        static_assert(std::is_trivial_v<DistanceType>, "DistanceType must be trivial.");

        using BufferType = Buffer;

        static_assert(common::is_crtp_of<Buffer, buffer::StaticBase>::value,
                      "Buffer must inherit from StaticBase<Derived>");

        using IndexToBufferMapType          = std::unordered_map<IndexType, BufferType>;
        using IndexToBufferMapIterator      = typename IndexToBufferMapType::iterator;
        using IndexToBufferMapConstIterator = typename std::unordered_map<IndexType, BufferType>::const_iterator;

        constexpr auto&& tightest_edge() && {
            return std::move(tightest_edge_);
        }

        constexpr auto begin() {
            return index_to_buffer_map_.begin();
        }

        constexpr auto end() {
            return index_to_buffer_map_.end();
        }

        constexpr auto begin() const {
            return index_to_buffer_map_.begin();
        }

        constexpr auto end() const {
            return index_to_buffer_map_.end();
        }

        constexpr auto cbegin() const {
            return index_to_buffer_map_.cbegin();
        }

        constexpr auto cend() const {
            return index_to_buffer_map_.cend();
        }

        constexpr auto find(const IndexType& query_index) -> IndexToBufferMapIterator {
            return index_to_buffer_map_.find(query_index);
        }

        constexpr auto find(const IndexType& query_index) const -> IndexToBufferMapConstIterator {
            return index_to_buffer_map_.find(query_index);
        }

        template <typename SamplesIterator>
        constexpr auto find_or_make_buffer_at(const IndexType&       index,
                                              const SamplesIterator& samples_range_first,
                                              const SamplesIterator& samples_range_last,
                                              std::size_t            n_features) -> IndexToBufferMapIterator {
            common::ignore_parameters(samples_range_last);

            // Attempt to find the buffer associated with the current index in the buffer map.
            auto index_to_buffer_it = this->find(index);
            // If the current index does not have an associated buffer in the map,
            if (index_to_buffer_it == this->end()) {
                auto buffer = BufferType(/**/ samples_range_first + index * n_features,
                                         /**/ samples_range_first + index * n_features + n_features,
                                         /**/ 1);
                // Attempt to insert the newly created buffer into the map. If an element with the same
                // index already exists, emplace does nothing. Otherwise, it inserts the new element.
                // The method returns a pair, where the first element is an iterator to the inserted element
                // (or to the element that prevented the insertion) and the second element is a boolean
                // indicating whether the insertion took place.
                // We are only interested in the first element of the pair.
                index_to_buffer_it = this->emplace(index, std::move(buffer)).first;
            }
            return index_to_buffer_it;
        }

        constexpr auto emplace(const IndexType& index, BufferType&& buffer)
            -> std::pair<IndexToBufferMapIterator, bool> {
            return index_to_buffer_map_.emplace(index, std::move(buffer));
        }

        template <typename QueryIndicesIterator,
                  typename QuerySamplesIterator,
                  typename ReferenceIndicesIterator,
                  typename ReferenceSamplesIterator>
        void partial_search_for_each_query(const QueryIndicesIterator&     query_indices_range_first,
                                           const QueryIndicesIterator&     query_indices_range_last,
                                           const QuerySamplesIterator&     query_samples_range_first,
                                           const QuerySamplesIterator&     query_samples_range_last,
                                           std::size_t                     query_n_features,
                                           const ReferenceIndicesIterator& reference_indices_range_first,
                                           const ReferenceIndicesIterator& reference_indices_range_last,
                                           const ReferenceSamplesIterator& reference_samples_range_first,
                                           const ReferenceSamplesIterator& reference_samples_range_last,
                                           std::size_t                     reference_n_features) {
            // Iterate through all query indices within the specified range of the query node.
            for (auto query_index_it = query_indices_range_first; query_index_it != query_indices_range_last;
                 ++query_index_it) {
                // find_or_make_buffer_at returns a query_to_buffer_it and we are only interested in the buffer
                auto& buffer_at_query_index_reference = this->find_or_make_buffer_at(/**/ *query_index_it,
                                                                                     /**/ query_samples_range_first,
                                                                                     /**/ query_samples_range_last,
                                                                                     /**/ query_n_features)
                                                            ->second;

                // Regardless of whether the buffer was just inserted or already existed, perform a partial search
                // operation on the buffer. This operation updates the buffer based on a range of reference samples.
                buffer_at_query_index_reference.partial_search(reference_indices_range_first,
                                                               reference_indices_range_last,
                                                               reference_samples_range_first,
                                                               reference_samples_range_last,
                                                               reference_n_features);

                if (buffer_at_query_index_reference.upper_bound() < std::get<2>(tightest_edge_)) {
                    tightest_edge_ = algorithms::make_edge(/**/ *query_index_it,
                                                           /**/ buffer_at_query_index_reference.upper_bound_index(),
                                                           /**/ buffer_at_query_index_reference.upper_bound());
                }
            }
        }

      private:
        IndexToBufferMapType index_to_buffer_map_;

        // query_index in the query set, reference_index in the reference set and their distance
        algorithms::Edge<IndexType, DistanceType> tightest_edge_ =
            algorithms::make_edge(common::infinity<IndexType>(),
                                  common::infinity<IndexType>(),
                                  common::infinity<DistanceType>());
    };

    template <typename Buffer>
    void single_tree_traversal(ReferenceNodePtr node, Buffer& buffer) const;

    template <typename Buffer>
    auto recursive_search_to_leaf_node(ReferenceNodePtr node, Buffer& buffer) const -> ReferenceNodePtr;

    template <typename Buffer>
    auto get_parent_node_after_sibling_traversal(const ReferenceNodePtr& node, Buffer& buffer) const
        -> ReferenceNodePtr;

    template <typename QueryNodePtr, typename QuerySamplesIterator, typename QueriesToBuffersMap>
    auto dual_tree_traversal(QueryNodePtr                query_node,
                             const QuerySamplesIterator& query_samples_range_first,
                             const QuerySamplesIterator& query_samples_range_last,
                             std::size_t                 query_n_features,
                             ReferenceNodePtr            reference_node,
                             QueriesToBuffersMap&        queries_to_buffers_map,
                             const std::tuple<algorithms::Edge<IndexType, DataType>, DataType>& cache) const
        -> std::tuple<algorithms::Edge<IndexType, DataType>, DataType>;

    template <typename QueryNodePtr, typename QuerySamplesIterator, typename QueriesToBuffersMap>
    auto update_cache_at_current_recursion_depth(
        QueryNodePtr                                                       query_node,
        const QuerySamplesIterator&                                        query_samples_range_first,
        const QuerySamplesIterator&                                        query_samples_range_last,
        std::size_t                                                        query_n_features,
        ReferenceNodePtr                                                   reference_node,
        const QueriesToBuffersMap&                                         queries_to_buffers_map,
        const std::tuple<algorithms::Edge<IndexType, DataType>, DataType>& cache) const
        -> std::tuple<algorithms::Edge<IndexType, DataType>, DataType>;

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
                                        common::is_crtp_of<ForwardedBuffer, buffer::StaticBase>::value,
                                    bool>>
ForwardedBuffer TreeTraverser<ReferenceIndexer>::operator()(ForwardedBuffer&& forwarded_buffer) const {
    auto buffer = std::forward<ForwardedBuffer>(forwarded_buffer);

    single_tree_traversal(reference_indexer_.root(), buffer);

    return buffer;
}

template <typename ReferenceIndexer>
template <typename ForwardedBufferBatch,
          typename std::enable_if_t<common::is_iterable_of_static_base<ForwardedBufferBatch, buffer::StaticBase>::value,
                                    bool>>
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
          typename std::enable_if_t<std::is_same_v<ForwardedQueryIndexer, ReferenceIndexer>, bool>>
auto TreeTraverser<ReferenceIndexer>::dual_tree_closest_edge(ForwardedQueryIndexer&& forwarded_query_indexer) const {
    using QuerySamplesIteratorType = typename ForwardedQueryIndexer::SamplesIteratorType;

    static_assert(common::is_iterator<QuerySamplesIteratorType>::value, "QuerySamplesIteratorType is not an iterator");

    auto query_indexer = std::forward<ForwardedQueryIndexer>(forwarded_query_indexer);

    auto queries_to_buffers_map = IndicesToBuffersMap<buffer::Unsorted<QuerySamplesIteratorType>>{};

    auto shortest_edge = algorithms::make_edge(/**/ common::infinity<IndexType>(),
                                               /**/ common::infinity<IndexType>(),
                                               /**/ common::infinity<DataType>());

    auto queries_max_upper_bound = common::infinity<DataType>();

    const auto cache = std::make_tuple(std::move(shortest_edge), std::move(queries_max_upper_bound));

    dual_tree_traversal(query_indexer.root(),
                        query_indexer.begin(),
                        query_indexer.end(),
                        query_indexer.n_features(),
                        reference_indexer_.root(),
                        queries_to_buffers_map,
                        cache);

    return std::move(queries_to_buffers_map).tightest_edge();
}

template <typename ReferenceIndexer>
template <typename QueryNodePtr, typename QuerySamplesIterator, typename QueriesToBuffersMap>
auto TreeTraverser<ReferenceIndexer>::dual_tree_traversal(
    QueryNodePtr                                                       query_node,
    const QuerySamplesIterator&                                        query_samples_range_first,
    const QuerySamplesIterator&                                        query_samples_range_last,
    std::size_t                                                        query_n_features,
    ReferenceNodePtr                                                   reference_node,
    QueriesToBuffersMap&                                               queries_to_buffers_map,
    const std::tuple<algorithms::Edge<IndexType, DataType>, DataType>& cache) const
    -> std::tuple<algorithms::Edge<IndexType, DataType>, DataType> {
    auto updated_cache = update_cache_at_current_recursion_depth(query_node,
                                                                 query_samples_range_first,
                                                                 query_samples_range_last,
                                                                 query_n_features,
                                                                 reference_node,
                                                                 queries_to_buffers_map,
                                                                 cache);

    const auto& [shortest_edge, queries_max_upper_bound] = updated_cache;

    if (std::get<2>(shortest_edge) >= queries_max_upper_bound) {
        return updated_cache;
    }
    // updates the buffers w.r.t. each query and each reference while storing the shortest edge
    queries_to_buffers_map.partial_search_for_each_query(query_node->indices_range_.first,
                                                         query_node->indices_range_.second,
                                                         query_samples_range_first,
                                                         query_samples_range_last,
                                                         query_n_features,
                                                         reference_node->indices_range_.first,
                                                         reference_node->indices_range_.second,
                                                         reference_indexer_.begin(),
                                                         reference_indexer_.end(),
                                                         reference_indexer_.n_features());

    if (!query_node->is_leaf()) {
        if (!reference_node->is_leaf()) {
            updated_cache = dual_tree_traversal(query_node->left_,
                                                query_samples_range_first,
                                                query_samples_range_last,
                                                query_n_features,
                                                reference_node->left_,
                                                queries_to_buffers_map,
                                                updated_cache);

            updated_cache = dual_tree_traversal(query_node->left_,
                                                query_samples_range_first,
                                                query_samples_range_last,
                                                query_n_features,
                                                reference_node->right_,
                                                queries_to_buffers_map,
                                                updated_cache);

            updated_cache = dual_tree_traversal(query_node->right_,
                                                query_samples_range_first,
                                                query_samples_range_last,
                                                query_n_features,
                                                reference_node->left_,
                                                queries_to_buffers_map,
                                                updated_cache);

            updated_cache = dual_tree_traversal(query_node->right_,
                                                query_samples_range_first,
                                                query_samples_range_last,
                                                query_n_features,
                                                reference_node->right_,
                                                queries_to_buffers_map,
                                                updated_cache);
        } else {
            updated_cache = dual_tree_traversal(query_node->left_,
                                                query_samples_range_first,
                                                query_samples_range_last,
                                                query_n_features,
                                                reference_node,
                                                queries_to_buffers_map,
                                                updated_cache);

            updated_cache = dual_tree_traversal(query_node->right_,
                                                query_samples_range_first,
                                                query_samples_range_last,
                                                query_n_features,
                                                reference_node,
                                                queries_to_buffers_map,
                                                updated_cache);
        }
    } else {
        if (!reference_node->is_leaf()) {
            updated_cache = dual_tree_traversal(query_node,
                                                query_samples_range_first,
                                                query_samples_range_last,
                                                query_n_features,
                                                reference_node->left_,
                                                queries_to_buffers_map,
                                                updated_cache);

            updated_cache = dual_tree_traversal(query_node,
                                                query_samples_range_first,
                                                query_samples_range_last,
                                                query_n_features,
                                                reference_node->right_,
                                                queries_to_buffers_map,
                                                updated_cache);
        }
    }
    // if query_node and reference_node are both leaf, return the cache without further recursion.
    return updated_cache;
}

template <typename ReferenceIndexer>
template <typename QueryNodePtr, typename QuerySamplesIterator, typename QueriesToBuffersMap>
auto TreeTraverser<ReferenceIndexer>::update_cache_at_current_recursion_depth(
    QueryNodePtr                                                       query_node,
    const QuerySamplesIterator&                                        query_samples_range_first,
    const QuerySamplesIterator&                                        query_samples_range_last,
    std::size_t                                                        query_n_features,
    ReferenceNodePtr                                                   reference_node,
    const QueriesToBuffersMap&                                         queries_to_buffers_map,
    const std::tuple<algorithms::Edge<IndexType, DataType>, DataType>& cache) const
    -> std::tuple<algorithms::Edge<IndexType, DataType>, DataType> {
    const auto& [shortest_edge, queries_max_upper_bound] = cache;

    const auto updated_shortest_edge = algorithms::dual_set_shortest_edge(query_node->indices_range_.first,
                                                                          query_node->indices_range_.second,
                                                                          query_samples_range_first,
                                                                          query_samples_range_last,
                                                                          query_n_features,
                                                                          reference_node->indices_range_.first,
                                                                          reference_node->indices_range_.second,
                                                                          reference_indexer_.begin(),
                                                                          reference_indexer_.end(),
                                                                          reference_indexer_.n_features(),
                                                                          shortest_edge);

    const auto updated_queries_max_upper_bound =
        algorithms::find_max_upper_bound(/**/ query_node->indices_range_.first,
                                         /**/ query_node->indices_range_.second,
                                         /**/ queries_to_buffers_map,
                                         /**/ queries_max_upper_bound);

    return make_tuple(updated_shortest_edge, updated_queries_max_upper_bound);
}

}  // namespace ffcl::search