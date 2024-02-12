#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/StaticBase.hpp"
#include "ffcl/search/count/StaticBase.hpp"

#include "ffcl/search/buffer/Unsorted.hpp"

#include <iterator>
#include <memory>
#include <unordered_map>

namespace ffcl::search {

/*
template <typename ReferenceIndex, typename Buffer>
class BufferMap {
  public:
    using ReferenceIndexType = ReferenceIndex;

    static_assert(std::is_trivial_v<ReferenceIndexType>, "ReferenceIndexType must be trivial.");

    static_assert(common::is_crtp_of<Buffer, buffer::StaticBase>::value,
                  "Buffer must inherit from StaticBase<Derived>");

    Buffer& operator[](const ReferenceIndex& reference_index);

    auto find(const ReferenceIndex& reference_index);

  private:
    std::unordered_map<ReferenceIndexType, Buffer> query_index_to_buffer_map_;
};

template <typename ReferenceIndex, typename Buffer>
Buffer& BufferMap<ReferenceIndex, Buffer>::operator[](const ReferenceIndex& reference_index) {
    return query_index_to_buffer_map_[reference_index];
}

template <typename ReferenceIndex, typename Buffer>
auto BufferMap<ReferenceIndex, Buffer>::find(const ReferenceIndex& reference_index) {
    return query_index_to_buffer_map_.find(reference_index);
}
*/

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

    static_assert(common::is_raw_or_smart_ptr<ReferenceNodePtr>(), "ReferenceNodePtr is not a raw or smart pointer");

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
              typename std::enable_if_t<!common::is_iterable_v<ForwardedQueryIndexer> &&
                                            !common::is_crtp_of<ForwardedQueryIndexer, buffer::StaticBase>::value,
                                        bool> = true>
    ForwardedQueryIndexer operator()(ForwardedQueryIndexer&& forwarded_query_indexer) const;

  private:
    template <typename Buffer>
    void single_tree_traversal(ReferenceNodePtr node, Buffer& buffer) const;

    template <typename Buffer>
    auto recursive_search_to_leaf_node(ReferenceNodePtr node, Buffer& buffer) const -> ReferenceNodePtr;

    template <typename Buffer>
    auto get_parent_node_after_sibling_traversal(const ReferenceNodePtr& node, Buffer& buffer) const
        -> ReferenceNodePtr;

    template <typename QueryNodePtr, typename SamplesIterator, typename BufferMap>
    void dual_tree_traversal(ReferenceNodePtr       reference_node,
                             QueryNodePtr           query_node,
                             const SamplesIterator& query_samples_range_first,
                             const SamplesIterator& query_samples_range_last,
                             std::size_t            query_n_features,
                             BufferMap&             buffer_map) const;

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
template <typename ForwardedQueryIndexer,
          typename std::enable_if_t<!common::is_iterable_v<ForwardedQueryIndexer> &&
                                        !common::is_crtp_of<ForwardedQueryIndexer, buffer::StaticBase>::value,
                                    bool>>
ForwardedQueryIndexer TreeTraverser<ReferenceIndexer>::operator()(
    ForwardedQueryIndexer&& forwarded_query_indexer) const {
    using QueryIndexType = typename ForwardedQueryIndexer::IndexType;

    static_assert(std::is_trivial_v<QueryIndexType>, "QueryIndexType must be trivial.");

    using QuerySamplesIteratorType = typename ForwardedQueryIndexer::SamplesIteratorType;

    static_assert(common::is_iterator<QuerySamplesIteratorType>::value, "QuerySamplesIteratorType is not an iterator");

    auto query_indexer = std::forward<ForwardedQueryIndexer>(forwarded_query_indexer);

    auto buffer_map = std::unordered_map<QueryIndexType, buffer::Unsorted<QuerySamplesIteratorType>>{};

    dual_tree_traversal(reference_indexer_.root(),
                        query_indexer.root(),
                        query_indexer.begin(),
                        query_indexer.end(),
                        query_indexer.n_samples(),
                        buffer_map);

    return std::forward<ForwardedQueryIndexer>(query_indexer);
}

template <typename ReferenceIndexer>
template <typename QueryNodePtr, typename SamplesIterator, typename BufferMap>
void TreeTraverser<ReferenceIndexer>::dual_tree_traversal(ReferenceNodePtr       reference_node,
                                                          QueryNodePtr           query_node,
                                                          const SamplesIterator& query_samples_range_first,
                                                          const SamplesIterator& query_samples_range_last,
                                                          std::size_t            query_n_features,
                                                          BufferMap&             buffer_map) const {
    common::ignore_parameters(query_samples_range_last);

    if (true) {
        return;
    }
    // Iterate through all query indices within the specified range of the query node.
    for (auto query_index_it = query_node->indices_range_.first; query_index_it != query_node->indices_range_.second;
         ++query_index_it) {
        // Attempt to find the buffer associated with the current query index in the buffer map.
        auto query_and_buffer_pair_it = buffer_map.find(*query_index_it);
        // If the current query index does not have an associated buffer in the map,
        if (query_and_buffer_pair_it == buffer_map.end()) {
            // create a new Unsorted buffer
            auto unsorted_buffer =
                buffer::Unsorted(query_samples_range_first + *query_index_it * query_n_features,
                                 query_samples_range_first + *query_index_it * query_n_features + query_n_features);
            // Attempt to insert the newly created buffer into the map. If an element with the same
            // query index already exists, emplace does nothing. Otherwise, it inserts the new element.
            // The method returns a pair, where the first element is an iterator to the inserted element
            // (or to the element that prevented the insertion) and the second element is a boolean
            // indicating whether the insertion took place.
            // We are only interested in the first element of the pair.
            query_and_buffer_pair_it = buffer_map.emplace(*query_index_it, std::move(unsorted_buffer)).first;
        }
        // Regardless of whether the buffer was just inserted or already existed, perform a partial search
        // operation on the buffer. This operation updates the buffer based on a range of reference indices,
        // the beginning and end of the reference indexer, and the number of features in the reference indexer.
        query_and_buffer_pair_it->second.partial_search(reference_node->indices_range_.first,
                                                        reference_node->indices_range_.second,
                                                        reference_indexer_.begin(),
                                                        reference_indexer_.end(),
                                                        reference_indexer_.n_features());
    }
    dual_tree_traversal(reference_node->left_,
                        query_node->left_,
                        query_samples_range_first,
                        query_samples_range_last,
                        query_n_features,
                        buffer_map);

    dual_tree_traversal(reference_node->left_,
                        query_node->right_,
                        query_samples_range_first,
                        query_samples_range_last,
                        query_n_features,
                        buffer_map);

    dual_tree_traversal(reference_node->right_,
                        query_node->left_,
                        query_samples_range_first,
                        query_samples_range_last,
                        query_n_features,
                        buffer_map);

    dual_tree_traversal(reference_node->right_,
                        query_node->right_,
                        query_samples_range_first,
                        query_samples_range_last,
                        query_n_features,
                        buffer_map);
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
        reference_node = recursive_search_to_leaf_node(reference_node->step_down(reference_indexer_.begin(),
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

}  // namespace ffcl::search