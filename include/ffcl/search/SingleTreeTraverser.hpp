#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/StaticBase.hpp"
#include "ffcl/search/count/StaticBase.hpp"

namespace ffcl::search {

template <typename ReferenceIndexer>
class SingleTreeTraverser {
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

    SingleTreeTraverser(const ReferenceIndexer& reference_indexer)
      : reference_indexer_{reference_indexer} {}

    SingleTreeTraverser(ReferenceIndexer&& reference_indexer)
      : reference_indexer_{std::move(reference_indexer)} {}

    template <typename Buffer>
    Buffer operator()(Buffer&& input_buffer) const {
        static_assert(common::is_crtp_of<Buffer, buffer::StaticBase>::value,
                      "Provided a Buffer that does not inherit from StaticBase<Derived>");

        auto processed_buffer = std::forward<Buffer>(input_buffer);
        single_tree_traversal(reference_indexer_.root(), processed_buffer);
        return processed_buffer;
    }

    std::size_t n_samples() const {
        return reference_indexer_.n_samples();
    }

    constexpr auto features_range_first(std::size_t sample_index) const {
        return reference_indexer_.features_range_first(sample_index);
    }

    constexpr auto features_range_last(std::size_t sample_index) const {
        return reference_indexer_.features_range_last(sample_index);
    }

  private:
    template <typename Buffer>
    void single_tree_traversal(KDNodeViewPtr node, Buffer& buffer) const {
        // current_node is a leaf node (and root in the special case where the entire tree is in a single node)
        auto current_node = recursive_search_to_leaf_node(node, buffer);
        // performs a partial search one step at a time from the leaf node until the input node is reached if
        // node parameter is a subtree
        while (current_node != node) {
            // performs a partial search starting from the specified node then returns its parent if it exists
            // (nullptr otherwise)
            current_node = get_parent_node_after_sibling_traversal(current_node, buffer);
        }
    }

    template <typename Buffer>
    auto recursive_search_to_leaf_node(KDNodeViewPtr node, Buffer& buffer) const -> KDNodeViewPtr {
        buffer.partial_search(node->indices_range_.first,
                              node->indices_range_.second,
                              reference_indexer_.begin(),
                              reference_indexer_.end(),
                              reference_indexer_.n_features());
        // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
        if (!node->is_leaf()) {
            node = recursive_search_to_leaf_node(node->step_down(reference_indexer_.begin(),
                                                                 reference_indexer_.end(),
                                                                 reference_indexer_.n_features(),
                                                                 buffer.centroid_begin(),
                                                                 buffer.centroid_end()),
                                                 buffer);
        }
        return node;
    }

    template <typename Buffer>
    auto get_parent_node_after_sibling_traversal(const KDNodeViewPtr& node, Buffer& buffer) const -> KDNodeViewPtr {
        // if a sibling node has been selected (is not nullptr)
        if (auto sibling_node = node->select_sibling_node(/**/ reference_indexer_.begin(),
                                                          /**/ reference_indexer_.end(),
                                                          /**/ reference_indexer_.n_features(),
                                                          /**/ buffer)) {
            // search for other candidates from the sibling node
            single_tree_traversal(sibling_node, buffer);
        }
        // returns nullptr if node doesnt have parent (or is root)
        return node->parent_.lock();
    }

    ReferenceIndexer reference_indexer_;
};

}  // namespace ffcl::search