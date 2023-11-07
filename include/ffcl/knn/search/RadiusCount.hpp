#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/math/heuristics/Distances.hpp"

#include "ffcl/knn/count/Radius.hpp"

namespace ffcl::knn::count {

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_radius(
    const IndicesIterator&                                                                   indices_range_first,
    const IndicesIterator&                                                                   indices_range_last,
    const SamplesIterator&                                                                   samples_range_first,
    const SamplesIterator&                                                                   samples_range_last,
    std::size_t                                                                              n_features,
    std::size_t                                                                              sample_index_query,
    count::Base<typename IndicesIterator::value_type, typename SamplesIterator::value_type>& buffer) {
    common::utils::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(samples_range_first + sample_index_query * n_features,
                                                samples_range_first + sample_index_query * n_features + n_features,
                                                samples_range_first + candidate_nearest_neighbor_index * n_features);

            buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_radius(
    const IndicesIterator&                                                                   indices_range_first,
    const IndicesIterator&                                                                   indices_range_last,
    const SamplesIterator&                                                                   samples_range_first,
    const SamplesIterator&                                                                   samples_range_last,
    std::size_t                                                                              n_features,
    const SamplesIterator&                                                                   feature_query_range_first,
    const SamplesIterator&                                                                   feature_query_range_last,
    count::Base<typename IndicesIterator::value_type, typename SamplesIterator::value_type>& buffer) {
    common::utils::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        const auto candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(feature_query_range_first,
                                            feature_query_range_last,
                                            samples_range_first + candidate_nearest_neighbor_index * n_features);

        buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
    }
}

template <typename KDTreePtr>
class SingleTreeTraverser {
  public:
    using IndexType           = typename KDTreePtr::element_type::IndexType;
    using DataType            = typename KDTreePtr::element_type::DataType;
    using IndicesIteratorType = typename KDTreePtr::element_type::IndicesIteratorType;
    using SamplesIteratorType = typename KDTreePtr::element_type::SamplesIteratorType;

    SingleTreeTraverser(KDTreePtr query_kdtree_ptr)
      : query_kdtree_ptr_{query_kdtree_ptr} {}

    template <typename BufferType>
    BufferType operator()(std::size_t query_index, BufferType& buffer) {
        single_tree_traversal(query_index, buffer, query_kdtree_ptr_->root());
        return buffer;
    }

    template <typename BufferType>
    BufferType operator()(const SamplesIteratorType& query_feature_first,
                          const SamplesIteratorType& query_feature_last,
                          BufferType&                buffer) {
        single_tree_traversal(query_feature_first, query_feature_last, buffer, query_kdtree_ptr_->root());
        return buffer;
    }

  private:
    template <typename BufferType>
    void single_tree_traversal(std::size_t                                     query_index,
                               BufferType&                                     buffer,
                               typename KDTreePtr::element_type::KDNodeViewPtr node) {
        // current_node is currently a leaf node (and root in the special case where the entire tree is in a single
        // node)
        auto current_kdnode = recurse_to_closest_leaf_node(
            /**/ query_index,
            /**/ buffer,
            /**/ node);

        // performs a nearest neighbor search one step at a time from the leaf node until the input node is reached if
        // node parameter is a subtree. A search through the entire tree
        while (current_kdnode != node) {
            // performs a nearest neighbor search starting from the specified node then returns its parent if it exists
            // (nullptr otherwise)
            current_kdnode = get_parent_node_after_sibling_traversal(
                /**/ query_index,
                /**/ buffer,
                /**/ current_kdnode);
        }
    }

    template <typename BufferType>
    auto recurse_to_closest_leaf_node(std::size_t                                     query_index,
                                      BufferType&                                     buffer,
                                      typename KDTreePtr::element_type::KDNodeViewPtr node) {
        knn::count::increment_neighbors_count_in_radius(node->indices_range_.first,
                                                        node->indices_range_.second,
                                                        query_kdtree_ptr_->begin(),
                                                        query_kdtree_ptr_->end(),
                                                        query_kdtree_ptr_->n_features(),
                                                        query_index,
                                                        buffer);

        // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
        if (!node->is_leaf()) {
            // get the pivot sample index in the dataset
            const auto pivot_index = node->indices_range_.first[0];
            // get the split value according to the current split dimension
            const auto pivot_split_value =
                (*query_kdtree_ptr_)[pivot_index * query_kdtree_ptr_->n_features() + node->cut_feature_index_];
            // get the value of the query according to the split dimension
            const auto query_split_value =
                (*query_kdtree_ptr_)[query_index * query_kdtree_ptr_->n_features() + node->cut_feature_index_];

            // traverse either the left or right child node depending on where the target sample is located relatively
            // to the cut value
            if (query_split_value < pivot_split_value) {
                node = recurse_to_closest_leaf_node(
                    /**/ query_index,
                    /**/ buffer,
                    /**/ node->left_);
            } else {
                node = recurse_to_closest_leaf_node(
                    /**/ query_index,
                    /**/ buffer,
                    /**/ node->right_);
            }
        }
        return node;
    }

    template <typename BufferType>
    auto get_parent_node_after_sibling_traversal(std::size_t                                     query_index,
                                                 BufferType&                                     buffer,
                                                 typename KDTreePtr::element_type::KDNodeViewPtr node) {
        auto parent_node = node->parent_.lock();
        // if node has a parent
        if (parent_node) {
            // get the pivot sample index in the dataset
            const auto pivot_index = parent_node->indices_range_.first[0];
            // get the split value according to the current split dimension
            const auto pivot_split_value =
                (*query_kdtree_ptr_)[pivot_index * query_kdtree_ptr_->n_features() + parent_node->cut_feature_index_];
            // get the value of the query according to the split dimension
            const auto query_split_value =
                (*query_kdtree_ptr_)[query_index * query_kdtree_ptr_->n_features() + parent_node->cut_feature_index_];
            // if the axiswise distance is equal to the current furthest nearest neighbor distance, there could be a
            // nearest neighbor to the other side of the hyperrectangle since the values that are equal to the pivot are
            // put to the right
            bool visit_sibling =
                node->is_left_child()
                    ? buffer.n_free_slots() || common::utils::abs(pivot_split_value - query_split_value) <=
                                                   buffer.upper_bound(parent_node->cut_feature_index_)
                    : buffer.n_free_slots() || common::utils::abs(pivot_split_value - query_split_value) <
                                                   buffer.upper_bound(parent_node->cut_feature_index_);
            // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
            // closer to the query sample than the current nearest neighbor
            if (visit_sibling) {
                // if the sibling node is not nullptr
                if (auto sibling_node = node->get_sibling_node()) {
                    // get the nearest neighbor from the sibling node
                    single_tree_traversal(
                        /**/ query_index,
                        /**/ buffer,
                        /**/ sibling_node);
                }
            }
        }
        // returns nullptr if node doesnt have parent (or is root)
        return parent_node;
    }

    template <typename BufferType>
    void single_tree_traversal(const SamplesIteratorType&                      query_feature_first,
                               const SamplesIteratorType&                      query_feature_last,
                               BufferType&                                     buffer,
                               typename KDTreePtr::element_type::KDNodeViewPtr node) {
        // current_node is currently a leaf node (and root in the special case where the entire tree is in a single
        // node)
        auto current_kdnode = recurse_to_closest_leaf_node(
            /**/ query_feature_first,
            /**/ query_feature_last,
            /**/ buffer,
            /**/ node);

        // performs a nearest neighbor search one step at a time from the leaf node until the input node is reached if
        // node parameter is a subtree. A search through the entire tree
        while (current_kdnode != node) {
            // performs a nearest neighbor search starting from the specified node then returns its parent if it exists
            // (nullptr otherwise)
            current_kdnode = get_parent_node_after_sibling_traversal(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ buffer,
                /**/ current_kdnode);
        }
    }

    template <typename BufferType>
    auto recurse_to_closest_leaf_node(const SamplesIteratorType&                      query_feature_first,
                                      const SamplesIteratorType&                      query_feature_last,
                                      BufferType&                                     buffer,
                                      typename KDTreePtr::element_type::KDNodeViewPtr node) {
        knn::count::increment_neighbors_count_in_radius(node->indices_range_.first,
                                                        node->indices_range_.second,
                                                        query_kdtree_ptr_->begin(),
                                                        query_kdtree_ptr_->end(),
                                                        query_kdtree_ptr_->n_features(),
                                                        query_feature_first,
                                                        query_feature_last,
                                                        buffer);

        // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
        if (!node->is_leaf()) {
            // get the pivot sample index in the dataset
            const auto pivot_index = node->indices_range_.first[0];
            // get the split value according to the current split dimension
            const auto pivot_split_value =
                (*query_kdtree_ptr_)[pivot_index * query_kdtree_ptr_->n_features() + node->cut_feature_index_];
            // get the value of the query according to the split dimension
            const auto query_split_value = query_feature_first[node->cut_feature_index_];

            // traverse either the left or right child node depending on where the target sample is located relatively
            // to the cut value
            if (query_split_value < pivot_split_value) {
                node = recurse_to_closest_leaf_node(
                    /**/ query_feature_first,
                    /**/ query_feature_last,
                    /**/ buffer,
                    /**/ node->left_);
            } else {
                node = recurse_to_closest_leaf_node(
                    /**/ query_feature_first,
                    /**/ query_feature_last,
                    /**/ buffer,
                    /**/ node->right_);
            }
        }
        return node;
    }

    template <typename BufferType>
    auto get_parent_node_after_sibling_traversal(const SamplesIteratorType&                      query_feature_first,
                                                 const SamplesIteratorType&                      query_feature_last,
                                                 BufferType&                                     buffer,
                                                 typename KDTreePtr::element_type::KDNodeViewPtr node) {
        auto parent_node = node->parent_.lock();
        // if node has a parent
        if (parent_node) {
            // get the pivot sample index in the dataset
            const auto pivot_index = parent_node->indices_range_.first[0];
            // get the split value according to the current split dimension
            const auto pivot_split_value =
                (*query_kdtree_ptr_)[pivot_index * query_kdtree_ptr_->n_features() + parent_node->cut_feature_index_];
            // get the value of the query according to the split dimension
            const auto query_split_value = query_feature_first[parent_node->cut_feature_index_];
            // if the axiswise distance is equal to the current furthest nearest neighbor distance, there could be a
            // nearest neighbor to the other side of the hyperrectangle since the values that are equal to the pivot are
            // put to the right
            bool visit_sibling =
                node->is_left_child()
                    ? buffer.n_free_slots() || common::utils::abs(pivot_split_value - query_split_value) <=
                                                   buffer.upper_bound(parent_node->cut_feature_index_)
                    : buffer.n_free_slots() || common::utils::abs(pivot_split_value - query_split_value) <
                                                   buffer.upper_bound(parent_node->cut_feature_index_);
            // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
            // closer to the query sample than the current nearest neighbor
            if (visit_sibling) {
                // if the sibling node is not nullptr
                if (auto sibling_node = node->get_sibling_node()) {
                    // get the nearest neighbor from the sibling node
                    single_tree_traversal(
                        /**/ query_feature_first,
                        /**/ query_feature_last,
                        /**/ buffer,
                        /**/ sibling_node);
                }
            }
        }
        // returns nullptr if node doesnt have parent (or is root)
        return parent_node;
    }

    KDTreePtr query_kdtree_ptr_;
};

template <typename KDTreePtr>
class Counter {
  public:
    using IndexType           = typename KDTreePtr::element_type::IndexType;
    using DataType            = typename KDTreePtr::element_type::DataType;
    using IndicesIteratorType = typename KDTreePtr::element_type::IndicesIteratorType;
    using SamplesIteratorType = typename KDTreePtr::element_type::SamplesIteratorType;

    Counter(KDTreePtr query_kdtree_ptr, const knn::count::Radius<IndexType, DataType>& counter)
      : query_kdtree_ptr_{query_kdtree_ptr}
      , counter_{counter} {}

    knn::count::Radius<IndexType, DataType> operator()(std::size_t query_index) const {
        return SingleTreeTraverser(query_kdtree_ptr_)(query_index, counter_);
    }

    knn::count::Radius<IndexType, DataType> operator()(const SamplesIteratorType& query_feature_first,
                                                       const SamplesIteratorType& query_feature_last) const {
        return SingleTreeTraverser(query_kdtree_ptr_)(query_feature_first, query_feature_last, counter_);
    }

  private:
    KDTreePtr                               query_kdtree_ptr_;
    knn::count::Radius<IndexType, DataType> counter_;
};

}  // namespace ffcl::knn::count