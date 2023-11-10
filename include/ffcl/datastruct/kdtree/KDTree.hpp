#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/kdtree/KDNodeView.hpp"
#include "ffcl/datastruct/kdtree/policy/AxisSelectionPolicy.hpp"
#include "ffcl/datastruct/kdtree/policy/SplittingRulePolicy.hpp"

#include "ffcl/common/math/random/Distributions.hpp"

#include "ffcl/knn/buffer/Base.hpp"
#include "ffcl/knn/buffer/Radius.hpp"
#include "ffcl/knn/buffer/Range.hpp"
#include "ffcl/knn/buffer/Singleton.hpp"
#include "ffcl/knn/buffer/Unsorted.hpp"

#include "ffcl/knn/count/Radius.hpp"
#include "ffcl/knn/count/Range.hpp"

#include "ffcl/knn/search/KNearestNeighborsSearch.hpp"
// #include "ffcl/knn/search/NearestNeighborSearch.hpp"
#include "ffcl/knn/search/RadiusCount.hpp"
// #include "ffcl/knn/search/RadiusSearch.hpp"
#include "ffcl/knn/search/RangeCount.hpp"
// #include "ffcl/knn/search/RangeSearch.hpp"

#include <sys/types.h>  // ssize_t
#include <algorithm>
#include <array>
#include <memory>
#include <tuple>

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/stream.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include <filesystem>
#include <fstream>

namespace ffcl::datastruct {

namespace fs = std::filesystem;

template <typename IndicesIterator, typename SamplesIterator>
class KDTree {
  public:
    using IndexType           = typename IndicesIterator::value_type;
    using DataType            = typename SamplesIterator::value_type;
    using IndicesIteratorType = IndicesIterator;
    using SamplesIteratorType = SamplesIterator;
    using KDNodeViewType      = typename KDNodeView<IndicesIterator, SamplesIterator>::KDNodeViewType;
    using KDNodeViewPtr       = typename KDNodeView<IndicesIterator, SamplesIterator>::KDNodeViewPtr;
    using HyperRangeType      = ffcl::bbox::HyperRangeType<SamplesIterator>;

    struct Options {
        Options()
          : bucket_size_{40}
          , max_depth_{common::infinity<ssize_t>()}
          , axis_selection_policy_ptr_{std::make_shared<
                kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>>()}
          , splitting_rule_policy_ptr_{
                std::make_shared<kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>>()} {}

        Options(const Options&) = default;

        Options& operator=(const Options& options) = default;

        Options& bucket_size(std::size_t bucket_size) {
            bucket_size_ = bucket_size;
            return *this;
        }

        Options& max_depth(ssize_t max_depth) {
            max_depth_ = max_depth;
            return *this;
        }

        template <typename AxisSelectionPolicy>
        Options& axis_selection_policy(const AxisSelectionPolicy& axis_selection_policy) {
            static_assert(std::is_base_of<kdtree::policy::AxisSelectionPolicy<IndicesIterator, SamplesIterator>,
                                          AxisSelectionPolicy>::value,
                          "The provided axis selection policy must be derived from "
                          "kdtree::policy::AxisSelectionPolicy<IndicesIterator, SamplesIterator>");

            axis_selection_policy_ptr_ = std::make_shared<AxisSelectionPolicy>(axis_selection_policy);
            return *this;
        }

        template <typename SplittingRulePolicy>
        Options& splitting_rule_policy(const SplittingRulePolicy& splitting_rule_policy) {
            static_assert(std::is_base_of<kdtree::policy::SplittingRulePolicy<IndicesIterator, SamplesIterator>,
                                          SplittingRulePolicy>::value,
                          "The provided splitting rule policy must be derived from "
                          "kdtree::policy::SplittingRulePolicy<IndicesIterator, SamplesIterator>");

            splitting_rule_policy_ptr_ = std::make_shared<SplittingRulePolicy>(splitting_rule_policy);
            return *this;
        }

        void serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
            writer.String("options");

            writer.StartObject();
            {
                writer.String("bucket_size");
                writer.Int64(bucket_size_);

                writer.String("max_depth");
                writer.Int64(max_depth_);
            }
            writer.EndObject();
        }
        // the maximum number of samples per leaf node
        std::size_t bucket_size_;
        // the maximum recursion depth
        ssize_t max_depth_;
        // the policy that will be responsible for selecting the axis at a given kdtree depth
        std::shared_ptr<kdtree::policy::AxisSelectionPolicy<IndicesIterator, SamplesIterator>>
            axis_selection_policy_ptr_;
        // the policy that will be responsible for splitting the selected axis around a pivot sample
        std::shared_ptr<kdtree::policy::SplittingRulePolicy<IndicesIterator, SamplesIterator>>
            splitting_rule_policy_ptr_;
    };

  public:
    KDTree(IndicesIterator indices_range_first,
           IndicesIterator indices_range_last,
           SamplesIterator samples_range_first,
           SamplesIterator samples_range_last,
           std::size_t     n_features,
           const Options&  options = Options());

    KDTree(const KDTree&) = delete;

    std::size_t n_samples() const;

    std::size_t n_features() const;

    SamplesIterator begin() {
        return samples_range_first_;
    }

    SamplesIterator end() {
        return samples_range_last_;
    }

    KDNodeViewPtr root() {
        return root_;
    }

    DataType operator()(std::size_t sample_index) const {
        return samples_range_first_[sample_index];
    }

    // existing samples

    // (1)
    auto nearest_neighbor_around_query_index(std::size_t query_index) const;

    // (2) & (4) & (6)
    template <typename BufferType>
    void buffered_k_nearest_neighbors_around_query_index(std::size_t query_index, BufferType& buffer) const;

    auto k_nearest_neighbors_around_query_index(std::size_t query_index, std::size_t n_neighbors) const;

    // (3)
    std::size_t radius_count_around_query_index(std::size_t query_index, const DataType& radius) const;

    // (4)
    auto radius_search_around_query_index(std::size_t query_index, const DataType& radius) const;

    // (5)
    std::size_t range_count_around_query_index(std::size_t query_index, const HyperRangeType& kd_bounding_box) const;

    // (6)
    auto range_search_around_query_index(std::size_t query_index, const HyperRangeType& kd_bounding_box) const;

    // new samples

    // (7)
    auto nearest_neighbor_around_query_sample(SamplesIterator query_feature_first,
                                              SamplesIterator query_feature_last) const;

    // (7) & (8) & (9) & (10) & (12)
    template <typename BufferType>
    void buffered_k_nearest_neighbors_around_query_sample(SamplesIterator query_feature_first,
                                                          SamplesIterator query_feature_last,
                                                          BufferType&     buffer) const;

    // (7) & (8) & (9) & (10) & (12)
    auto k_nearest_neighbors_around_query_sample(SamplesIterator query_feature_first,
                                                 SamplesIterator query_feature_last,
                                                 std::size_t     n_neighbors) const;

    // (9)
    std::size_t radius_count_around_query_sample(SamplesIterator query_feature_first,
                                                 SamplesIterator query_feature_last,
                                                 const DataType& radius) const;

    // (10)
    auto radius_search_around_query_sample(SamplesIterator query_feature_first,
                                           SamplesIterator query_feature_last,
                                           const DataType& radius) const;

    // (11)
    std::size_t range_count_around_query_sample(SamplesIterator       query_feature_first,
                                                SamplesIterator       query_feature_last,
                                                const HyperRangeType& kd_bounding_box) const;

    // (12)
    auto range_search_around_query_sample(SamplesIterator       query_feature_first,
                                          SamplesIterator       query_feature_last,
                                          const HyperRangeType& kd_bounding_box) const;

    // serialization

    void serialize(const KDNodeViewPtr& kdnode, rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    void serialize(const fs::path& filepath) const;

  private:
    KDNodeViewPtr build(IndicesIterator indices_range_first,
                        IndicesIterator indices_range_last,
                        ssize_t         cut_feature_index,
                        ssize_t         depth,
                        HyperRangeType& kd_bounding_box);

    // existing samples

    // (1) & (2) & (3) & (4) & (5) & (6)
    template <typename BufferType>
    void inner_k_nearest_neighbors_around_query_index(std::size_t   query_index,
                                                      BufferType&   buffer,
                                                      KDNodeViewPtr kdnode = nullptr) const;
    // (1) & (2) & (3) & (4) & (5) & (6)
    template <typename BufferType>
    KDNodeViewPtr recurse_to_closest_leaf_node(std::size_t query_index, BufferType& buffer, KDNodeViewPtr kdnode) const;
    // (1) & (2) & (3) & (4) & (5) & (6)
    template <typename BufferType>
    KDNodeViewPtr get_parent_node_after_sibling_traversal(std::size_t   query_index,
                                                          BufferType&   buffer,
                                                          KDNodeViewPtr kdnode) const;

    // new samples

    // (7) & (8) & (9) & (10) & (11) & (12)
    template <typename BufferType>
    void inner_k_nearest_neighbors_around_query_sample(SamplesIterator query_feature_first,
                                                       SamplesIterator query_feature_last,
                                                       BufferType&     buffer,
                                                       KDNodeViewPtr   kdnode = nullptr) const;
    // (7) & (8) & (9) & (10) & (11) & (12)
    template <typename BufferType>
    KDNodeViewPtr recurse_to_closest_leaf_node(SamplesIterator query_feature_first,
                                               SamplesIterator query_feature_last,
                                               BufferType&     buffer,
                                               KDNodeViewPtr   kdnode) const;
    // (7) & (8) & (9) & (10) & (11) & (12)
    template <typename BufferType>
    KDNodeViewPtr get_parent_node_after_sibling_traversal(SamplesIterator query_feature_first,
                                                          SamplesIterator query_feature_last,
                                                          BufferType&     buffer,
                                                          KDNodeViewPtr   kdnode) const;

    // Options used to configure the indexing structure.
    Options options_;
    // Iterator pointing to the first element of the dataset.
    SamplesIterator samples_range_first_;
    // Iterator pointing to the last element of the dataset.
    SamplesIterator samples_range_last_;
    // The number of features in the dataset, used to represent data as a vectorized 2D array.
    std::size_t n_features_;
    // A hyperrectangle (bounding box) specifying the value bounds of the subset of data represented by the index array
    // from the entire dataset. This hyperrectangle is defined with respect to each dimension.
    HyperRangeType kd_bounding_box_;
    // The root node of the indexing structure.
    KDNodeViewPtr root_;
};

template <typename IndicesIterator, typename SamplesIterator>
KDTree<IndicesIterator, SamplesIterator>::KDTree(IndicesIterator indices_range_first,
                                                 IndicesIterator indices_range_last,
                                                 SamplesIterator samples_range_first,
                                                 SamplesIterator samples_range_last,
                                                 std::size_t     n_features,
                                                 const Options&  options)
  : options_{options}
  , samples_range_first_{samples_range_first}
  , samples_range_last_{samples_range_last}
  , n_features_{n_features}
  , kd_bounding_box_{bbox::make_kd_bounding_box(indices_range_first,
                                                indices_range_last,
                                                samples_range_first_,
                                                samples_range_last_,
                                                n_features_)}
  , root_{build(indices_range_first,
                indices_range_last,
                (*options_.axis_selection_policy_ptr_)(indices_range_first,
                                                       indices_range_last,
                                                       samples_range_first_,
                                                       samples_range_last_,
                                                       n_features_,
                                                       0,
                                                       kd_bounding_box_),
                0,
                kd_bounding_box_)} {}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t KDTree<IndicesIterator, SamplesIterator>::n_samples() const {
    return common::get_n_samples(samples_range_first_, samples_range_last_, n_features_);
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t KDTree<IndicesIterator, SamplesIterator>::n_features() const {
    return n_features_;
}

template <typename IndicesIterator, typename SamplesIterator>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr KDTree<IndicesIterator, SamplesIterator>::build(
    IndicesIterator indices_range_first,
    IndicesIterator indices_range_last,
    ssize_t         cut_feature_index,
    ssize_t         depth,
    HyperRangeType& kd_bounding_box) {
    KDNodeViewPtr kdnode;
    // number of samples in the current node
    const std::size_t n_node_samples = std::distance(indices_range_first, indices_range_last);
    // if the current number of samples is greater than the target bucket size, the node is not leaf
    if (depth < options_.max_depth_ && n_node_samples > options_.bucket_size_) {
        // select the cut_feature_index according to the one with the most spread (min-max values)
        cut_feature_index = (*options_.axis_selection_policy_ptr_)(
            /**/ indices_range_first,
            /**/ indices_range_last,
            /**/ samples_range_first_,
            /**/ samples_range_last_,
            /**/ n_features_,
            /**/ depth,
            /**/ kd_bounding_box);

        auto [cut_index, left_indices_range, cut_indices_range, right_indices_range] =
            (*options_.splitting_rule_policy_ptr_)(
                /**/ indices_range_first,
                /**/ indices_range_last,
                /**/ samples_range_first_,
                /**/ samples_range_last_,
                /**/ n_features_,
                /**/ cut_feature_index);

        kdnode =
            std::make_shared<KDNodeViewType>(cut_indices_range, cut_feature_index, kd_bounding_box[cut_feature_index]);

        const auto cut_feature_value = samples_range_first_[*cut_indices_range.first * n_features_ + cut_feature_index];
        {
            // set the right bound of the left child to the cut value
            kd_bounding_box[cut_feature_index].second = cut_feature_value;

            kdnode->left_ = build(/**/ left_indices_range.first,
                                  /**/ left_indices_range.second,
                                  /**/ cut_feature_index,
                                  /**/ depth + 1,
                                  /**/ kd_bounding_box);
            // provide a parent pointer to the child node for reversed traversal of the kdtree
            kdnode->left_->parent_ = kdnode;

            // reset the right bound of the bounding box to the current kdnode right bound
            kd_bounding_box[cut_feature_index].second = kdnode->kd_bounding_box_.second;
        }
        {
            // set the left bound of the right child to the cut value
            kd_bounding_box[cut_feature_index].first = cut_feature_value;

            kdnode->right_ = build(/**/ right_indices_range.first,
                                   /**/ right_indices_range.second,
                                   /**/ cut_feature_index,
                                   /**/ depth + 1,
                                   /**/ kd_bounding_box);
            // provide a parent pointer to the child node for reversed traversal of the kdtree
            kdnode->right_->parent_ = kdnode;

            // reset the left bound of the bounding box to the current kdnode left bound
            kd_bounding_box[cut_feature_index].first = kdnode->kd_bounding_box_.first;
        }
    } else {
        kdnode = std::make_shared<KDNodeViewType>(std::make_pair(indices_range_first, indices_range_last),
                                                  kd_bounding_box[cut_feature_index]);
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::nearest_neighbor_around_query_index(std::size_t query_index) const {
    auto buffer = knn::buffer::Singleton<IndicesIterator, SamplesIterator>();

    inner_k_nearest_neighbors_around_query_index(query_index, buffer, root_);

    return buffer.closest_neighbor_index_distance_pair();
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename BufferType>
void KDTree<IndicesIterator, SamplesIterator>::buffered_k_nearest_neighbors_around_query_index(
    std::size_t query_index,
    BufferType& buffer) const {
    static_assert(std::is_base_of_v<knn::buffer::Base<IndicesIterator, SamplesIterator>, BufferType> ||
                      std::is_base_of_v<knn::count::Base<IndicesIterator, SamplesIterator>, BufferType>,
                  "BufferType must inherit from knn::buffer::Base<IndicesIterator, SamplesIterator> or "
                  "knn::count::Base<IndicesIterator, SamplesIterator>");

    inner_k_nearest_neighbors_around_query_index(query_index, buffer);
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::k_nearest_neighbors_around_query_index(std::size_t query_index,
                                                                                      std::size_t n_neighbors) const {
    knn::buffer::Unsorted<IndicesIterator, SamplesIterator> buffer(n_neighbors);

    inner_k_nearest_neighbors_around_query_index(query_index, buffer);

    return buffer;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename BufferType>
void KDTree<IndicesIterator, SamplesIterator>::inner_k_nearest_neighbors_around_query_index(
    std::size_t   query_index,
    BufferType&   buffer,
    KDNodeViewPtr kdnode) const {
    // current_node is currently a leaf node (and root in the special case where the entire tree is in a single node)
    auto current_kdnode = recurse_to_closest_leaf_node(
        /**/ query_index,
        /**/ buffer,
        /**/ kdnode == nullptr ? root_ : kdnode);

    // performs a nearest neighbor search one step at a time from the leaf node until the input kdnode is reached if
    // kdnode parameter is a subtree. A search through the entire tree
    while (current_kdnode != kdnode) {
        // performs a nearest neighbor search starting from the specified node then returns its parent if it exists
        // (nullptr otherwise)
        current_kdnode = get_parent_node_after_sibling_traversal(
            /**/ query_index,
            /**/ buffer,
            /**/ current_kdnode);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename BufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::recurse_to_closest_leaf_node(std::size_t   query_index,
                                                                       BufferType&   buffer,
                                                                       KDNodeViewPtr kdnode) const {
    // update the current k neighbors indices and distances. No op if no candidate is closer or if the ranges are empty
    buffer(kdnode->indices_range_.first,
           kdnode->indices_range_.second,
           samples_range_first_,
           samples_range_last_,
           n_features_,
           query_index);

    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode->indices_range_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_range_first_[pivot_index * n_features_ + kdnode->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = samples_range_first_[query_index * n_features_ + kdnode->cut_feature_index_];

        // traverse either the left or right child node depending on where the target sample is located relatively to
        // the cut value
        if (query_split_value < pivot_split_value) {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_index,
                /**/ buffer,
                /**/ kdnode->left_);
        } else {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_index,
                /**/ buffer,
                /**/ kdnode->right_);
        }
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename BufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::get_parent_node_after_sibling_traversal(std::size_t   query_index,
                                                                                  BufferType&   buffer,
                                                                                  KDNodeViewPtr kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();
    // if kdnode has a parent
    if (kdnode_parent) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode_parent->indices_range_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value =
            samples_range_first_[pivot_index * n_features_ + kdnode_parent->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value =
            samples_range_first_[query_index * n_features_ + kdnode_parent->cut_feature_index_];
        // if the axiswise distance is equal to the current furthest nearest neighbor distance, there could be a nearest
        // neighbor to the other side of the hyperrectangle since the values that are equal to the pivot are put to the
        // right
        bool visit_sibling = kdnode->is_left_child()
                                 ? buffer.n_free_slots() || common::abs(pivot_split_value - query_split_value) <=
                                                                buffer.upper_bound(kdnode_parent->cut_feature_index_)
                                 : buffer.n_free_slots() || common::abs(pivot_split_value - query_split_value) <
                                                                buffer.upper_bound(kdnode_parent->cut_feature_index_);
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // get the nearest neighbor from the sibling node
                inner_k_nearest_neighbors_around_query_index(
                    /**/ query_index,
                    /**/ buffer,
                    /**/ sibling_node);
            }
        }
    }
    // returns nullptr if kdnode doesnt have parent (or is root)
    return kdnode_parent;
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t KDTree<IndicesIterator, SamplesIterator>::radius_count_around_query_index(std::size_t     query_index,
                                                                                      const DataType& radius) const {
    auto buffer = knn::count::Radius<IndicesIterator, SamplesIterator>(radius);

    inner_k_nearest_neighbors_around_query_index(query_index, buffer);

    return buffer.count();
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::radius_search_around_query_index(std::size_t     query_index,
                                                                                const DataType& radius) const {
    auto buffer = knn::buffer::Radius<IndicesIterator, SamplesIterator>(radius);

    inner_k_nearest_neighbors_around_query_index(query_index, buffer);

    return buffer;
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t KDTree<IndicesIterator, SamplesIterator>::range_count_around_query_index(
    std::size_t           query_index,
    const HyperRangeType& kd_bounding_box) const {
    const auto translated_kd_bounding_box = bbox::relative_to_absolute_coordinates(
        /**/ samples_range_first_ + query_index * n_features_,
        /**/ samples_range_first_ + query_index * n_features_ + n_features_,
        /**/ kd_bounding_box);

    auto buffer = knn::count::Range<IndicesIterator, SamplesIterator>(translated_kd_bounding_box);

    inner_k_nearest_neighbors_around_query_index(query_index, buffer);

    return buffer.count();
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::range_search_around_query_index(
    std::size_t           query_index,
    const HyperRangeType& kd_bounding_box) const {
    const auto translated_kd_bounding_box = bbox::relative_to_absolute_coordinates(
        /**/ samples_range_first_ + query_index * n_features_,
        /**/ samples_range_first_ + query_index * n_features_ + n_features_,
        /**/ kd_bounding_box);

    auto buffer = knn::buffer::Range<IndicesIterator, SamplesIterator>(translated_kd_bounding_box);

    inner_k_nearest_neighbors_around_query_index(query_index, buffer);

    return buffer;
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::nearest_neighbor_around_query_sample(
    SamplesIterator query_feature_first,
    SamplesIterator query_feature_last) const {
    auto buffer = knn::buffer::Singleton<IndicesIterator, SamplesIterator>();

    inner_k_nearest_neighbors_around_query_sample(query_feature_first, query_feature_last, buffer, root_);

    return buffer.closest_neighbor_index_distance_pair();
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename BufferType>
void KDTree<IndicesIterator, SamplesIterator>::buffered_k_nearest_neighbors_around_query_sample(
    SamplesIterator query_feature_first,
    SamplesIterator query_feature_last,
    BufferType&     buffer) const {
    static_assert(std::is_base_of_v<knn::buffer::Base<IndicesIterator, SamplesIterator>, BufferType> ||
                      std::is_base_of_v<knn::count::Base<IndicesIterator, SamplesIterator>, BufferType>,
                  "BufferType must inherit from knn::buffer::Base<IndicesIterator, SamplesIterator> or "
                  "knn::count::Base<IndicesIterator, SamplesIterator>");

    inner_k_nearest_neighbors_around_query_sample(query_feature_first, query_feature_last, buffer);
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::k_nearest_neighbors_around_query_sample(
    SamplesIterator query_feature_first,
    SamplesIterator query_feature_last,
    std::size_t     n_neighbors) const {
    knn::buffer::Unsorted<IndicesIterator, SamplesIterator> buffer(n_neighbors);

    inner_k_nearest_neighbors_around_query_sample(query_feature_first, query_feature_last, buffer);

    return buffer;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename BufferType>
void KDTree<IndicesIterator, SamplesIterator>::inner_k_nearest_neighbors_around_query_sample(
    SamplesIterator query_feature_first,
    SamplesIterator query_feature_last,
    BufferType&     buffer,
    KDNodeViewPtr   kdnode) const {
    // current_node is currently a leaf node (and root in the special case where the entire tree is in a single node)
    auto current_kdnode = recurse_to_closest_leaf_node(
        /**/ query_feature_first,
        /**/ query_feature_last,
        /**/ buffer,
        /**/ kdnode == nullptr ? root_ : kdnode);

    // performs a nearest neighbor search one step at a time from the leaf node until the input kdnode is reached if
    // kdnode parameter is a subtree. A search through the entire tree
    while (current_kdnode != kdnode) {
        // performs a nearest neighbor search starting from the specified node then returns its parent if it exists
        // (nullptr otherwise)
        current_kdnode = get_parent_node_after_sibling_traversal(
            /**/ query_feature_first,
            /**/ query_feature_last,
            /**/ buffer,
            /**/ current_kdnode);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename BufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::recurse_to_closest_leaf_node(SamplesIterator query_feature_first,
                                                                       SamplesIterator query_feature_last,
                                                                       BufferType&     buffer,
                                                                       KDNodeViewPtr   kdnode) const {
    // update the current k neighbors indices and distances. No op if no candidate is closer or if the ranges are empty
    buffer(kdnode->indices_range_.first,
           kdnode->indices_range_.second,
           samples_range_first_,
           samples_range_last_,
           n_features_,
           query_feature_first,
           query_feature_last);

    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode->indices_range_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_range_first_[pivot_index * n_features_ + kdnode->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = query_feature_first[kdnode->cut_feature_index_];

        // traverse either the left or right child node depending on where the target sample is located relatively to
        // the cut value
        if (query_split_value < pivot_split_value) {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ buffer,
                /**/ kdnode->left_);
        } else {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ buffer,
                /**/ kdnode->right_);
        }
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename BufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::get_parent_node_after_sibling_traversal(SamplesIterator query_feature_first,
                                                                                  SamplesIterator query_feature_last,
                                                                                  BufferType&     buffer,
                                                                                  KDNodeViewPtr   kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();
    // if kdnode has a parent
    if (kdnode_parent) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode_parent->indices_range_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value =
            samples_range_first_[pivot_index * n_features_ + kdnode_parent->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = query_feature_first[kdnode_parent->cut_feature_index_];
        // if the axiswise distance is equal to the current furthest nearest neighbor distance, there could be a nearest
        // neighbor to the other side of the hyperrectangle since the values that are equal to the pivot are put to the
        // right
        bool visit_sibling = kdnode->is_left_child()
                                 ? buffer.n_free_slots() || common::abs(pivot_split_value - query_split_value) <=
                                                                buffer.upper_bound(kdnode_parent->cut_feature_index_)
                                 : buffer.n_free_slots() || common::abs(pivot_split_value - query_split_value) <
                                                                buffer.upper_bound(kdnode_parent->cut_feature_index_);
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // get the nearest neighbor from the sibling node
                inner_k_nearest_neighbors_around_query_sample(
                    /**/ query_feature_first,
                    /**/ query_feature_last,
                    /**/ buffer,
                    /**/ sibling_node);
            }
        }
    }
    // returns nullptr if kdnode doesnt have parent (or is root)
    return kdnode_parent;
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t KDTree<IndicesIterator, SamplesIterator>::radius_count_around_query_sample(
    SamplesIterator query_feature_first,
    SamplesIterator query_feature_last,
    const DataType& radius) const {
    auto buffer = knn::count::Radius<IndicesIterator, SamplesIterator>(radius);

    inner_k_nearest_neighbors_around_query_sample(query_feature_first, query_feature_last, buffer);

    return buffer.count();
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::radius_search_around_query_sample(SamplesIterator query_feature_first,
                                                                                 SamplesIterator query_feature_last,
                                                                                 const DataType& radius) const {
    auto buffer = knn::buffer::Radius<IndicesIterator, SamplesIterator>(radius);

    inner_k_nearest_neighbors_around_query_sample(query_feature_first, query_feature_last, buffer);

    return buffer;
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t KDTree<IndicesIterator, SamplesIterator>::range_count_around_query_sample(
    SamplesIterator       query_feature_first,
    SamplesIterator       query_feature_last,
    const HyperRangeType& kd_bounding_box) const {
    const auto translated_kd_bounding_box =
        bbox::relative_to_absolute_coordinates(query_feature_first, query_feature_last, kd_bounding_box);

    auto buffer = knn::count::Range<IndicesIterator, SamplesIterator>(translated_kd_bounding_box);

    inner_k_nearest_neighbors_around_query_sample(query_feature_first, query_feature_last, buffer);

    return buffer.count();
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::range_search_around_query_sample(
    SamplesIterator       query_feature_first,
    SamplesIterator       query_feature_last,
    const HyperRangeType& kd_bounding_box) const {
    const auto translated_kd_bounding_box =
        bbox::relative_to_absolute_coordinates(query_feature_first, query_feature_last, kd_bounding_box);

    auto buffer = knn::buffer::Range<IndicesIterator, SamplesIterator>(translated_kd_bounding_box);

    inner_k_nearest_neighbors_around_query_sample(query_feature_first, query_feature_last, buffer);

    return buffer;
}

template <typename IndicesIterator, typename SamplesIterator>
void KDTree<IndicesIterator, SamplesIterator>::serialize(const KDNodeViewPtr&                        kdnode,
                                                         rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
    writer.StartObject();
    {
        writer.String("axis");
        writer.Int64(kdnode->cut_feature_index_);

        writer.String("points");
        kdnode->serialize(writer, samples_range_first_, samples_range_last_, n_features_);

        // continue the recursion if the current node is not leaf
        if (!kdnode->is_leaf()) {
            {
                writer.String("left");
                serialize(kdnode->left_, writer);
            }
            {
                writer.String("right");
                serialize(kdnode->right_, writer);
            }
        }
    }
    writer.EndObject();
}

template <typename IndicesIterator, typename SamplesIterator>
void KDTree<IndicesIterator, SamplesIterator>::serialize(const fs::path& filepath) const {
    static_assert(std::is_floating_point_v<DataType> || std::is_integral_v<DataType>,
                  "Unsupported type during kdtree serialization");

    rapidjson::Document document;

    rapidjson::StringBuffer buffer;

    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

    writer.StartObject();
    {
        writer.String("n_samples");
        writer.Int64(common::get_n_samples(samples_range_first_, samples_range_last_, n_features_));

        writer.String("n_features");
        writer.Int64(n_features_);

        options_.serialize(writer);

        writer.String("bounding_box");
        writer.StartArray();
        for (std::size_t feature_index = 0; feature_index < n_features_; ++feature_index) {
            writer.StartArray();

            if constexpr (std::is_integral_v<DataType>) {
                writer.Int64(kd_bounding_box_[feature_index].first);
                writer.Int64(kd_bounding_box_[feature_index].second);

            } else if constexpr (std::is_floating_point_v<DataType>) {
                writer.Double(kd_bounding_box_[feature_index].first);
                writer.Double(kd_bounding_box_[feature_index].second);
            }
            writer.EndArray();
        }
        writer.EndArray();

        writer.String("root");
        serialize(root_, writer);
    }
    writer.EndObject();

    document.Parse(buffer.GetString());

    std::ofstream output_file(filepath);

    rapidjson::OStreamWrapper output_stream_wrapper(output_file);

    rapidjson::Writer<rapidjson::OStreamWrapper> filewriter(output_stream_wrapper);

    document.Accept(filewriter);

    output_file.close();
}

}  // namespace ffcl::datastruct