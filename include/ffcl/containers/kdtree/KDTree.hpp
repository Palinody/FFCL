#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDNodeView.hpp"
#include "ffcl/containers/kdtree/policy/AxisSelectionPolicy.hpp"
#include "ffcl/containers/kdtree/policy/SplittingRulePolicy.hpp"

#include "ffcl/math/heuristics/NearestNeighbor.hpp"
#include "ffcl/math/random/Distributions.hpp"

#include <sys/types.h>  // ssize_t
#include <algorithm>
#include <array>
#include <memory>

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/stream.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include <filesystem>
#include <fstream>

namespace ffcl::containers {

namespace fs = std::filesystem;

template <typename SamplesIterator>
class KDTree {
  public:
    using DataType      = typename SamplesIterator::value_type;
    using KDNodeViewPtr = std::shared_ptr<KDNodeView<SamplesIterator>>;

    struct Options {
        Options()
          : bucket_size_{40}
          , max_depth_{common::utils::infinity<ssize_t>()}
          , axis_selection_policy_ptr_{std::make_unique<kdtree::policy::HighestVarianceBuild<SamplesIterator>>()}
          , splitting_rule_policy_ptr_{std::make_unique<kdtree::policy::QuickselectMedianRange<SamplesIterator>>()} {}

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
            static_assert(
                std::is_base_of<kdtree::policy::AxisSelectionPolicy<SamplesIterator>, AxisSelectionPolicy>::value,
                "The provided axis selection policy must be derived from "
                "kdtree::policy::AxisSelectionPolicy<SamplesIterator>");

            axis_selection_policy_ptr_ = std::make_unique<AxisSelectionPolicy>(axis_selection_policy);
            return *this;
        }

        template <typename SplittingRulePolicy>
        Options& splitting_rule_policy(const SplittingRulePolicy& splitting_rule_policy) {
            static_assert(
                std::is_base_of<kdtree::policy::SplittingRulePolicy<SamplesIterator>, SplittingRulePolicy>::value,
                "The provided splitting rule policy must be derived from "
                "kdtree::policy::SplittingRulePolicy<SamplesIterator>");

            splitting_rule_policy_ptr_ = std::make_unique<SplittingRulePolicy>(splitting_rule_policy);
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
        std::shared_ptr<kdtree::policy::AxisSelectionPolicy<SamplesIterator>> axis_selection_policy_ptr_;
        // the policy that will be responsible for splitting the selected axis around a pivot sample
        std::shared_ptr<kdtree::policy::SplittingRulePolicy<SamplesIterator>> splitting_rule_policy_ptr_;
    };

  public:
    KDTree(SamplesIterator samples_first,
           SamplesIterator samples_last,
           std::size_t     n_features,
           const Options&  options = Options());

    KDTree(const KDTree&) = delete;

    // existing samples

    auto nearest_neighbor_around_query_index(std::size_t query_index) const;

    auto k_nearest_neighbors_around_query_index(std::size_t query_index, std::size_t n_neighbors = 1) const;

    auto radius_count_around_query_index(std::size_t query_index, DataType radius) const;  // not implemented

    std::vector<std::size_t> radius_search_around_query_index(std::size_t query_index,
                                                              DataType    radius) const;  // not implemented

    // new samples

    auto nearest_neighbor_around_query_sample(SamplesIterator query_feature_first,
                                              SamplesIterator query_feature_last) const;  // not implemented

    auto k_nearest_neighbors_around_query_sample(SamplesIterator query_feature_first,
                                                 SamplesIterator query_feature_last,
                                                 std::size_t     n_neighbors = 1) const;  // not implemented

    auto radius_count_around_query_sample(SamplesIterator query_feature_first,
                                          SamplesIterator query_feature_last,
                                          DataType        radius) const;  // not implemented

    // serialization

    void serialize(const KDNodeViewPtr& kdnode, rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    void serialize(const fs::path& filepath) const;

  private:
    KDNodeViewPtr build(SamplesIterator                     samples_first,
                        SamplesIterator                     samples_last,
                        ssize_t                             cut_feature_index,
                        ssize_t                             depth,
                        BoundingBoxKDType<SamplesIterator>& kd_bounding_box);

    void nearest_neighbor_around_query_index(std::size_t   query_index,
                                             ssize_t&      current_nearest_neighbor_index,
                                             DataType&     current_nearest_neighbor_distance,
                                             KDNodeViewPtr kdnode = nullptr) const;

    void k_nearest_neighbors_around_query_index(std::size_t                              query_index,
                                                NearestNeighborsBuffer<SamplesIterator>& nearest_neighbors_buffer,
                                                KDNodeViewPtr                            kdnode = nullptr) const;

    KDNodeViewPtr recurse_to_closest_leaf_node(std::size_t   query_index,
                                               ssize_t&      current_nearest_neighbor_index,
                                               DataType&     current_nearest_neighbor_distance,
                                               KDNodeViewPtr kdnode) const;

    KDNodeViewPtr recurse_to_closest_leaf_node(std::size_t                              query_index,
                                               NearestNeighborsBuffer<SamplesIterator>& nearest_neighbors_buffer,
                                               KDNodeViewPtr                            kdnode) const;

    KDNodeViewPtr get_parent_node_after_sibling_traversal(std::size_t   query_index,
                                                          ssize_t&      current_nearest_neighbor_index,
                                                          DataType&     current_nearest_neighbor_distance,
                                                          KDNodeViewPtr kdnode) const;

    KDNodeViewPtr get_parent_node_after_sibling_traversal(
        std::size_t                              query_index,
        NearestNeighborsBuffer<SamplesIterator>& nearest_neighbors_buffer,
        KDNodeViewPtr                            kdnode) const;

    SamplesIterator samples_first_;

    SamplesIterator samples_last_;

    std::size_t n_features_;

    // bounding box hyper rectangle (w.r.t. each dimension)
    BoundingBoxKDType<SamplesIterator> kd_bounding_box_;

    Options options_;

    KDNodeViewPtr root_;
};

template <typename SamplesIterator>
KDTree<SamplesIterator>::KDTree(SamplesIterator samples_first,
                                SamplesIterator samples_last,
                                std::size_t     n_features,
                                const Options&  options)
  : samples_first_{samples_first}
  , samples_last_{samples_last}
  , n_features_{n_features}
  , kd_bounding_box_{kdtree::algorithms::make_kd_bounding_box(samples_first_, samples_last_, n_features_)}
  , options_{options}
  , root_{build(samples_first_,
                samples_last_,
                (*options_.axis_selection_policy_ptr_)(samples_first_, samples_last_, n_features_, 0, kd_bounding_box_),
                0,
                kd_bounding_box_)} {}

template <typename SamplesIterator>
typename KDTree<SamplesIterator>::KDNodeViewPtr KDTree<SamplesIterator>::build(
    SamplesIterator                     samples_first,
    SamplesIterator                     samples_last,
    ssize_t                             cut_feature_index,
    ssize_t                             depth,
    BoundingBoxKDType<SamplesIterator>& kd_bounding_box) {
    KDNodeViewPtr kdnode;

    // number of samples in the current node
    const std::size_t n_node_samples = common::utils::get_n_samples(samples_first, samples_last, n_features_);

    // the current kdnode is not leaf
    if (n_node_samples > options_.bucket_size_ && depth < options_.max_depth_) {
        // select the cut_feature_index according to the one with the most spread (min-max values)
        cut_feature_index =
            (*options_.axis_selection_policy_ptr_)(samples_first, samples_last, n_features_, depth, kd_bounding_box);

        auto [cut_index, left_range, cut_range, right_range] = (*options_.splitting_rule_policy_ptr_)(
            /**/ samples_first,
            /**/ samples_last,
            /**/ n_features_,
            /**/ cut_feature_index);

        kdnode =
            std::make_shared<KDNodeView<SamplesIterator>>(cut_range, n_features_, cut_feature_index, kd_bounding_box);

        const auto cut_value = cut_range.first[cut_feature_index];
        {
            // set the right bound of the left child to the cut value
            kd_bounding_box[cut_feature_index].second = cut_value;

            kdnode->left_ = build(
                /**/ left_range.first,
                /**/ left_range.second,
                /**/ cut_feature_index,
                /**/ depth + 1,
                /**/ kd_bounding_box);
            // provide a parent pointer to the child node for reversed traversal of the kdtree
            kdnode->left_->parent_ = kdnode;

            // reset the right bound of the bounding box to the current kdnode right bound
            kd_bounding_box[cut_feature_index].second = kdnode->kd_bounding_box_[cut_feature_index].second;
        }
        {
            // set the left bound of the right child to the cut value
            kd_bounding_box[cut_feature_index].first = cut_value;

            kdnode->right_ = build(/**/ right_range.first,
                                   /**/ right_range.second,
                                   /**/ cut_feature_index,
                                   /**/ depth + 1,
                                   /**/ kd_bounding_box);
            // provide a parent pointer to the child node for reversed traversal of the kdtree
            kdnode->right_->parent_ = kdnode;

            // reset the left bound of the bounding box to the current kdnode left bound
            kd_bounding_box[cut_feature_index].first = kdnode->kd_bounding_box_[cut_feature_index].first;
        }
    } else {
        kdnode = std::make_shared<KDNodeView<SamplesIterator>>(
            std::make_pair(samples_first, samples_last), n_features_, kd_bounding_box);
    }
    return kdnode;
}

template <typename SamplesIterator>
auto KDTree<SamplesIterator>::nearest_neighbor_around_query_index(std::size_t query_index) const {
    ssize_t current_nearest_neighbor_index    = -1;
    auto    current_nearest_neighbor_distance = common::utils::infinity<DataType>();

    nearest_neighbor_around_query_index(
        query_index, current_nearest_neighbor_index, current_nearest_neighbor_distance, root_);

    return std::make_pair(current_nearest_neighbor_index, current_nearest_neighbor_distance);
}

template <typename SamplesIterator>
void KDTree<SamplesIterator>::nearest_neighbor_around_query_index(std::size_t   query_index,
                                                                  ssize_t&      current_nearest_neighbor_index,
                                                                  DataType&     current_nearest_neighbor_distance,
                                                                  KDNodeViewPtr kdnode) const {
    // current_node is currently a leaf node (and root in the special case where the entire tree is in a single node)
    auto current_kdnode = recurse_to_closest_leaf_node(query_index,
                                                       current_nearest_neighbor_index,
                                                       current_nearest_neighbor_distance,
                                                       kdnode == nullptr ? root_ : kdnode);

    // performs a nearest neighbor search one step at a time from the leaf node until the input kdnode is reached if
    // kdnode parameter is a subtree. A search through the entire tree
    while (current_kdnode != kdnode) {
        // performs a nearest neighbor search starting from the specified node then returns its parent if it exists
        // (nullptr otherwise)
        current_kdnode = get_parent_node_after_sibling_traversal(
            /**/ query_index,
            /**/ current_nearest_neighbor_index,
            /**/ current_nearest_neighbor_distance,
            /**/ current_kdnode);
    }
}

template <typename SamplesIterator>
auto KDTree<SamplesIterator>::k_nearest_neighbors_around_query_index(std::size_t query_index,
                                                                     std::size_t n_neighbors) const {
    NearestNeighborsBuffer<SamplesIterator> nearest_neighbors_buffer(n_neighbors);

    k_nearest_neighbors_around_query_index(query_index, nearest_neighbors_buffer);

    return nearest_neighbors_buffer.move_data_to_indices_distances_pair();
}

template <typename SamplesIterator>
void KDTree<SamplesIterator>::k_nearest_neighbors_around_query_index(
    std::size_t                              query_index,
    NearestNeighborsBuffer<SamplesIterator>& nearest_neighbors_buffer,
    KDNodeViewPtr                            kdnode) const {
    // current_node is currently a leaf node (and root in the special case where the entire tree is in a single node)
    auto current_kdnode = recurse_to_closest_leaf_node(
        /**/ query_index,
        /**/ nearest_neighbors_buffer,
        /**/ kdnode == nullptr ? root_ : kdnode);

    // performs a nearest neighbor search one step at a time from the leaf node until the input kdnode is reached if
    // kdnode parameter is a subtree. A search through the entire tree
    while (current_kdnode != kdnode) {
        // performs a nearest neighbor search starting from the specified node then returns its parent if it exists
        // (nullptr otherwise)
        current_kdnode = get_parent_node_after_sibling_traversal(
            /**/ query_index,
            /**/ nearest_neighbors_buffer,
            /**/ current_kdnode);
    }
}

template <typename SamplesIterator>
typename KDTree<SamplesIterator>::KDNodeViewPtr KDTree<SamplesIterator>::recurse_to_closest_leaf_node(
    std::size_t   query_index,
    ssize_t&      current_nearest_neighbor_index,
    DataType&     current_nearest_neighbor_distance,
    KDNodeViewPtr kdnode) const {
    // update the current neighbor index and distance. No op if no candidate is closer or if the ranges are empty
    math::heuristics::nearest_neighbor_range(kdnode->samples_iterator_pair_.first,
                                             kdnode->samples_iterator_pair_.second,
                                             samples_first_,
                                             samples_last_,
                                             kdnode->n_features_,
                                             query_index,
                                             current_nearest_neighbor_index,
                                             current_nearest_neighbor_distance);

    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the split value according to the current split dimension
        const auto pivot_split_value = kdnode->samples_iterator_pair_.first[kdnode->cut_feature_index_];

        // get the value of the query according to the split dimension
        const auto query_split_value = samples_first_[query_index * n_features_ + kdnode->cut_feature_index_];

        // traverse either the left or right child node depending on where the target sample is located relatively to
        // the cut value
        if (query_split_value < pivot_split_value) {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_index,
                /**/ current_nearest_neighbor_index,
                /**/ current_nearest_neighbor_distance,
                /**/ kdnode->left_);
        } else {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_index,
                /**/ current_nearest_neighbor_index,
                /**/ current_nearest_neighbor_distance,
                /**/ kdnode->right_);
        }
    }
    return kdnode;
}

template <typename SamplesIterator>
typename KDTree<SamplesIterator>::KDNodeViewPtr KDTree<SamplesIterator>::recurse_to_closest_leaf_node(
    std::size_t                              query_index,
    NearestNeighborsBuffer<SamplesIterator>& nearest_neighbors_buffer,
    KDNodeViewPtr                            kdnode) const {
    // update the current neighbor index and distance. No op if no candidate is closer or if the ranges are empty
    math::heuristics::k_nearest_neighbors_range(kdnode->samples_iterator_pair_.first,
                                                kdnode->samples_iterator_pair_.second,
                                                samples_first_,
                                                samples_last_,
                                                kdnode->n_features_,
                                                query_index,
                                                nearest_neighbors_buffer);

    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the split value according to the current split dimension
        const auto pivot_split_value = kdnode->samples_iterator_pair_.first[kdnode->cut_feature_index_];

        // get the value of the query according to the split dimension
        const auto query_split_value = samples_first_[query_index * n_features_ + kdnode->cut_feature_index_];

        // traverse either the left or right child node depending on where the target sample is located relatively to
        // the cut value
        if (query_split_value < pivot_split_value) {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_index,
                /**/ nearest_neighbors_buffer,
                /**/ kdnode->left_);
        } else {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_index,
                /**/ nearest_neighbors_buffer,
                /**/ kdnode->right_);
        }
    }
    return kdnode;
}

template <typename SamplesIterator>
typename KDTree<SamplesIterator>::KDNodeViewPtr KDTree<SamplesIterator>::get_parent_node_after_sibling_traversal(
    std::size_t   query_index,
    ssize_t&      current_nearest_neighbor_index,
    DataType&     current_nearest_neighbor_distance,
    KDNodeViewPtr kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();

    // if kdnode has a parent
    if (kdnode_parent) {
        // get the split value according to the current split dimension
        const auto pivot_split_value = kdnode_parent->samples_iterator_pair_.first[kdnode_parent->cut_feature_index_];

        // get the value of the query according to the split dimension
        const auto query_split_value = samples_first_[query_index * n_features_ + kdnode_parent->cut_feature_index_];

        // if the axiswise distance is equal to the current nearest neighbor distance, there could be a nearest neighbor
        // to the other side of the hyperrectangle since the values that are equal to the pivot are put to the right
        bool visit_sibling =
            kdnode->is_left_child()
                ? common::utils::abs(pivot_split_value - query_split_value) < current_nearest_neighbor_distance
                : common::utils::abs(pivot_split_value - query_split_value) <= current_nearest_neighbor_distance;
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // update the nearest neighbor from the sibling node
                nearest_neighbor_around_query_index(
                    /**/ query_index,
                    /**/ current_nearest_neighbor_index,
                    /**/ current_nearest_neighbor_distance,
                    /**/ sibling_node);
            }
        }
    }
    // returns nullptr if kdnode doesnt have parent (or is root)
    return kdnode_parent;
}

template <typename SamplesIterator>
typename KDTree<SamplesIterator>::KDNodeViewPtr KDTree<SamplesIterator>::get_parent_node_after_sibling_traversal(
    std::size_t                              query_index,
    NearestNeighborsBuffer<SamplesIterator>& nearest_neighbors_buffer,
    KDNodeViewPtr                            kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();

    // if kdnode has a parent
    if (kdnode_parent) {
        // get the split value according to the current split dimension
        const auto pivot_split_value = kdnode_parent->samples_iterator_pair_.first[kdnode_parent->cut_feature_index_];

        // get the value of the query according to the split dimension
        const auto query_split_value = samples_first_[query_index * n_features_ + kdnode_parent->cut_feature_index_];

        // if the axiswise distance is equal to the current nearest neighbor distance, there could be a nearest neighbor
        // to the other side of the hyperrectangle since the values that are equal to the pivot are put to the right
        bool visit_sibling = kdnode->is_left_child()
                                 ? common::utils::abs(pivot_split_value - query_split_value) <
                                       nearest_neighbors_buffer.furthest_k_nearest_neighbor_distance()
                                 : common::utils::abs(pivot_split_value - query_split_value) <=
                                       nearest_neighbors_buffer.furthest_k_nearest_neighbor_distance();
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // update the nearest neighbor from the sibling node
                k_nearest_neighbors_around_query_index(
                    /**/ query_index,
                    /**/ nearest_neighbors_buffer,
                    /**/ sibling_node);
            }
        }
    }
    // returns nullptr if kdnode doesnt have parent (or is root)
    return kdnode_parent;
}

template <typename SamplesIterator>
void KDTree<SamplesIterator>::serialize(const KDNodeViewPtr&                        kdnode,
                                        rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
    writer.StartObject();
    {
        writer.String("axis");
        writer.Int64(kdnode->cut_feature_index_);

        writer.String("points");
        kdnode->serialize(writer);

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

template <typename SamplesIterator>
void KDTree<SamplesIterator>::serialize(const fs::path& filepath) const {
    static_assert(std::is_floating_point_v<DataType> || std::is_integral_v<DataType>,
                  "Unsupported type during kdtree serialization");

    rapidjson::Document                        document;
    rapidjson::StringBuffer                    buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

    writer.StartObject();
    {
        writer.String("n_samples");
        writer.Int64(common::utils::get_n_samples(samples_first_, samples_last_, n_features_));

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

    std::ofstream                                output_file(filepath);
    rapidjson::OStreamWrapper                    output_stream_wrapper(output_file);
    rapidjson::Writer<rapidjson::OStreamWrapper> filewriter(output_stream_wrapper);
    document.Accept(filewriter);
    output_file.close();
}

}  // namespace ffcl::containers