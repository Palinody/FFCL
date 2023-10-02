#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/kdtree/KDNodeView.hpp"
#include "ffcl/datastruct/kdtree/policy/AxisSelectionPolicy.hpp"
#include "ffcl/datastruct/kdtree/policy/SplittingRulePolicy.hpp"

#include "ffcl/math/random/Distributions.hpp"

#include "ffcl/knn/NearestNeighbor.hpp"

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
    using DataType       = typename SamplesIterator::value_type;
    using KDNodeViewType = typename KDNodeView<IndicesIterator, SamplesIterator>::KDNodeViewType;
    using KDNodeViewPtr  = typename KDNodeView<IndicesIterator, SamplesIterator>::KDNodeViewPtr;
    using HyperRangeType = ffcl::bbox::HyperRangeType<SamplesIterator>;

    struct Options {
        Options()
          : bucket_size_{40}
          , max_depth_{common::utils::infinity<ssize_t>()}
          , axis_selection_policy_ptr_{std::make_unique<
                kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>>()}
          , splitting_rule_policy_ptr_{
                std::make_unique<kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>>()} {}

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

            axis_selection_policy_ptr_ = std::make_unique<AxisSelectionPolicy>(axis_selection_policy);
            return *this;
        }

        template <typename SplittingRulePolicy>
        Options& splitting_rule_policy(const SplittingRulePolicy& splitting_rule_policy) {
            static_assert(std::is_base_of<kdtree::policy::SplittingRulePolicy<IndicesIterator, SamplesIterator>,
                                          SplittingRulePolicy>::value,
                          "The provided splitting rule policy must be derived from "
                          "kdtree::policy::SplittingRulePolicy<IndicesIterator, SamplesIterator>");

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
        std::shared_ptr<kdtree::policy::AxisSelectionPolicy<IndicesIterator, SamplesIterator>>
            axis_selection_policy_ptr_;
        // the policy that will be responsible for splitting the selected axis around a pivot sample
        std::shared_ptr<kdtree::policy::SplittingRulePolicy<IndicesIterator, SamplesIterator>>
            splitting_rule_policy_ptr_;
    };

  public:
    KDTree(IndicesIterator index_first,
           IndicesIterator index_last,
           SamplesIterator samples_first,
           SamplesIterator samples_last,
           std::size_t     n_features,
           const Options&  options = Options());

    KDTree(const KDTree&) = delete;

    std::size_t n_samples() const;

    std::size_t n_features() const;

    // existing samples

    auto k_mutual_reachability_distance(std::size_t query_index_1,
                                        std::size_t query_index_2,
                                        std::size_t k_nearest_neighbors) const;

    auto buffered_k_mutual_reachability_distance(std::size_t                        query_index_1,
                                                 std::size_t                        query_index_2,
                                                 const std::shared_ptr<DataType[]>& core_distances) const;
    // (1)
    auto nearest_neighbor_around_query_index(std::size_t query_index) const;

    // (2)
    template <typename NearestNeighborsBufferType>
    void buffered_k_nearest_neighbors_around_query_index(std::size_t                 query_index,
                                                         NearestNeighborsBufferType& nearest_neighbors_buffer) const;

    auto k_nearest_neighbors_around_query_index(std::size_t query_index, std::size_t n_neighbors) const;

    // (3)
    std::size_t radius_count_around_query_index(std::size_t query_index, const DataType& radius) const;

    // (4)
    template <typename NearestNeighborsBufferType>
    void buffered_radius_search_around_query_index(std::size_t                 query_index,
                                                   const DataType&             radius,
                                                   NearestNeighborsBufferType& nearest_neighbors_buffer) const;

    auto radius_search_around_query_index(std::size_t query_index, const DataType& radius) const;

    // (5)
    std::size_t range_count_around_query_index(std::size_t query_index, const HyperRangeType& kd_bounding_box) const;

    // (6)
    template <typename NearestNeighborsBufferType>
    void buffered_range_search_around_query_index(std::size_t                 query_index,
                                                  const HyperRangeType&       kd_bounding_box,
                                                  NearestNeighborsBufferType& nearest_neighbors_buffer) const;

    auto range_search_around_query_index(std::size_t query_index, const HyperRangeType& kd_bounding_box) const;

    // new samples

    // (7)
    auto nearest_neighbor_around_query_sample(SamplesIterator query_feature_first,
                                              SamplesIterator query_feature_last) const;

    // (8)
    template <typename NearestNeighborsBufferType>
    void buffered_k_nearest_neighbors_around_query_sample(SamplesIterator             query_feature_first,
                                                          SamplesIterator             query_feature_last,
                                                          NearestNeighborsBufferType& nearest_neighbors_buffer) const;

    auto k_nearest_neighbors_around_query_sample(SamplesIterator query_feature_first,
                                                 SamplesIterator query_feature_last,
                                                 std::size_t     n_neighbors) const;

    // (9)
    std::size_t radius_count_around_query_sample(SamplesIterator query_feature_first,
                                                 SamplesIterator query_feature_last,
                                                 const DataType& radius) const;

    // (10)
    template <typename NearestNeighborsBufferType>
    void buffered_radius_search_around_query_sample(SamplesIterator             query_feature_first,
                                                    SamplesIterator             query_feature_last,
                                                    const DataType&             radius,
                                                    NearestNeighborsBufferType& nearest_neighbors_buffer) const;

    auto radius_search_around_query_sample(SamplesIterator query_feature_first,
                                           SamplesIterator query_feature_last,
                                           const DataType& radius) const;

    // (11)
    std::size_t range_count_around_query_sample(SamplesIterator       query_feature_first,
                                                SamplesIterator       query_feature_last,
                                                const HyperRangeType& kd_bounding_box) const;

    // (12)
    template <typename NearestNeighborsBufferType>
    void buffered_range_search_around_query_sample(SamplesIterator             query_feature_first,
                                                   SamplesIterator             query_feature_last,
                                                   const HyperRangeType&       kd_bounding_box,
                                                   NearestNeighborsBufferType& nearest_neighbors_buffer) const;

    auto range_search_around_query_sample(SamplesIterator       query_feature_first,
                                          SamplesIterator       query_feature_last,
                                          const HyperRangeType& kd_bounding_box) const;

    // serialization

    void serialize(const KDNodeViewPtr& kdnode, rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    void serialize(const fs::path& filepath) const;

  private:
    KDNodeViewPtr build(IndicesIterator index_first,
                        IndicesIterator index_last,
                        ssize_t         cut_feature_index,
                        ssize_t         depth,
                        HyperRangeType& kd_bounding_box);

    // existing samples

    // (1)
    void inner_nearest_neighbor_around_query_index(std::size_t   query_index,
                                                   ssize_t&      current_nearest_neighbor_index,
                                                   DataType&     current_nearest_neighbor_distance,
                                                   KDNodeViewPtr kdnode = nullptr) const;
    // (1)
    KDNodeViewPtr recurse_to_closest_leaf_node(std::size_t   query_index,
                                               ssize_t&      current_nearest_neighbor_index,
                                               DataType&     current_nearest_neighbor_distance,
                                               KDNodeViewPtr kdnode) const;
    // (1)
    KDNodeViewPtr get_parent_node_after_sibling_traversal(std::size_t   query_index,
                                                          ssize_t&      current_nearest_neighbor_index,
                                                          DataType&     current_nearest_neighbor_distance,
                                                          KDNodeViewPtr kdnode) const;
    // (2)
    template <typename NearestNeighborsBufferType>
    void inner_k_nearest_neighbors_around_query_index(std::size_t                 query_index,
                                                      NearestNeighborsBufferType& nearest_neighbors_buffer,
                                                      KDNodeViewPtr               kdnode = nullptr) const;
    // (2)
    template <typename NearestNeighborsBufferType>
    KDNodeViewPtr recurse_to_closest_leaf_node(std::size_t                 query_index,
                                               NearestNeighborsBufferType& nearest_neighbors_buffer,
                                               KDNodeViewPtr               kdnode) const;
    // (2)
    template <typename NearestNeighborsBufferType>
    KDNodeViewPtr get_parent_node_after_sibling_traversal(std::size_t                 query_index,
                                                          NearestNeighborsBufferType& nearest_neighbors_buffer,
                                                          KDNodeViewPtr               kdnode) const;
    // (3)
    void inner_radius_count_around_query_index(std::size_t     query_index,
                                               const DataType& radius,
                                               std::size_t&    neighbors_count,
                                               KDNodeViewPtr   kdnode = nullptr) const;
    // (3)
    KDNodeViewPtr recurse_to_closest_leaf_node(std::size_t     query_index,
                                               const DataType& radius,
                                               std::size_t&    neighbors_count,
                                               KDNodeViewPtr   kdnode) const;
    // (3)
    KDNodeViewPtr get_parent_node_after_sibling_traversal(std::size_t     query_index,
                                                          const DataType& radius,
                                                          std::size_t&    neighbors_count,
                                                          KDNodeViewPtr   kdnode) const;
    // (4)
    template <typename NearestNeighborsBufferType>
    void inner_radius_search_around_query_index(std::size_t                 query_index,
                                                const DataType&             radius,
                                                NearestNeighborsBufferType& nearest_neighbors_buffer,
                                                KDNodeViewPtr               kdnode = nullptr) const;
    // (4)
    template <typename NearestNeighborsBufferType>
    KDNodeViewPtr recurse_to_closest_leaf_node(std::size_t                 query_index,
                                               const DataType&             radius,
                                               NearestNeighborsBufferType& nearest_neighbors_buffer,
                                               KDNodeViewPtr               kdnode) const;
    // (4)
    template <typename NearestNeighborsBufferType>
    KDNodeViewPtr get_parent_node_after_sibling_traversal(std::size_t                 query_index,
                                                          const DataType&             radius,
                                                          NearestNeighborsBufferType& nearest_neighbors_buffer,
                                                          KDNodeViewPtr               kdnode) const;

    // (5)
    void inner_range_count_around_query_index(std::size_t           query_index,
                                              const HyperRangeType& kd_bounding_box,
                                              std::size_t&          neighbors_count,
                                              KDNodeViewPtr         kdnode = nullptr) const;
    // (5)
    KDNodeViewPtr recurse_to_closest_leaf_node(std::size_t           query_index,
                                               const HyperRangeType& kd_bounding_box,
                                               std::size_t&          neighbors_count,
                                               KDNodeViewPtr         kdnode) const;
    // (5)
    KDNodeViewPtr get_parent_node_after_sibling_traversal(std::size_t           query_index,
                                                          const HyperRangeType& kd_bounding_box,
                                                          std::size_t&          neighbors_count,
                                                          KDNodeViewPtr         kdnode) const;

    // (6)
    template <typename NearestNeighborsBufferType>
    void inner_range_search_around_query_index(std::size_t                 query_index,
                                               const HyperRangeType&       kd_bounding_box,
                                               NearestNeighborsBufferType& nearest_neighbors_buffer,
                                               KDNodeViewPtr               kdnode = nullptr) const;
    // (6)
    template <typename NearestNeighborsBufferType>
    KDNodeViewPtr recurse_to_closest_leaf_node(std::size_t                 query_index,
                                               const HyperRangeType&       kd_bounding_box,
                                               NearestNeighborsBufferType& nearest_neighbors_buffer,
                                               KDNodeViewPtr               kdnode) const;
    // (6)
    template <typename NearestNeighborsBufferType>
    KDNodeViewPtr get_parent_node_after_sibling_traversal(std::size_t                 query_index,
                                                          const HyperRangeType&       kd_bounding_box,
                                                          NearestNeighborsBufferType& nearest_neighbors_buffer,
                                                          KDNodeViewPtr               kdnode) const;

    // new samples

    // (7)
    void inner_nearest_neighbor_around_query_sample(SamplesIterator query_feature_first,
                                                    SamplesIterator query_feature_last,
                                                    ssize_t&        current_nearest_neighbor_index,
                                                    DataType&       current_nearest_neighbor_distance,
                                                    KDNodeViewPtr   kdnode = nullptr) const;
    // (7)
    KDNodeViewPtr recurse_to_closest_leaf_node(SamplesIterator query_feature_first,
                                               SamplesIterator query_feature_last,
                                               ssize_t&        current_nearest_neighbor_index,
                                               DataType&       current_nearest_neighbor_distance,
                                               KDNodeViewPtr   kdnode) const;
    // (7)
    KDNodeViewPtr get_parent_node_after_sibling_traversal(SamplesIterator query_feature_first,
                                                          SamplesIterator query_feature_last,
                                                          ssize_t&        current_nearest_neighbor_index,
                                                          DataType&       current_nearest_neighbor_distance,
                                                          KDNodeViewPtr   kdnode) const;
    // (8)
    template <typename NearestNeighborsBufferType>
    void inner_k_nearest_neighbors_around_query_sample(SamplesIterator             query_feature_first,
                                                       SamplesIterator             query_feature_last,
                                                       NearestNeighborsBufferType& nearest_neighbors_buffer,
                                                       KDNodeViewPtr               kdnode = nullptr) const;
    // (8)
    template <typename NearestNeighborsBufferType>
    KDNodeViewPtr recurse_to_closest_leaf_node(SamplesIterator             query_feature_first,
                                               SamplesIterator             query_feature_last,
                                               NearestNeighborsBufferType& nearest_neighbors_buffer,
                                               KDNodeViewPtr               kdnode) const;
    // (8)
    template <typename NearestNeighborsBufferType>
    KDNodeViewPtr get_parent_node_after_sibling_traversal(SamplesIterator             query_feature_first,
                                                          SamplesIterator             query_feature_last,
                                                          NearestNeighborsBufferType& nearest_neighbors_buffer,
                                                          KDNodeViewPtr               kdnode) const;
    // (9)
    void inner_radius_count_around_query_sample(SamplesIterator query_feature_first,
                                                SamplesIterator query_feature_last,
                                                const DataType& radius,
                                                std::size_t&    neighbors_count,
                                                KDNodeViewPtr   kdnode = nullptr) const;
    // (9)
    KDNodeViewPtr recurse_to_closest_leaf_node(SamplesIterator query_feature_first,
                                               SamplesIterator query_feature_last,
                                               const DataType& radius,
                                               std::size_t&    neighbors_count,
                                               KDNodeViewPtr   kdnode) const;
    // (9)
    KDNodeViewPtr get_parent_node_after_sibling_traversal(SamplesIterator query_feature_first,
                                                          SamplesIterator query_feature_last,
                                                          const DataType& radius,
                                                          std::size_t&    neighbors_count,
                                                          KDNodeViewPtr   kdnode) const;
    // (10)
    template <typename NearestNeighborsBufferType>
    void inner_radius_search_around_query_sample(SamplesIterator             query_feature_first,
                                                 SamplesIterator             query_feature_last,
                                                 const DataType&             radius,
                                                 NearestNeighborsBufferType& nearest_neighbors_buffer,
                                                 KDNodeViewPtr               kdnode = nullptr) const;
    // (10)
    template <typename NearestNeighborsBufferType>
    KDNodeViewPtr recurse_to_closest_leaf_node(SamplesIterator             query_feature_first,
                                               SamplesIterator             query_feature_last,
                                               const DataType&             radius,
                                               NearestNeighborsBufferType& nearest_neighbors_buffer,
                                               KDNodeViewPtr               kdnode) const;
    // (10)
    template <typename NearestNeighborsBufferType>
    KDNodeViewPtr get_parent_node_after_sibling_traversal(SamplesIterator             query_feature_first,
                                                          SamplesIterator             query_feature_last,
                                                          const DataType&             radius,
                                                          NearestNeighborsBufferType& nearest_neighbors_buffer,
                                                          KDNodeViewPtr               kdnode) const;
    // (11)
    void inner_range_count_around_query_sample(SamplesIterator       query_feature_first,
                                               SamplesIterator       query_feature_last,
                                               const HyperRangeType& kd_bounding_box,
                                               std::size_t&          neighbors_count,
                                               KDNodeViewPtr         kdnode = nullptr) const;
    // (11)
    KDNodeViewPtr recurse_to_closest_leaf_node(SamplesIterator       query_feature_first,
                                               SamplesIterator       query_feature_last,
                                               const HyperRangeType& kd_bounding_box,
                                               std::size_t&          neighbors_count,
                                               KDNodeViewPtr         kdnode) const;
    // (11)
    KDNodeViewPtr get_parent_node_after_sibling_traversal(SamplesIterator       query_feature_first,
                                                          SamplesIterator       query_feature_last,
                                                          const HyperRangeType& kd_bounding_box,
                                                          std::size_t&          neighbors_count,
                                                          KDNodeViewPtr         kdnode) const;
    // (12)
    template <typename NearestNeighborsBufferType>
    void inner_range_search_around_query_sample(SamplesIterator             query_feature_first,
                                                SamplesIterator             query_feature_last,
                                                const HyperRangeType&       kd_bounding_box,
                                                NearestNeighborsBufferType& nearest_neighbors_buffer,
                                                KDNodeViewPtr               kdnode = nullptr) const;
    // (12)
    template <typename NearestNeighborsBufferType>
    KDNodeViewPtr recurse_to_closest_leaf_node(SamplesIterator             query_feature_first,
                                               SamplesIterator             query_feature_last,
                                               const HyperRangeType&       kd_bounding_box,
                                               NearestNeighborsBufferType& nearest_neighbors_buffer,
                                               KDNodeViewPtr               kdnode) const;
    // (12)
    template <typename NearestNeighborsBufferType>
    KDNodeViewPtr get_parent_node_after_sibling_traversal(SamplesIterator             query_feature_first,
                                                          SamplesIterator             query_feature_last,
                                                          const HyperRangeType&       kd_bounding_box,
                                                          NearestNeighborsBufferType& nearest_neighbors_buffer,
                                                          KDNodeViewPtr               kdnode) const;

    // Iterator pointing to the first element of the dataset.
    SamplesIterator samples_first_;
    // Iterator pointing to the last element of the dataset.
    SamplesIterator samples_last_;
    // The number of features in the dataset, used to represent data as a vectorized 2D array.
    std::size_t n_features_;
    // A hyperrectangle (bounding box) specifying the value bounds of the subset of data represented by the index array
    // from the entire dataset. This hyperrectangle is defined with respect to each dimension.
    HyperRangeType kd_bounding_box_;
    // Options used to configure the indexing structure.
    Options options_;
    // The root node of the indexing structure.
    KDNodeViewPtr root_;
};

template <typename IndicesIterator, typename SamplesIterator>
KDTree<IndicesIterator, SamplesIterator>::KDTree(IndicesIterator index_first,
                                                 IndicesIterator index_last,
                                                 SamplesIterator samples_first,
                                                 SamplesIterator samples_last,
                                                 std::size_t     n_features,
                                                 const Options&  options)
  : samples_first_{samples_first}
  , samples_last_{samples_last}
  , n_features_{n_features}
  , kd_bounding_box_{ffcl::bbox::make_kd_bounding_box(index_first,
                                                      index_last,
                                                      samples_first_,
                                                      samples_last_,
                                                      n_features_)}
  , options_{options}
  , root_{build(index_first,
                index_last,
                (*options_.axis_selection_policy_ptr_)(index_first,
                                                       index_last,
                                                       samples_first_,
                                                       samples_last_,
                                                       n_features_,
                                                       0,
                                                       kd_bounding_box_),
                0,
                kd_bounding_box_)} {}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t KDTree<IndicesIterator, SamplesIterator>::n_samples() const {
    return common::utils::get_n_samples(samples_first_, samples_last_, n_features_);
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t KDTree<IndicesIterator, SamplesIterator>::n_features() const {
    return n_features_;
}

template <typename IndicesIterator, typename SamplesIterator>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr KDTree<IndicesIterator, SamplesIterator>::build(
    IndicesIterator index_first,
    IndicesIterator index_last,
    ssize_t         cut_feature_index,
    ssize_t         depth,
    HyperRangeType& kd_bounding_box) {
    KDNodeViewPtr kdnode;
    // number of samples in the current node
    const std::size_t n_node_samples = std::distance(index_first, index_last);
    // if the current number of samples is greater than the target bucket size, the node is not leaf
    if (depth < options_.max_depth_ && n_node_samples > options_.bucket_size_) {
        // select the cut_feature_index according to the one with the most spread (min-max values)
        cut_feature_index = (*options_.axis_selection_policy_ptr_)(
            /**/ index_first,
            /**/ index_last,
            /**/ samples_first_,
            /**/ samples_last_,
            /**/ n_features_,
            /**/ depth,
            /**/ kd_bounding_box);

        auto [cut_index, left_index_range, cut_index_range, right_index_range] = (*options_.splitting_rule_policy_ptr_)(
            /**/ index_first,
            /**/ index_last,
            /**/ samples_first_,
            /**/ samples_last_,
            /**/ n_features_,
            /**/ cut_feature_index);

        kdnode =
            std::make_shared<KDNodeViewType>(cut_index_range, cut_feature_index, kd_bounding_box[cut_feature_index]);

        const auto cut_feature_value = samples_first_[*cut_index_range.first * n_features_ + cut_feature_index];
        {
            // set the right bound of the left child to the cut value
            kd_bounding_box[cut_feature_index].second = cut_feature_value;

            kdnode->left_ = build(/**/ left_index_range.first,
                                  /**/ left_index_range.second,
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

            kdnode->right_ = build(/**/ right_index_range.first,
                                   /**/ right_index_range.second,
                                   /**/ cut_feature_index,
                                   /**/ depth + 1,
                                   /**/ kd_bounding_box);
            // provide a parent pointer to the child node for reversed traversal of the kdtree
            kdnode->right_->parent_ = kdnode;

            // reset the left bound of the bounding box to the current kdnode left bound
            kd_bounding_box[cut_feature_index].first = kdnode->kd_bounding_box_.first;
        }
    } else {
        kdnode = std::make_shared<KDNodeViewType>(std::make_pair(index_first, index_last),
                                                  kd_bounding_box[cut_feature_index]);
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::k_mutual_reachability_distance(std::size_t query_index_1,
                                                                              std::size_t query_index_2,
                                                                              std::size_t k_nearest_neighbors) const {
    if (query_index_1 != query_index_2) {
        const auto furthest_nn_distance_1 =
            this->k_nearest_neighbors_around_query_index(query_index_1, k_nearest_neighbors)
                .furthest_k_nearest_neighbor_distance();

        const auto furthest_nn_distance_2 =
            this->k_nearest_neighbors_around_query_index(query_index_2, k_nearest_neighbors)
                .furthest_k_nearest_neighbor_distance();

        const auto queries_distance =
            math::heuristics::auto_distance(samples_first_ + query_index_1 * n_features_,
                                            samples_first_ + query_index_1 * n_features_ + n_features_,
                                            samples_first_ + query_index_2 * n_features_);

        return std::max({furthest_nn_distance_1, furthest_nn_distance_2, queries_distance});
    } else {
        return static_cast<DataType>(0);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::buffered_k_mutual_reachability_distance(
    std::size_t                        query_index_1,
    std::size_t                        query_index_2,
    const std::shared_ptr<DataType[]>& core_distances) const {
    if (query_index_1 != query_index_2) {
        const auto queries_distance =
            math::heuristics::auto_distance(samples_first_ + query_index_1 * n_features_,
                                            samples_first_ + query_index_1 * n_features_ + n_features_,
                                            samples_first_ + query_index_2 * n_features_);

        return std::max({core_distances[query_index_1], core_distances[query_index_2], queries_distance});
    } else {
        return static_cast<DataType>(0);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::nearest_neighbor_around_query_index(std::size_t query_index) const {
    ssize_t current_nearest_neighbor_index    = -1;
    auto    current_nearest_neighbor_distance = common::utils::infinity<DataType>();

    inner_nearest_neighbor_around_query_index(
        query_index, current_nearest_neighbor_index, current_nearest_neighbor_distance, root_);

    return std::make_pair(current_nearest_neighbor_index, current_nearest_neighbor_distance);
}

template <typename IndicesIterator, typename SamplesIterator>
void KDTree<IndicesIterator, SamplesIterator>::inner_nearest_neighbor_around_query_index(
    std::size_t   query_index,
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

template <typename IndicesIterator, typename SamplesIterator>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::recurse_to_closest_leaf_node(std::size_t   query_index,
                                                                       ssize_t&      current_nearest_neighbor_index,
                                                                       DataType&     current_nearest_neighbor_distance,
                                                                       KDNodeViewPtr kdnode) const {
    // update the current neighbor index and distance. No op if no candidate is closer or if the ranges are empty
    ffcl::knn::nearest_neighbor(kdnode->indices_iterator_pair_.first,
                                kdnode->indices_iterator_pair_.second,
                                samples_first_,
                                samples_last_,
                                n_features_,
                                query_index,
                                current_nearest_neighbor_index,
                                current_nearest_neighbor_distance);
    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode->cut_feature_index_];
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

template <typename IndicesIterator, typename SamplesIterator>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::get_parent_node_after_sibling_traversal(
    std::size_t   query_index,
    ssize_t&      current_nearest_neighbor_index,
    DataType&     current_nearest_neighbor_distance,
    KDNodeViewPtr kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();
    // if kdnode has a parent
    if (kdnode_parent) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode_parent->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode_parent->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = samples_first_[query_index * n_features_ + kdnode_parent->cut_feature_index_];
        // if the axiswise distance is equal to the current nearest neighbor distance, there could be a nearest neighbor
        // to the other side of the hyperrectangle since the values that are equal to the pivot are put to the right
        bool visit_sibling =
            kdnode->is_left_child()
                ? common::utils::abs(pivot_split_value - query_split_value) <= current_nearest_neighbor_distance
                : common::utils::abs(pivot_split_value - query_split_value) < current_nearest_neighbor_distance;
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // update the nearest neighbor from the sibling node
                inner_nearest_neighbor_around_query_index(
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

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
void KDTree<IndicesIterator, SamplesIterator>::buffered_k_nearest_neighbors_around_query_index(
    std::size_t                 query_index,
    NearestNeighborsBufferType& nearest_neighbors_buffer) const {
    static_assert(std::is_base_of_v<knn::NearestNeighborsBufferBase<SamplesIterator>, NearestNeighborsBufferType>,
                  "Derived class must inherit from NearestNeighborsBufferBase");

    inner_k_nearest_neighbors_around_query_index(query_index, nearest_neighbors_buffer);
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::k_nearest_neighbors_around_query_index(std::size_t query_index,
                                                                                      std::size_t n_neighbors) const {
    knn::NearestNeighborsBuffer<SamplesIterator> nearest_neighbors_buffer(n_neighbors);

    inner_k_nearest_neighbors_around_query_index(query_index, nearest_neighbors_buffer);

    return nearest_neighbors_buffer;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
void KDTree<IndicesIterator, SamplesIterator>::inner_k_nearest_neighbors_around_query_index(
    std::size_t                 query_index,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
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

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::recurse_to_closest_leaf_node(
    std::size_t                 query_index,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    // update the current k neighbors indices and distances. No op if no candidate is closer or if the ranges are empty
    ffcl::knn::k_nearest_neighbors(kdnode->indices_iterator_pair_.first,
                                   kdnode->indices_iterator_pair_.second,
                                   samples_first_,
                                   samples_last_,
                                   n_features_,
                                   query_index,
                                   nearest_neighbors_buffer);
    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode->cut_feature_index_];
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

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::get_parent_node_after_sibling_traversal(
    std::size_t                 query_index,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();
    // if kdnode has a parent
    if (kdnode_parent) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode_parent->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode_parent->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = samples_first_[query_index * n_features_ + kdnode_parent->cut_feature_index_];
        // if the axiswise distance is equal to the current furthest nearest neighbor distance, there could be a nearest
        // neighbor to the other side of the hyperrectangle since the values that are equal to the pivot are put to the
        // right
        bool visit_sibling = kdnode->is_left_child()
                                 ? nearest_neighbors_buffer.n_free_slots() ||
                                       common::utils::abs(pivot_split_value - query_split_value) <=
                                           nearest_neighbors_buffer.furthest_k_nearest_neighbor_distance()
                                 : nearest_neighbors_buffer.n_free_slots() ||
                                       common::utils::abs(pivot_split_value - query_split_value) <
                                           nearest_neighbors_buffer.furthest_k_nearest_neighbor_distance();
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // get the nearest neighbor from the sibling node
                inner_k_nearest_neighbors_around_query_index(
                    /**/ query_index,
                    /**/ nearest_neighbors_buffer,
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
    std::size_t neighbors_count = 0;

    inner_radius_count_around_query_index(query_index, radius, neighbors_count);

    return neighbors_count;
}

template <typename IndicesIterator, typename SamplesIterator>
void KDTree<IndicesIterator, SamplesIterator>::inner_radius_count_around_query_index(std::size_t     query_index,
                                                                                     const DataType& radius,
                                                                                     std::size_t&    neighbors_count,
                                                                                     KDNodeViewPtr   kdnode) const {
    // current_node is currently a leaf node (and root in the special case where the entire tree is in a single node)
    auto current_kdnode = recurse_to_closest_leaf_node(/**/ query_index,
                                                       /**/ radius,
                                                       /**/ neighbors_count,
                                                       /**/ kdnode == nullptr ? root_ : kdnode);

    // performs a radius count one step at a time from the leaf node until the input kdnode is reached if
    // kdnode parameter is a subtree. A search through the entire tree
    while (current_kdnode != kdnode) {
        // performs a radius count starting from the specified node then returns its parent if it exists
        // (nullptr otherwise)
        current_kdnode = get_parent_node_after_sibling_traversal(
            /**/ query_index,
            /**/ radius,
            /**/ neighbors_count,
            /**/ current_kdnode);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::recurse_to_closest_leaf_node(std::size_t     query_index,
                                                                       const DataType& radius,
                                                                       std::size_t&    neighbors_count,
                                                                       KDNodeViewPtr   kdnode) const {
    // update the current neighbors count if they are inside of the radius. No op if no candidate is closer or if the
    // ranges are empty
    ffcl::knn::increment_neighbors_count_in_radius(kdnode->indices_iterator_pair_.first,
                                                   kdnode->indices_iterator_pair_.second,
                                                   samples_first_,
                                                   samples_last_,
                                                   n_features_,
                                                   query_index,
                                                   radius,
                                                   neighbors_count);
    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = samples_first_[query_index * n_features_ + kdnode->cut_feature_index_];

        // traverse either the left or right child node depending on where the target sample is located relatively to
        // the cut value
        if (query_split_value < pivot_split_value) {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_index,
                /**/ radius,
                /**/ neighbors_count,
                /**/ kdnode->left_);
        } else {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_index,
                /**/ radius,
                /**/ neighbors_count,
                /**/ kdnode->right_);
        }
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::get_parent_node_after_sibling_traversal(std::size_t     query_index,
                                                                                  const DataType& radius,
                                                                                  std::size_t&    neighbors_count,
                                                                                  KDNodeViewPtr   kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();
    // if kdnode has a parent
    if (kdnode_parent) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode_parent->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode_parent->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = samples_first_[query_index * n_features_ + kdnode_parent->cut_feature_index_];
        // if the axiswise distance is equal to the radius of the search, there could be a nearest
        // neighbor to the other side of the hyperrectangle since the values that are equal to the pivot are put to the
        // right
        bool visit_sibling = kdnode->is_left_child()
                                 ? common::utils::abs(pivot_split_value - query_split_value) <= radius
                                 : common::utils::abs(pivot_split_value - query_split_value) < radius;
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // perform radius count from the sibling node
                inner_radius_count_around_query_index(
                    /**/ query_index,
                    /**/ radius,
                    /**/ neighbors_count,
                    /**/ sibling_node);
            }
        }
    }
    // returns nullptr if kdnode doesnt have parent (or is root)
    return kdnode_parent;
}

// /*
template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
void KDTree<IndicesIterator, SamplesIterator>::buffered_radius_search_around_query_index(
    std::size_t                 query_index,
    const DataType&             radius,
    NearestNeighborsBufferType& nearest_neighbors_buffer) const {
    static_assert(std::is_base_of_v<knn::NearestNeighborsBufferBase<SamplesIterator>, NearestNeighborsBufferType>,
                  "Derived class must inherit from NearestNeighborsBufferBase");

    inner_radius_search_around_query_index(query_index, radius, nearest_neighbors_buffer);
}
// */

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::radius_search_around_query_index(std::size_t     query_index,
                                                                                const DataType& radius) const {
    knn::NearestNeighborsBuffer<SamplesIterator> nearest_neighbors_buffer;

    inner_radius_search_around_query_index(query_index, radius, nearest_neighbors_buffer);

    return nearest_neighbors_buffer;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
void KDTree<IndicesIterator, SamplesIterator>::inner_radius_search_around_query_index(
    std::size_t                 query_index,
    const DataType&             radius,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    // current_node is currently a leaf node (and root in the special case where the entire tree is in a single node)
    auto current_kdnode = recurse_to_closest_leaf_node(
        /**/ query_index,
        /**/ radius,
        /**/ nearest_neighbors_buffer,
        /**/ kdnode == nullptr ? root_ : kdnode);

    // performs a nearest neighbor search one step at a time from the leaf node until the input kdnode is reached if
    // kdnode parameter is a subtree. A search through the entire tree
    while (current_kdnode != kdnode) {
        // performs a nearest neighbor search starting from the specified node then returns its parent if it exists
        // (nullptr otherwise)
        current_kdnode = get_parent_node_after_sibling_traversal(
            /**/ query_index,
            /**/ radius,
            /**/ nearest_neighbors_buffer,
            /**/ current_kdnode);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::recurse_to_closest_leaf_node(
    std::size_t                 query_index,
    const DataType&             radius,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    // update the current k neighbors indices and distances. No op if no candidate is closer or if the ranges are empty
    ffcl::knn::k_nearest_neighbors_in_radius(kdnode->indices_iterator_pair_.first,
                                             kdnode->indices_iterator_pair_.second,
                                             samples_first_,
                                             samples_last_,
                                             n_features_,
                                             query_index,
                                             radius,
                                             nearest_neighbors_buffer);
    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = samples_first_[query_index * n_features_ + kdnode->cut_feature_index_];

        // traverse either the left or right child node depending on where the target sample is located relatively to
        // the cut value
        if (query_split_value < pivot_split_value) {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_index,
                /**/ radius,
                /**/ nearest_neighbors_buffer,
                /**/ kdnode->left_);
        } else {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_index,
                /**/ radius,
                /**/ nearest_neighbors_buffer,
                /**/ kdnode->right_);
        }
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::get_parent_node_after_sibling_traversal(
    std::size_t                 query_index,
    const DataType&             radius,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();
    // if kdnode has a parent
    if (kdnode_parent) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode_parent->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode_parent->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = samples_first_[query_index * n_features_ + kdnode_parent->cut_feature_index_];
        // if the axiswise distance is equal to the current furthest nearest neighbor distance, there could be a nearest
        // neighbor to the other side of the hyperrectangle since the values that are equal to the pivot are put to the
        // right
        bool visit_sibling = kdnode->is_left_child()
                                 ? common::utils::abs(pivot_split_value - query_split_value) <= radius
                                 : common::utils::abs(pivot_split_value - query_split_value) < radius;
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // get the nearest neighbor from the sibling node
                inner_radius_search_around_query_index(
                    /**/ query_index,
                    /**/ radius,
                    /**/ nearest_neighbors_buffer,
                    /**/ sibling_node);
            }
        }
    }
    // returns nullptr if kdnode doesnt have parent (or is root)
    return kdnode_parent;
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t KDTree<IndicesIterator, SamplesIterator>::range_count_around_query_index(
    std::size_t           query_index,
    const HyperRangeType& kd_bounding_box) const {
    std::size_t neighbors_count = 0;

    const auto translated_kd_bounding_box = ffcl::bbox::relative_coordinates_sequence_to_range_bounding_box(
        /**/ samples_first_ + query_index * n_features_,
        /**/ samples_first_ + query_index * n_features_ + n_features_,
        /**/ kd_bounding_box);

    inner_range_count_around_query_index(query_index, translated_kd_bounding_box, neighbors_count);

    return neighbors_count;
}

template <typename IndicesIterator, typename SamplesIterator>
void KDTree<IndicesIterator, SamplesIterator>::inner_range_count_around_query_index(
    std::size_t           query_index,
    const HyperRangeType& kd_bounding_box,
    std::size_t&          neighbors_count,
    KDNodeViewPtr         kdnode) const {
    // current_node is currently a leaf node (and root in the special case where the entire tree is in a single node)
    auto current_kdnode = recurse_to_closest_leaf_node(/**/ query_index,
                                                       /**/ kd_bounding_box,
                                                       /**/ neighbors_count,
                                                       /**/ kdnode == nullptr ? root_ : kdnode);

    // performs a radius count one step at a time from the leaf node until the input kdnode is reached if
    // kdnode parameter is a subtree. A search through the entire tree
    while (current_kdnode != kdnode) {
        // performs a radius count starting from the specified node then returns its parent if it exists
        // (nullptr otherwise)
        current_kdnode = get_parent_node_after_sibling_traversal(
            /**/ query_index,
            /**/ kd_bounding_box,
            /**/ neighbors_count,
            /**/ current_kdnode);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::recurse_to_closest_leaf_node(std::size_t           query_index,
                                                                       const HyperRangeType& kd_bounding_box,
                                                                       std::size_t&          neighbors_count,
                                                                       KDNodeViewPtr         kdnode) const {
    // update the current neighbors count if they are inside of the radius. No op if no candidate is closer or if the
    // ranges are empty
    ffcl::knn::increment_neighbors_count_in_kd_bounding_box(kdnode->indices_iterator_pair_.first,
                                                            kdnode->indices_iterator_pair_.second,
                                                            samples_first_,
                                                            samples_last_,
                                                            n_features_,
                                                            query_index,
                                                            kd_bounding_box,
                                                            neighbors_count);
    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = samples_first_[query_index * n_features_ + kdnode->cut_feature_index_];

        // traverse either the left or right child node depending on where the target sample is located relatively to
        // the cut value
        if (query_split_value < pivot_split_value) {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_index,
                /**/ kd_bounding_box,
                /**/ neighbors_count,
                /**/ kdnode->left_);
        } else {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_index,
                /**/ kd_bounding_box,
                /**/ neighbors_count,
                /**/ kdnode->right_);
        }
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::get_parent_node_after_sibling_traversal(std::size_t           query_index,
                                                                                  const HyperRangeType& kd_bounding_box,
                                                                                  std::size_t&          neighbors_count,
                                                                                  KDNodeViewPtr         kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();
    // if kdnode has a parent
    if (kdnode_parent) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode_parent->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode_parent->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = samples_first_[query_index * n_features_ + kdnode_parent->cut_feature_index_];
        // if the axiswise distance is equal to size of the bounding box divided by 2 along the current cut dimension
        // (the query is at the center of the bounding box), there could be a nearest neighbor to the other side of the
        // hyperrectangle since the values that are equal to the pivot are put to the right
        bool visit_sibling = kdnode->is_left_child() ? common::utils::abs(pivot_split_value - query_split_value) <=
                                                           (kd_bounding_box[kdnode_parent->cut_feature_index_].second -
                                                            kd_bounding_box[kdnode_parent->cut_feature_index_].first) /
                                                               2
                                                     : common::utils::abs(pivot_split_value - query_split_value) <
                                                           (kd_bounding_box[kdnode_parent->cut_feature_index_].second -
                                                            kd_bounding_box[kdnode_parent->cut_feature_index_].first) /
                                                               2;
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // perform radius count from the sibling node
                inner_range_count_around_query_index(
                    /**/ query_index,
                    /**/ kd_bounding_box,
                    /**/ neighbors_count,
                    /**/ sibling_node);
            }
        }
    }
    // returns nullptr if kdnode doesnt have parent (or is root)
    return kdnode_parent;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
void KDTree<IndicesIterator, SamplesIterator>::buffered_range_search_around_query_index(
    std::size_t                 query_index,
    const HyperRangeType&       kd_bounding_box,
    NearestNeighborsBufferType& nearest_neighbors_buffer) const {
    static_assert(std::is_base_of_v<knn::NearestNeighborsBufferBase<SamplesIterator>, NearestNeighborsBufferType>,
                  "Derived class must inherit from NearestNeighborsBufferBase");

    const auto translated_kd_bounding_box = ffcl::bbox::relative_coordinates_sequence_to_range_bounding_box(
        /**/ samples_first_ + query_index * n_features_,
        /**/ samples_first_ + query_index * n_features_ + n_features_,
        /**/ kd_bounding_box);

    inner_range_search_around_query_index(query_index, translated_kd_bounding_box, nearest_neighbors_buffer);
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::range_search_around_query_index(
    std::size_t           query_index,
    const HyperRangeType& kd_bounding_box) const {
    const auto translated_kd_bounding_box = ffcl::bbox::relative_coordinates_sequence_to_range_bounding_box(
        /**/ samples_first_ + query_index * n_features_,
        /**/ samples_first_ + query_index * n_features_ + n_features_,
        /**/ kd_bounding_box);

    knn::NearestNeighborsBuffer<SamplesIterator> nearest_neighbors_buffer;

    inner_range_search_around_query_index(query_index, translated_kd_bounding_box, nearest_neighbors_buffer);

    return nearest_neighbors_buffer;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
void KDTree<IndicesIterator, SamplesIterator>::inner_range_search_around_query_index(
    std::size_t                 query_index,
    const HyperRangeType&       kd_bounding_box,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    // current_node is currently a leaf node (and root in the special case where the entire tree is in a single node)
    auto current_kdnode = recurse_to_closest_leaf_node(
        /**/ query_index,
        /**/ kd_bounding_box,
        /**/ nearest_neighbors_buffer,
        /**/ kdnode == nullptr ? root_ : kdnode);

    // performs a nearest neighbor search one step at a time from the leaf node until the input kdnode is reached if
    // kdnode parameter is a subtree. A search through the entire tree
    while (current_kdnode != kdnode) {
        // performs a nearest neighbor search starting from the specified node then returns its parent if it exists
        // (nullptr otherwise)
        current_kdnode = get_parent_node_after_sibling_traversal(
            /**/ query_index,
            /**/ kd_bounding_box,
            /**/ nearest_neighbors_buffer,
            /**/ current_kdnode);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::recurse_to_closest_leaf_node(
    std::size_t                 query_index,
    const HyperRangeType&       kd_bounding_box,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    // update the current k neighbors indices and distances. No op if no candidate is closer or if the ranges are empty
    ffcl::knn::k_nearest_neighbors_in_kd_bounding_box(kdnode->indices_iterator_pair_.first,
                                                      kdnode->indices_iterator_pair_.second,
                                                      samples_first_,
                                                      samples_last_,
                                                      n_features_,
                                                      query_index,
                                                      kd_bounding_box,
                                                      nearest_neighbors_buffer);
    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = samples_first_[query_index * n_features_ + kdnode->cut_feature_index_];

        // traverse either the left or right child node depending on where the target sample is located relatively to
        // the cut value
        if (query_split_value < pivot_split_value) {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_index,
                /**/ kd_bounding_box,
                /**/ nearest_neighbors_buffer,
                /**/ kdnode->left_);
        } else {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_index,
                /**/ kd_bounding_box,
                /**/ nearest_neighbors_buffer,
                /**/ kdnode->right_);
        }
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::get_parent_node_after_sibling_traversal(
    std::size_t                 query_index,
    const HyperRangeType&       kd_bounding_box,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();
    // if kdnode has a parent
    if (kdnode_parent) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode_parent->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode_parent->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = samples_first_[query_index * n_features_ + kdnode_parent->cut_feature_index_];
        // if the axiswise distance is equal to size of the bounding box divided by 2 along the current cut dimension
        // (the query is at the center of the bounding box), there could be a nearest neighbor to the other side of the
        // hyperrectangle since the values that are equal to the pivot are put to the right
        bool visit_sibling = kdnode->is_left_child() ? common::utils::abs(pivot_split_value - query_split_value) <=
                                                           (kd_bounding_box[kdnode_parent->cut_feature_index_].second -
                                                            kd_bounding_box[kdnode_parent->cut_feature_index_].first) /
                                                               2
                                                     : common::utils::abs(pivot_split_value - query_split_value) <
                                                           (kd_bounding_box[kdnode_parent->cut_feature_index_].second -
                                                            kd_bounding_box[kdnode_parent->cut_feature_index_].first) /
                                                               2;
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // get the nearest neighbor from the sibling node
                inner_range_search_around_query_index(
                    /**/ query_index,
                    /**/ kd_bounding_box,
                    /**/ nearest_neighbors_buffer,
                    /**/ sibling_node);
            }
        }
    }
    // returns nullptr if kdnode doesnt have parent (or is root)
    return kdnode_parent;
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::nearest_neighbor_around_query_sample(
    SamplesIterator query_feature_first,
    SamplesIterator query_feature_last) const {
    ssize_t current_nearest_neighbor_index    = -1;
    auto    current_nearest_neighbor_distance = common::utils::infinity<DataType>();

    inner_nearest_neighbor_around_query_sample(query_feature_first,
                                               query_feature_last,
                                               current_nearest_neighbor_index,
                                               current_nearest_neighbor_distance,
                                               root_);

    return std::make_pair(current_nearest_neighbor_index, current_nearest_neighbor_distance);
}

template <typename IndicesIterator, typename SamplesIterator>
void KDTree<IndicesIterator, SamplesIterator>::inner_nearest_neighbor_around_query_sample(
    SamplesIterator query_feature_first,
    SamplesIterator query_feature_last,
    ssize_t&        current_nearest_neighbor_index,
    DataType&       current_nearest_neighbor_distance,
    KDNodeViewPtr   kdnode) const {
    // current_node is currently a leaf node (and root in the special case where the entire tree is in a single node)
    auto current_kdnode = recurse_to_closest_leaf_node(query_feature_first,
                                                       query_feature_last,
                                                       current_nearest_neighbor_index,
                                                       current_nearest_neighbor_distance,
                                                       kdnode == nullptr ? root_ : kdnode);
    // performs a nearest neighbor search one step at a time from the leaf node until the input kdnode is reached if
    // kdnode parameter is a subtree. A search through the entire tree
    while (current_kdnode != kdnode) {
        // performs a nearest neighbor search starting from the specified node then returns its parent if it exists
        // (nullptr otherwise)
        current_kdnode = get_parent_node_after_sibling_traversal(
            /**/ query_feature_first,
            /**/ query_feature_last,
            /**/ current_nearest_neighbor_index,
            /**/ current_nearest_neighbor_distance,
            /**/ current_kdnode);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::recurse_to_closest_leaf_node(SamplesIterator query_feature_first,
                                                                       SamplesIterator query_feature_last,
                                                                       ssize_t&        current_nearest_neighbor_index,
                                                                       DataType&     current_nearest_neighbor_distance,
                                                                       KDNodeViewPtr kdnode) const {
    // update the current neighbor index and distance. No op if no candidate is closer or if the ranges are empty
    ffcl::knn::nearest_neighbor(kdnode->indices_iterator_pair_.first,
                                kdnode->indices_iterator_pair_.second,
                                samples_first_,
                                samples_last_,
                                n_features_,
                                query_feature_first,
                                query_feature_last,
                                current_nearest_neighbor_index,
                                current_nearest_neighbor_distance);
    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = query_feature_first[kdnode->cut_feature_index_];

        // traverse either the left or right child node depending on where the target sample is located relatively to
        // the cut value
        if (query_split_value < pivot_split_value) {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ current_nearest_neighbor_index,
                /**/ current_nearest_neighbor_distance,
                /**/ kdnode->left_);
        } else {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ current_nearest_neighbor_index,
                /**/ current_nearest_neighbor_distance,
                /**/ kdnode->right_);
        }
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::get_parent_node_after_sibling_traversal(
    SamplesIterator query_feature_first,
    SamplesIterator query_feature_last,
    ssize_t&        current_nearest_neighbor_index,
    DataType&       current_nearest_neighbor_distance,
    KDNodeViewPtr   kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();
    // if kdnode has a parent
    if (kdnode_parent) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode_parent->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode_parent->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = query_feature_first[kdnode_parent->cut_feature_index_];
        // if the axiswise distance is equal to the current nearest neighbor distance, there could be a nearest neighbor
        // to the other side of the hyperrectangle since the values that are equal to the pivot are put to the right
        bool visit_sibling =
            kdnode->is_left_child()
                ? common::utils::abs(pivot_split_value - query_split_value) <= current_nearest_neighbor_distance
                : common::utils::abs(pivot_split_value - query_split_value) < current_nearest_neighbor_distance;
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // update the nearest neighbor from the sibling node
                inner_nearest_neighbor_around_query_sample(
                    /**/ query_feature_first,
                    /**/ query_feature_last,
                    /**/ current_nearest_neighbor_index,
                    /**/ current_nearest_neighbor_distance,
                    /**/ sibling_node);
            }
        }
    }
    // returns nullptr if kdnode doesnt have parent (or is root)
    return kdnode_parent;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
void KDTree<IndicesIterator, SamplesIterator>::buffered_k_nearest_neighbors_around_query_sample(
    SamplesIterator             query_feature_first,
    SamplesIterator             query_feature_last,
    NearestNeighborsBufferType& nearest_neighbors_buffer) const {
    static_assert(std::is_base_of_v<knn::NearestNeighborsBufferBase<SamplesIterator>, NearestNeighborsBufferType>,
                  "Derived class must inherit from NearestNeighborsBufferBase");

    inner_k_nearest_neighbors_around_query_sample(query_feature_first, query_feature_last, nearest_neighbors_buffer);
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::k_nearest_neighbors_around_query_sample(
    SamplesIterator query_feature_first,
    SamplesIterator query_feature_last,
    std::size_t     n_neighbors) const {
    knn::NearestNeighborsBuffer<SamplesIterator> nearest_neighbors_buffer(n_neighbors);

    inner_k_nearest_neighbors_around_query_sample(query_feature_first, query_feature_last, nearest_neighbors_buffer);

    return nearest_neighbors_buffer;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
void KDTree<IndicesIterator, SamplesIterator>::inner_k_nearest_neighbors_around_query_sample(
    SamplesIterator             query_feature_first,
    SamplesIterator             query_feature_last,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    // current_node is currently a leaf node (and root in the special case where the entire tree is in a single node)
    auto current_kdnode = recurse_to_closest_leaf_node(
        /**/ query_feature_first,
        /**/ query_feature_last,
        /**/ nearest_neighbors_buffer,
        /**/ kdnode == nullptr ? root_ : kdnode);

    // performs a nearest neighbor search one step at a time from the leaf node until the input kdnode is reached if
    // kdnode parameter is a subtree. A search through the entire tree
    while (current_kdnode != kdnode) {
        // performs a nearest neighbor search starting from the specified node then returns its parent if it exists
        // (nullptr otherwise)
        current_kdnode = get_parent_node_after_sibling_traversal(
            /**/ query_feature_first,
            /**/ query_feature_last,
            /**/ nearest_neighbors_buffer,
            /**/ current_kdnode);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::recurse_to_closest_leaf_node(
    SamplesIterator             query_feature_first,
    SamplesIterator             query_feature_last,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    // update the current k neighbors indices and distances. No op if no candidate is closer or if the ranges are empty
    ffcl::knn::k_nearest_neighbors(kdnode->indices_iterator_pair_.first,
                                   kdnode->indices_iterator_pair_.second,
                                   samples_first_,
                                   samples_last_,
                                   n_features_,
                                   query_feature_first,
                                   query_feature_last,
                                   nearest_neighbors_buffer);

    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = query_feature_first[kdnode->cut_feature_index_];

        // traverse either the left or right child node depending on where the target sample is located relatively to
        // the cut value
        if (query_split_value < pivot_split_value) {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ nearest_neighbors_buffer,
                /**/ kdnode->left_);
        } else {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ nearest_neighbors_buffer,
                /**/ kdnode->right_);
        }
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::get_parent_node_after_sibling_traversal(
    SamplesIterator             query_feature_first,
    SamplesIterator             query_feature_last,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();
    // if kdnode has a parent
    if (kdnode_parent) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode_parent->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode_parent->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = query_feature_first[kdnode_parent->cut_feature_index_];
        // if the axiswise distance is equal to the current furthest nearest neighbor distance, there could be a nearest
        // neighbor to the other side of the hyperrectangle since the values that are equal to the pivot are put to the
        // right
        bool visit_sibling = kdnode->is_left_child()
                                 ? nearest_neighbors_buffer.n_free_slots() ||
                                       common::utils::abs(pivot_split_value - query_split_value) <=
                                           nearest_neighbors_buffer.furthest_k_nearest_neighbor_distance()
                                 : nearest_neighbors_buffer.n_free_slots() ||
                                       common::utils::abs(pivot_split_value - query_split_value) <
                                           nearest_neighbors_buffer.furthest_k_nearest_neighbor_distance();
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // get the nearest neighbor from the sibling node
                inner_k_nearest_neighbors_around_query_sample(
                    /**/ query_feature_first,
                    /**/ query_feature_last,
                    /**/ nearest_neighbors_buffer,
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
    std::size_t neighbors_count = 0;

    inner_radius_count_around_query_sample(query_feature_first, query_feature_last, radius, neighbors_count);

    return neighbors_count;
}

template <typename IndicesIterator, typename SamplesIterator>
void KDTree<IndicesIterator, SamplesIterator>::inner_radius_count_around_query_sample(
    SamplesIterator query_feature_first,
    SamplesIterator query_feature_last,
    const DataType& radius,
    std::size_t&    neighbors_count,
    KDNodeViewPtr   kdnode) const {
    // current_node is currently a leaf node (and root in the special case where the entire tree is in a single node)
    auto current_kdnode = recurse_to_closest_leaf_node(/**/ query_feature_first,
                                                       /**/ query_feature_last,
                                                       /**/ radius,
                                                       /**/ neighbors_count,
                                                       /**/ kdnode == nullptr ? root_ : kdnode);

    // performs a radius count one step at a time from the leaf node until the input kdnode is reached if
    // kdnode parameter is a subtree. A search through the entire tree
    while (current_kdnode != kdnode) {
        // performs a radius count starting from the specified node then returns its parent if it exists
        // (nullptr otherwise)
        current_kdnode = get_parent_node_after_sibling_traversal(
            /**/ query_feature_first,
            /**/ query_feature_last,
            /**/ radius,
            /**/ neighbors_count,
            /**/ current_kdnode);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::recurse_to_closest_leaf_node(SamplesIterator query_feature_first,
                                                                       SamplesIterator query_feature_last,
                                                                       const DataType& radius,
                                                                       std::size_t&    neighbors_count,
                                                                       KDNodeViewPtr   kdnode) const {
    // update the current neighbors count if they are inside of the radius. No op if no candidate is closer or if the
    // ranges are empty
    ffcl::knn::increment_neighbors_count_in_radius(kdnode->indices_iterator_pair_.first,
                                                   kdnode->indices_iterator_pair_.second,
                                                   samples_first_,
                                                   samples_last_,
                                                   n_features_,
                                                   query_feature_first,
                                                   query_feature_last,
                                                   radius,
                                                   neighbors_count);
    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = query_feature_first[kdnode->cut_feature_index_];

        // traverse either the left or right child node depending on where the target sample is located relatively to
        // the cut value
        if (query_split_value < pivot_split_value) {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ radius,
                /**/ neighbors_count,
                /**/ kdnode->left_);
        } else {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ radius,
                /**/ neighbors_count,
                /**/ kdnode->right_);
        }
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::get_parent_node_after_sibling_traversal(SamplesIterator query_feature_first,
                                                                                  SamplesIterator query_feature_last,
                                                                                  const DataType& radius,
                                                                                  std::size_t&    neighbors_count,
                                                                                  KDNodeViewPtr   kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();
    // if kdnode has a parent
    if (kdnode_parent) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode_parent->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode_parent->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = query_feature_first[kdnode_parent->cut_feature_index_];
        // if the axiswise distance is equal to the radius of the search, there could be a nearest
        // neighbor to the other side of the hyperrectangle since the values that are equal to the pivot are put to the
        // right
        bool visit_sibling = kdnode->is_left_child()
                                 ? common::utils::abs(pivot_split_value - query_split_value) <= radius
                                 : common::utils::abs(pivot_split_value - query_split_value) < radius;
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // perform radius count from the sibling node
                inner_radius_count_around_query_sample(
                    /**/ query_feature_first,
                    /**/ query_feature_last,
                    /**/ radius,
                    /**/ neighbors_count,
                    /**/ sibling_node);
            }
        }
    }
    // returns nullptr if kdnode doesnt have parent (or is root)
    return kdnode_parent;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
void KDTree<IndicesIterator, SamplesIterator>::buffered_radius_search_around_query_sample(
    SamplesIterator             query_feature_first,
    SamplesIterator             query_feature_last,
    const DataType&             radius,
    NearestNeighborsBufferType& nearest_neighbors_buffer) const {
    static_assert(std::is_base_of_v<knn::NearestNeighborsBufferBase<SamplesIterator>, NearestNeighborsBufferType>,
                  "Derived class must inherit from NearestNeighborsBufferBase");

    inner_radius_search_around_query_sample(query_feature_first, query_feature_last, radius, nearest_neighbors_buffer);
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::radius_search_around_query_sample(SamplesIterator query_feature_first,
                                                                                 SamplesIterator query_feature_last,
                                                                                 const DataType& radius) const {
    knn::NearestNeighborsBuffer<SamplesIterator> nearest_neighbors_buffer;

    inner_radius_search_around_query_sample(query_feature_first, query_feature_last, radius, nearest_neighbors_buffer);

    return nearest_neighbors_buffer;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
void KDTree<IndicesIterator, SamplesIterator>::inner_radius_search_around_query_sample(
    SamplesIterator             query_feature_first,
    SamplesIterator             query_feature_last,
    const DataType&             radius,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    // current_node is currently a leaf node (and root in the special case where the entire tree is in a single node)
    auto current_kdnode = recurse_to_closest_leaf_node(
        /**/ query_feature_first,
        /**/ query_feature_last,
        /**/ radius,
        /**/ nearest_neighbors_buffer,
        /**/ kdnode == nullptr ? root_ : kdnode);

    // performs a nearest neighbor search one step at a time from the leaf node until the input kdnode is reached if
    // kdnode parameter is a subtree. A search through the entire tree
    while (current_kdnode != kdnode) {
        // performs a nearest neighbor search starting from the specified node then returns its parent if it exists
        // (nullptr otherwise)
        current_kdnode = get_parent_node_after_sibling_traversal(
            /**/ query_feature_first,
            /**/ query_feature_last,
            /**/ radius,
            /**/ nearest_neighbors_buffer,
            /**/ current_kdnode);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::recurse_to_closest_leaf_node(
    SamplesIterator             query_feature_first,
    SamplesIterator             query_feature_last,
    const DataType&             radius,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    // update the current k neighbors indices and distances. No op if no candidate is closer or if the ranges are empty
    ffcl::knn::k_nearest_neighbors_in_radius(kdnode->indices_iterator_pair_.first,
                                             kdnode->indices_iterator_pair_.second,
                                             samples_first_,
                                             samples_last_,
                                             n_features_,
                                             query_feature_first,
                                             query_feature_last,
                                             radius,
                                             nearest_neighbors_buffer);
    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = query_feature_first[kdnode->cut_feature_index_];

        // traverse either the left or right child node depending on where the target sample is located relatively to
        // the cut value
        if (query_split_value < pivot_split_value) {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ radius,
                /**/ nearest_neighbors_buffer,
                /**/ kdnode->left_);
        } else {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ radius,
                /**/ nearest_neighbors_buffer,
                /**/ kdnode->right_);
        }
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::get_parent_node_after_sibling_traversal(
    SamplesIterator             query_feature_first,
    SamplesIterator             query_feature_last,
    const DataType&             radius,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();
    // if kdnode has a parent
    if (kdnode_parent) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode_parent->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode_parent->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = query_feature_first[kdnode_parent->cut_feature_index_];
        // if the axiswise distance is equal to the current furthest nearest neighbor distance, there could be a nearest
        // neighbor to the other side of the hyperrectangle since the values that are equal to the pivot are put to the
        // right
        bool visit_sibling = kdnode->is_left_child()
                                 ? common::utils::abs(pivot_split_value - query_split_value) <= radius
                                 : common::utils::abs(pivot_split_value - query_split_value) < radius;
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // get the nearest neighbor from the sibling node
                inner_radius_search_around_query_sample(
                    /**/ query_feature_first,
                    /**/ query_feature_last,
                    /**/ radius,
                    /**/ nearest_neighbors_buffer,
                    /**/ sibling_node);
            }
        }
    }
    // returns nullptr if kdnode doesnt have parent (or is root)
    return kdnode_parent;
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t KDTree<IndicesIterator, SamplesIterator>::range_count_around_query_sample(
    SamplesIterator       query_feature_first,
    SamplesIterator       query_feature_last,
    const HyperRangeType& kd_bounding_box) const {
    std::size_t neighbors_count = 0;

    const auto translated_kd_bounding_box = ffcl::bbox::relative_coordinates_sequence_to_range_bounding_box(
        query_feature_first, query_feature_last, kd_bounding_box);

    inner_range_count_around_query_sample(
        query_feature_first, query_feature_last, translated_kd_bounding_box, neighbors_count);

    return neighbors_count;
}

template <typename IndicesIterator, typename SamplesIterator>
void KDTree<IndicesIterator, SamplesIterator>::inner_range_count_around_query_sample(
    SamplesIterator       query_feature_first,
    SamplesIterator       query_feature_last,
    const HyperRangeType& kd_bounding_box,
    std::size_t&          neighbors_count,
    KDNodeViewPtr         kdnode) const {
    // current_node is currently a leaf node (and root in the special case where the entire tree is in a single node)
    auto current_kdnode = recurse_to_closest_leaf_node(/**/ query_feature_first,
                                                       /**/ query_feature_last,
                                                       /**/ kd_bounding_box,
                                                       /**/ neighbors_count,
                                                       /**/ kdnode == nullptr ? root_ : kdnode);

    // performs a radius count one step at a time from the leaf node until the input kdnode is reached if
    // kdnode parameter is a subtree. A search through the entire tree
    while (current_kdnode != kdnode) {
        // performs a radius count starting from the specified node then returns its parent if it exists
        // (nullptr otherwise)
        current_kdnode = get_parent_node_after_sibling_traversal(
            /**/ query_feature_first,
            /**/ query_feature_last,
            /**/ kd_bounding_box,
            /**/ neighbors_count,
            /**/ current_kdnode);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::recurse_to_closest_leaf_node(SamplesIterator       query_feature_first,
                                                                       SamplesIterator       query_feature_last,
                                                                       const HyperRangeType& kd_bounding_box,
                                                                       std::size_t&          neighbors_count,
                                                                       KDNodeViewPtr         kdnode) const {
    // update the current neighbors count if they are inside of the radius. No op if no candidate is closer or if the
    // ranges are empty
    ffcl::knn::increment_neighbors_count_in_kd_bounding_box(kdnode->indices_iterator_pair_.first,
                                                            kdnode->indices_iterator_pair_.second,
                                                            samples_first_,
                                                            samples_last_,
                                                            n_features_,
                                                            query_feature_first,
                                                            query_feature_last,
                                                            kd_bounding_box,
                                                            neighbors_count);
    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = query_feature_first[kdnode->cut_feature_index_];

        // traverse either the left or right child node depending on where the target sample is located relatively to
        // the cut value
        if (query_split_value < pivot_split_value) {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ kd_bounding_box,
                /**/ neighbors_count,
                /**/ kdnode->left_);
        } else {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ kd_bounding_box,
                /**/ neighbors_count,
                /**/ kdnode->right_);
        }
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::get_parent_node_after_sibling_traversal(SamplesIterator query_feature_first,
                                                                                  SamplesIterator query_feature_last,
                                                                                  const HyperRangeType& kd_bounding_box,
                                                                                  std::size_t&          neighbors_count,
                                                                                  KDNodeViewPtr         kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();
    // if kdnode has a parent
    if (kdnode_parent) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode_parent->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode_parent->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = query_feature_first[kdnode_parent->cut_feature_index_];
        // if the axiswise distance is equal to size of the bounding box divided by 2 along the current cut dimension
        // (the query is at the center of the bounding box), there could be a nearest neighbor to the other side of the
        // hyperrectangle since the values that are equal to the pivot are put to the right
        bool visit_sibling = kdnode->is_left_child() ? common::utils::abs(pivot_split_value - query_split_value) <=
                                                           (kd_bounding_box[kdnode_parent->cut_feature_index_].second -
                                                            kd_bounding_box[kdnode_parent->cut_feature_index_].first) /
                                                               2
                                                     : common::utils::abs(pivot_split_value - query_split_value) <
                                                           (kd_bounding_box[kdnode_parent->cut_feature_index_].second -
                                                            kd_bounding_box[kdnode_parent->cut_feature_index_].first) /
                                                               2;
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // perform radius count from the sibling node
                inner_range_count_around_query_sample(
                    /**/ query_feature_first,
                    /**/ query_feature_last,
                    /**/ kd_bounding_box,
                    /**/ neighbors_count,
                    /**/ sibling_node);
            }
        }
    }
    // returns nullptr if kdnode doesnt have parent (or is root)
    return kdnode_parent;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
void KDTree<IndicesIterator, SamplesIterator>::buffered_range_search_around_query_sample(
    SamplesIterator             query_feature_first,
    SamplesIterator             query_feature_last,
    const HyperRangeType&       kd_bounding_box,
    NearestNeighborsBufferType& nearest_neighbors_buffer) const {
    static_assert(std::is_base_of_v<knn::NearestNeighborsBufferBase<SamplesIterator>, NearestNeighborsBufferType>,
                  "Derived class must inherit from NearestNeighborsBufferBase");

    const auto translated_kd_bounding_box = ffcl::bbox::relative_coordinates_sequence_to_range_bounding_box(
        query_feature_first, query_feature_last, kd_bounding_box);

    inner_range_search_around_query_sample(
        query_feature_first, query_feature_last, translated_kd_bounding_box, nearest_neighbors_buffer);
}

template <typename IndicesIterator, typename SamplesIterator>
auto KDTree<IndicesIterator, SamplesIterator>::range_search_around_query_sample(
    SamplesIterator       query_feature_first,
    SamplesIterator       query_feature_last,
    const HyperRangeType& kd_bounding_box) const {
    const auto translated_kd_bounding_box = ffcl::bbox::relative_coordinates_sequence_to_range_bounding_box(
        query_feature_first, query_feature_last, kd_bounding_box);

    knn::NearestNeighborsBuffer<SamplesIterator> nearest_neighbors_buffer;

    inner_range_search_around_query_sample(
        query_feature_first, query_feature_last, translated_kd_bounding_box, nearest_neighbors_buffer);

    return nearest_neighbors_buffer;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
void KDTree<IndicesIterator, SamplesIterator>::inner_range_search_around_query_sample(
    SamplesIterator             query_feature_first,
    SamplesIterator             query_feature_last,
    const HyperRangeType&       kd_bounding_box,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    // current_node is currently a leaf node (and root in the special case where the entire tree is in a single node)
    auto current_kdnode = recurse_to_closest_leaf_node(
        /**/ query_feature_first,
        /**/ query_feature_last,
        /**/ kd_bounding_box,
        /**/ nearest_neighbors_buffer,
        /**/ kdnode == nullptr ? root_ : kdnode);

    // performs a nearest neighbor search one step at a time from the leaf node until the input kdnode is reached if
    // kdnode parameter is a subtree. A search through the entire tree
    while (current_kdnode != kdnode) {
        // performs a nearest neighbor search starting from the specified node then returns its parent if it exists
        // (nullptr otherwise)
        current_kdnode = get_parent_node_after_sibling_traversal(
            /**/ query_feature_first,
            /**/ query_feature_last,
            /**/ kd_bounding_box,
            /**/ nearest_neighbors_buffer,
            /**/ current_kdnode);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::recurse_to_closest_leaf_node(
    SamplesIterator             query_feature_first,
    SamplesIterator             query_feature_last,
    const HyperRangeType&       kd_bounding_box,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    // update the current k neighbors indices and distances. No op if no candidate is closer or if the ranges are empty
    ffcl::knn::k_nearest_neighbors_in_kd_bounding_box(kdnode->indices_iterator_pair_.first,
                                                      kdnode->indices_iterator_pair_.second,
                                                      samples_first_,
                                                      samples_last_,
                                                      n_features_,
                                                      query_feature_first,
                                                      query_feature_last,
                                                      kd_bounding_box,
                                                      nearest_neighbors_buffer);
    // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
    if (!kdnode->is_leaf()) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = query_feature_first[kdnode->cut_feature_index_];

        // traverse either the left or right child node depending on where the target sample is located relatively to
        // the cut value
        if (query_split_value < pivot_split_value) {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ kd_bounding_box,
                /**/ nearest_neighbors_buffer,
                /**/ kdnode->left_);
        } else {
            kdnode = recurse_to_closest_leaf_node(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ kd_bounding_box,
                /**/ nearest_neighbors_buffer,
                /**/ kdnode->right_);
        }
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
template <typename NearestNeighborsBufferType>
typename KDTree<IndicesIterator, SamplesIterator>::KDNodeViewPtr
KDTree<IndicesIterator, SamplesIterator>::get_parent_node_after_sibling_traversal(
    SamplesIterator             query_feature_first,
    SamplesIterator             query_feature_last,
    const HyperRangeType&       kd_bounding_box,
    NearestNeighborsBufferType& nearest_neighbors_buffer,
    KDNodeViewPtr               kdnode) const {
    auto kdnode_parent = kdnode->parent_.lock();
    // if kdnode has a parent
    if (kdnode_parent) {
        // get the pivot sample index in the dataset
        const auto pivot_index = kdnode_parent->indices_iterator_pair_.first[0];
        // get the split value according to the current split dimension
        const auto pivot_split_value = samples_first_[pivot_index * n_features_ + kdnode_parent->cut_feature_index_];
        // get the value of the query according to the split dimension
        const auto query_split_value = query_feature_first[kdnode_parent->cut_feature_index_];
        // if the axiswise distance is equal to size of the bounding box divided by 2 along the current cut dimension
        // (the query is at the center of the bounding box), there could be a nearest neighbor to the other side of the
        // hyperrectangle since the values that are equal to the pivot are put to the right
        bool visit_sibling = kdnode->is_left_child() ? common::utils::abs(pivot_split_value - query_split_value) <=
                                                           (kd_bounding_box[kdnode_parent->cut_feature_index_].second -
                                                            kd_bounding_box[kdnode_parent->cut_feature_index_].first) /
                                                               2
                                                     : common::utils::abs(pivot_split_value - query_split_value) <
                                                           (kd_bounding_box[kdnode_parent->cut_feature_index_].second -
                                                            kd_bounding_box[kdnode_parent->cut_feature_index_].first) /
                                                               2;
        // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
        // closer to the query sample than the current nearest neighbor
        if (visit_sibling) {
            // if the sibling kdnode is not nullptr
            if (auto sibling_node = kdnode->get_sibling_node()) {
                // get the nearest neighbor from the sibling node
                inner_range_search_around_query_sample(
                    /**/ query_feature_first,
                    /**/ query_feature_last,
                    /**/ kd_bounding_box,
                    /**/ nearest_neighbors_buffer,
                    /**/ sibling_node);
            }
        }
    }
    // returns nullptr if kdnode doesnt have parent (or is root)
    return kdnode_parent;
}

template <typename IndicesIterator, typename SamplesIterator>
void KDTree<IndicesIterator, SamplesIterator>::serialize(const KDNodeViewPtr&                        kdnode,
                                                         rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
    writer.StartObject();
    {
        writer.String("axis");
        writer.Int64(kdnode->cut_feature_index_);

        writer.String("points");
        kdnode->serialize(writer, samples_first_, samples_last_, n_features_);

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

    std::ofstream output_file(filepath);

    rapidjson::OStreamWrapper output_stream_wrapper(output_file);

    rapidjson::Writer<rapidjson::OStreamWrapper> filewriter(output_stream_wrapper);

    document.Accept(filewriter);

    output_file.close();
}

}  // namespace ffcl::datastruct