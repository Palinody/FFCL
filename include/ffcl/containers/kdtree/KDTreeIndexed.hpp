#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDNodeIndexView.hpp"
#include "ffcl/containers/kdtree/policy/IndexedAxisSelectionPolicy.hpp"
#include "ffcl/containers/kdtree/policy/IndexedSplittingRulePolicy.hpp"

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

template <typename IndicesIterator, typename SamplesIterator>
class KDTreeIndexed {
  public:
    using DataType           = typename SamplesIterator::value_type;
    using KDNodeIndexViewPtr = std::shared_ptr<KDNodeIndexView<IndicesIterator, SamplesIterator>>;

    struct Options {
        Options()
          : bucket_size_{40}
          , max_depth_{common::utils::infinity<ssize_t>()}
          , axis_selection_policy_ptr_{std::make_unique<
                kdtree::policy::IndexedHighestVarianceBuild<IndicesIterator, SamplesIterator>>()}
          , splitting_rule_policy_ptr_{
                std::make_unique<kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>>()} {}

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

        template <typename IndexedAxisSelectionPolicy>
        Options& axis_selection_policy(const IndexedAxisSelectionPolicy& axis_selection_policy) {
            static_assert(std::is_base_of<kdtree::policy::IndexedAxisSelectionPolicy<IndicesIterator, SamplesIterator>,
                                          IndexedAxisSelectionPolicy>::value,
                          "The provided axis selection policy must be derived from "
                          "kdtree::policy::IndexedAxisSelectionPolicy<IndicesIterator, SamplesIterator>");

            axis_selection_policy_ptr_ = std::make_unique<IndexedAxisSelectionPolicy>(axis_selection_policy);
            return *this;
        }

        template <typename IndexedSplittingRulePolicy>
        Options& splitting_rule_policy(const IndexedSplittingRulePolicy& splitting_rule_policy) {
            static_assert(std::is_base_of<kdtree::policy::IndexedSplittingRulePolicy<IndicesIterator, SamplesIterator>,
                                          IndexedSplittingRulePolicy>::value,
                          "The provided splitting rule policy must be derived from "
                          "kdtree::policy::IndexedSplittingRulePolicy<IndicesIterator, SamplesIterator>");

            splitting_rule_policy_ptr_ = std::make_unique<IndexedSplittingRulePolicy>(splitting_rule_policy);
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
        std::shared_ptr<kdtree::policy::IndexedAxisSelectionPolicy<IndicesIterator, SamplesIterator>>
            axis_selection_policy_ptr_;
        // the policy that will be responsible for splitting the selected axis around a pivot sample
        std::shared_ptr<kdtree::policy::IndexedSplittingRulePolicy<IndicesIterator, SamplesIterator>>
            splitting_rule_policy_ptr_;
    };

  public:
    KDTreeIndexed(IndicesIterator index_first,
                  IndicesIterator index_last,
                  SamplesIterator samples_first,
                  SamplesIterator samples_last,
                  std::size_t     n_features,
                  const Options&  options = Options());

    KDTreeIndexed(const KDTreeIndexed&) = delete;

    ssize_t get_nearest_neighbor_index(std::size_t sample_index_query) const;

    void serialize(const KDNodeIndexViewPtr& kdnode, rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    void serialize(const fs::path& filepath) const;

  private:
    KDNodeIndexViewPtr build(IndicesIterator                     index_first,
                             IndicesIterator                     index_last,
                             SamplesIterator                     samples_first,
                             SamplesIterator                     samples_last,
                             ssize_t                             cut_feature_index,
                             ssize_t                             depth,
                             BoundingBoxKDType<SamplesIterator>& kd_bounding_box);

    ssize_t get_nearest_neighbor_index(std::size_t               sample_index_query,
                                       const KDNodeIndexViewPtr& kdnode,
                                       std::size_t               current_nearest_neighbor_index,
                                       DataType                  current_nearest_neighbor_distance) const;

    SamplesIterator samples_first_, samples_last_;
    std::size_t     n_features_;
    // bounding box hyper rectangle (w.r.t. each dimension)
    BoundingBoxKDType<SamplesIterator> kd_bounding_box_;

    Options options_;

    KDNodeIndexViewPtr root_;
};

template <typename IndicesIterator, typename SamplesIterator>
KDTreeIndexed<IndicesIterator, SamplesIterator>::KDTreeIndexed(IndicesIterator index_first,
                                                               IndicesIterator index_last,
                                                               SamplesIterator samples_first,
                                                               SamplesIterator samples_last,
                                                               std::size_t     n_features,
                                                               const Options&  options)
  : samples_first_{samples_first}
  , samples_last_{samples_last}
  , n_features_{n_features}
  , kd_bounding_box_{kdtree::algorithms::make_kd_bounding_box(index_first,
                                                              index_last,
                                                              samples_first,
                                                              samples_last,
                                                              n_features_)}
  , options_{options}
  , root_{build(index_first,
                index_last,
                samples_first,
                samples_last,
                (*options_.axis_selection_policy_ptr_)(index_first,
                                                       index_last,
                                                       samples_first,
                                                       samples_last,
                                                       n_features_,
                                                       0,
                                                       kd_bounding_box_),
                0,
                kd_bounding_box_)} {}

template <typename IndicesIterator, typename SamplesIterator>
std::shared_ptr<KDNodeIndexView<IndicesIterator, SamplesIterator>>
KDTreeIndexed<IndicesIterator, SamplesIterator>::build(IndicesIterator                     index_first,
                                                       IndicesIterator                     index_last,
                                                       SamplesIterator                     samples_first,
                                                       SamplesIterator                     samples_last,
                                                       ssize_t                             cut_feature_index,
                                                       ssize_t                             depth,
                                                       BoundingBoxKDType<SamplesIterator>& kd_bounding_box) {
    const std::size_t n_samples = std::distance(index_first, index_last);

    if (n_samples == 0) {
        return nullptr;
    }
    KDNodeIndexViewPtr kdnode;

    // the current kdnode is not leaf
    if (n_samples > options_.bucket_size_ && depth < options_.max_depth_) {
        // select the cut_feature_index according to the one with the most spread (min-max values)
        cut_feature_index = (*options_.axis_selection_policy_ptr_)(
            index_first, index_last, samples_first, samples_last, n_features_, depth, kd_bounding_box);

        // [cut_index, left_range, cut_range, right_range]
        auto [cut_index, left_index_range, cut_index_range, right_index_range] = (*options_.splitting_rule_policy_ptr_)(
            index_first, index_last, samples_first, samples_last, n_features_, cut_feature_index);

        kdnode = std::make_shared<KDNodeIndexView<IndicesIterator, SamplesIterator>>(
            cut_index_range,
            std::make_pair(samples_first, samples_last),
            n_features_,
            cut_feature_index,
            kd_bounding_box);

        const auto cut_value = samples_first[*cut_index_range.first * n_features_ + cut_feature_index];

        // set the right bound of the left child to the cut value
        kd_bounding_box[cut_feature_index].second = cut_value;

        kdnode->left_ = build(left_index_range.first,
                              left_index_range.second,
                              samples_first,
                              samples_last,
                              cut_feature_index,
                              depth + 1,
                              kd_bounding_box);

        // reset the right bound of the bounding box to the current kdnode right bound
        kd_bounding_box[cut_feature_index].second = kdnode->kd_bounding_box_[cut_feature_index].second;

        if (n_samples > 2) {
            // set the left bound of the right child to the cut value
            kd_bounding_box[cut_feature_index].first = cut_value;

            kdnode->right_ = build(right_index_range.first,
                                   right_index_range.second,
                                   samples_first,
                                   samples_last,
                                   cut_feature_index,
                                   depth + 1,
                                   kd_bounding_box);

            // reset the left bound of the bounding box to the current kdnode left bound
            kd_bounding_box[cut_feature_index].first = kdnode->kd_bounding_box_[cut_feature_index].first;
        }
    } else {
        kdnode = std::make_shared<KDNodeIndexView<IndicesIterator, SamplesIterator>>(
            std::make_pair(index_first, index_last),
            std::make_pair(samples_first, samples_last),
            n_features_,
            kd_bounding_box);
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
ssize_t KDTreeIndexed<IndicesIterator, SamplesIterator>::get_nearest_neighbor_index(
    std::size_t sample_index_query) const {
    const std::size_t current_nearest_neighbor_index = *root_->indices_iterator_pair_.first;
    const DataType    current_nearest_neighbor_distance =
        math::heuristics::auto_distance(samples_first_ + sample_index_query * n_features_,
                                        samples_first_ + sample_index_query * n_features_ + n_features_,
                                        samples_first_ + current_nearest_neighbor_index * n_features_);

    printf("Current nearest index: %ld, distance: %.3f\n",
           current_nearest_neighbor_index,
           current_nearest_neighbor_distance);

    return get_nearest_neighbor_index(
        sample_index_query, root_, current_nearest_neighbor_index, current_nearest_neighbor_distance);
}

template <typename IndicesIterator, typename SamplesIterator>
ssize_t KDTreeIndexed<IndicesIterator, SamplesIterator>::get_nearest_neighbor_index(
    std::size_t               sample_index_query,
    const KDNodeIndexViewPtr& kdnode,
    std::size_t               current_nearest_neighbor_index,
    DataType                  current_nearest_neighbor_distance) const {
    // draft function
    return 0;
    if (!kdnode->is_leaf()) {
        const std::size_t candidate_nearest_neighbor_index = *root_->indices_iterator_pair_.first;
        const DataType    candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(samples_first_ + sample_index_query * n_features_,
                                            samples_first_ + sample_index_query * n_features_ + n_features_,
                                            samples_first_ + candidate_nearest_neighbor_index * n_features_);

        if (candidate_nearest_neighbor_distance < current_nearest_neighbor_distance) {
            current_nearest_neighbor_index    = candidate_nearest_neighbor_index;
            current_nearest_neighbor_distance = candidate_nearest_neighbor_distance;
        }
    } else {
        // use ffcl/math/heuristics/NearestNeighbor.hpp for sequences
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void KDTreeIndexed<IndicesIterator, SamplesIterator>::serialize(
    const KDNodeIndexViewPtr&                   kdnode,
    rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
    writer.StartObject();
    {
        writer.String("axis");
        writer.Int64(kdnode->cut_feature_index_);

        writer.String("points");
        kdnode->serialize(writer);

        // continue the recursion if the current node is not leaf
        if (!kdnode->is_leaf()) {
            writer.String("left");
            serialize(kdnode->left_, writer);

            // The right pointer might be nullptr when a node had 2 samples. The median computation chooses the
            // second sample as the pivot because the median of 2 samples will output index 1. The other index will
            // be 0 and thus the left child
            if (kdnode->right_) {
                writer.String("right");
                serialize(kdnode->right_, writer);
            }
        }
    }
    writer.EndObject();
}

template <typename IndicesIterator, typename SamplesIterator>
void KDTreeIndexed<IndicesIterator, SamplesIterator>::serialize(const fs::path& filepath) const {
    static_assert(std::is_floating_point_v<DataType> || std::is_integral_v<DataType>,
                  "Unsupported type during kdtree serialization");

    rapidjson::Document                        document;
    rapidjson::StringBuffer                    buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

    writer.StartObject();
    {
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