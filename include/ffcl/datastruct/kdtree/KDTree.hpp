#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/kdtree/KDNodeView.hpp"
#include "ffcl/datastruct/kdtree/policy/AxisSelectionPolicy.hpp"
#include "ffcl/datastruct/kdtree/policy/SplittingRulePolicy.hpp"

#include "ffcl/common/math/random/Distributions.hpp"

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
    using IndexType = typename std::iterator_traits<IndicesIterator>::value_type;
    using DataType  = typename std::iterator_traits<SamplesIterator>::value_type;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    using IndicesIteratorType = IndicesIterator;
    using SamplesIteratorType = SamplesIterator;

    static_assert(common::is_iterator<IndicesIteratorType>::value, "IndicesIteratorType is not an iterator");
    static_assert(common::is_iterator<SamplesIteratorType>::value, "SamplesIteratorType is not an iterator");

    using KDNodeViewType = typename KDNodeView<IndicesIterator, SamplesIterator>::KDNodeViewType;
    using KDNodeViewPtr  = typename KDNodeView<IndicesIterator, SamplesIterator>::KDNodeViewPtr;

    using HyperRangeType = bbox::HyperRangeType<SamplesIterator>;

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
            static_assert(std::is_base_of_v<kdtree::policy::AxisSelectionPolicy<IndicesIterator, SamplesIterator>,
                                            AxisSelectionPolicy>,
                          "The provided axis selection policy must be derived from "
                          "kdtree::policy::AxisSelectionPolicy<IndicesIterator, SamplesIterator>");

            axis_selection_policy_ptr_ = std::make_shared<AxisSelectionPolicy>(axis_selection_policy);
            return *this;
        }

        template <typename SplittingRulePolicy>
        Options& splitting_rule_policy(const SplittingRulePolicy& splitting_rule_policy) {
            static_assert(std::is_base_of_v<kdtree::policy::SplittingRulePolicy<IndicesIterator, SamplesIterator>,
                                            SplittingRulePolicy>,
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

    // KDTree(const KDTree&) = delete;

    std::size_t n_samples() const;

    std::size_t n_features() const;

    constexpr auto begin() const {
        return samples_range_first_;
    }

    constexpr auto end() const {
        return samples_range_last_;
    }

    constexpr auto cbegin() const {
        return samples_range_first_;
    }

    constexpr auto cend() const {
        return samples_range_last_;
    }

    constexpr auto root() const {
        return root_;
    }

    constexpr auto operator[](std::size_t sample_index) const {
        return features_range_first(sample_index);
    }

    constexpr auto features_range_first(std::size_t sample_index) const {
        return samples_range_first_ + sample_index * n_features_;
    }

    constexpr auto features_range_last(std::size_t sample_index) const {
        return samples_range_first_ + sample_index * n_features_ + n_features_;
    }

    // serialization

    void serialize(const KDNodeViewPtr& kdnode, rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    void serialize(const fs::path& filepath) const;

  private:
    KDNodeViewPtr build(const IndicesIterator& indices_range_first,
                        const IndicesIterator& indices_range_last,
                        ssize_t                cut_feature_index,
                        ssize_t                depth,
                        HyperRangeType&        kd_bounding_box);

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
    const IndicesIterator& indices_range_first,
    const IndicesIterator& indices_range_last,
    ssize_t                cut_feature_index,
    ssize_t                depth,
    HyperRangeType&        kd_bounding_box) {
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

        // left_partition_range, middle_partition_range, right_partition_range
        auto [cut_index, left_indices_range, cut_indices_range, right_indices_range] =
            (*options_.splitting_rule_policy_ptr_)(
                /**/ indices_range_first,
                /**/ indices_range_last,
                /**/ samples_range_first_,
                /**/ samples_range_last_,
                /**/ n_features_,
                /**/ cut_feature_index);

        common::ignore_parameters(cut_index);

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
            kd_bounding_box[cut_feature_index].second = kdnode->cut_feature_range_.second;
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
            kd_bounding_box[cut_feature_index].first = kdnode->cut_feature_range_.first;
        }
    } else {
        kdnode = std::make_shared<KDNodeViewType>(std::make_pair(indices_range_first, indices_range_last),
                                                  kd_bounding_box[cut_feature_index]);
    }
    return kdnode;
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