#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDNodeView.hpp"
#include "ffcl/containers/kdtree/KDTreeUtils.hpp"
#include "ffcl/containers/kdtree/policy/AxisSelectionPolicy.hpp"
#include "ffcl/containers/kdtree/policy/SplittingRulePolicy.hpp"

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

template <typename Iterator>
class KDTree {
  public:
    struct Options {
        Options() = default;

        Options(const Options&) = default;

        Options& bucket_size(std::size_t bucket_size) {
            bucket_size_ = bucket_size;
            return *this;
        }

        Options& max_depth(ssize_t max_depth) {
            max_depth_ = max_depth;
            return *this;
        }

        Options& operator=(const Options& options) {
            bucket_size_ = options.bucket_size_;
            max_depth_   = options.max_depth_;
            return *this;
        }

        void serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
            writer.String("options");

            writer.StartObject();

            writer.String("bucket_size");
            writer.Int64(bucket_size_);

            writer.String("max_depth");
            writer.Int64(max_depth_);

            writer.EndObject();
        }
        // the maximum number of samples per leaf node
        std::size_t bucket_size_ = 40;
        // the maximum recursion depth. Defaults to infinity
        ssize_t max_depth_ = common::utils::infinity<ssize_t>();
    };

  public:
    template <typename AxisSelectionPolicyFunction, typename SplittingRulePolicyFunction>
    KDTree(const IteratorPairType<Iterator>&  iterator_pair,
           std::size_t                        n_features,
           const AxisSelectionPolicyFunction& axis_selection_policy,
           const SplittingRulePolicyFunction& splitting_rule_policy,
           const Options&                     options = Options());

    template <typename AxisSelectionPolicyFunction>
    KDTree(const IteratorPairType<Iterator>&  iterator_pair,
           std::size_t                        n_features,
           const AxisSelectionPolicyFunction& axis_selection_policy,
           const Options&                     options = Options());

    KDTree(const IteratorPairType<Iterator>& iterator_pair, std::size_t n_features, const Options& options = Options());

    KDTree(const KDTree&) = delete;

    void serialize_kdtree(const std::shared_ptr<KDNodeView<Iterator>>& kdnode,
                          rapidjson::Writer<rapidjson::StringBuffer>&  writer) const;

    void serialize(const fs::path& filepath) const;

  private:
    template <typename AxisSelectionPolicyFunction, typename SplittingRulePolicyFunction>
    std::shared_ptr<KDNodeView<Iterator>> build(const IteratorPairType<Iterator>&  iterator_pair,
                                                ssize_t                            cut_feature_index,
                                                ssize_t                            depth,
                                                BoundingBoxKDType<Iterator>&       kd_bounding_box,
                                                const AxisSelectionPolicyFunction& axis_selection_policy,
                                                const SplittingRulePolicyFunction& splitting_rule_policy);

    IteratorPairType<Iterator> iterator_pair_;

    std::size_t n_features_;
    // bounding box hyper rectangle (w.r.t. each dimension)
    BoundingBoxKDType<Iterator> kd_bounding_box_;

    Options options_;

    std::shared_ptr<KDNodeView<Iterator>> root_;
};

template <typename Iterator>
template <typename AxisSelectionPolicyFunction, typename SplittingRulePolicyFunction>
KDTree<Iterator>::KDTree(const IteratorPairType<Iterator>&  iterator_pair,
                         std::size_t                        n_features,
                         const AxisSelectionPolicyFunction& axis_selection_policy,
                         const SplittingRulePolicyFunction& splitting_rule_policy,
                         const Options&                     options)
  : iterator_pair_{iterator_pair}
  , n_features_{n_features}
  , kd_bounding_box_{kdtree::utils::make_kd_bounding_box(std::get<0>(iterator_pair_),
                                                         std::get<1>(iterator_pair_),
                                                         n_features_)}
  , options_{options}
  , root_{build(iterator_pair_,
                axis_selection_policy(iterator_pair_, n_features_, 0, kd_bounding_box_),
                0,
                kd_bounding_box_,
                axis_selection_policy,
                splitting_rule_policy)} {}

template <typename Iterator>
template <typename AxisSelectionPolicyFunction>
KDTree<Iterator>::KDTree(const IteratorPairType<Iterator>&  iterator_pair,
                         std::size_t                        n_features,
                         const AxisSelectionPolicyFunction& axis_selection_policy,
                         const Options&                     options)
  : KDTree<Iterator>::KDTree(iterator_pair,
                             n_features,
                             axis_selection_policy,
                             kdtree::policy::QuickselectMedianRange<Iterator>(),
                             options) {}

template <typename Iterator>
KDTree<Iterator>::KDTree(const IteratorPairType<Iterator>& iterator_pair,
                         std::size_t                       n_features,
                         const Options&                    options)
  : KDTree<Iterator>::KDTree(iterator_pair,
                             n_features,
                             kdtree::policy::MaximumSpreadBuild<Iterator>(),
                             kdtree::policy::QuickselectMedianRange<Iterator>(),
                             options) {}

template <typename Iterator>
template <typename AxisSelectionPolicyFunction, typename SplittingRulePolicyFunction>
std::shared_ptr<KDNodeView<Iterator>> KDTree<Iterator>::build(
    const IteratorPairType<Iterator>&  iterator_pair,
    ssize_t                            cut_feature_index,
    ssize_t                            depth,
    BoundingBoxKDType<Iterator>&       kd_bounding_box,
    const AxisSelectionPolicyFunction& axis_selection_policy,
    const SplittingRulePolicyFunction& splitting_rule_policy) {
    const std::size_t n_samples = common::utils::get_n_samples(iterator_pair.first, iterator_pair.second, n_features_);

    if (n_samples == 0) {
        return nullptr;
    }
    std::shared_ptr<KDNodeView<Iterator>> node;

    // the current node is not leaf
    if (n_samples > options_.bucket_size_ && depth < options_.max_depth_) {
        // select the cut_feature_index according to the one with the most spread (min-max values)
        cut_feature_index = axis_selection_policy(iterator_pair, n_features_, depth, kd_bounding_box);

        auto [cut_index, left_range, cut_range, right_range] =
            splitting_rule_policy(iterator_pair, n_features_, cut_feature_index);

        node = std::make_shared<KDNodeView<Iterator>>(cut_range, n_features_, cut_feature_index, kd_bounding_box);

        const auto cut_value = *(cut_range.first + cut_feature_index);

        // set the right bound of the left child to the cut value
        kd_bounding_box[cut_feature_index].second = cut_value;

        node->left_ = build(
            left_range, cut_feature_index, depth + 1, kd_bounding_box, axis_selection_policy, splitting_rule_policy);

        // reset the right bound of the bounding box to the current node right bound
        kd_bounding_box[cut_feature_index].second = node->kd_bounding_box_[cut_feature_index].second;

        if (n_samples > 2) {
            // set the left bound of the right child to the cut value
            kd_bounding_box[cut_feature_index].first = cut_value;

            node->right_ = build(right_range,
                                 cut_feature_index,
                                 depth + 1,
                                 kd_bounding_box,
                                 axis_selection_policy,
                                 splitting_rule_policy);

            // reset the left bound of the bounding box to the current node left bound
            kd_bounding_box[cut_feature_index].first = node->kd_bounding_box_[cut_feature_index].first;
        }
    } else {
        node = std::make_shared<KDNodeView<Iterator>>(iterator_pair, n_features_, kd_bounding_box);
    }
    return node;
}

template <typename Iterator>
void KDTree<Iterator>::serialize_kdtree(const std::shared_ptr<KDNodeView<Iterator>>& kdnode,
                                        rapidjson::Writer<rapidjson::StringBuffer>&  writer) const {
    writer.StartObject();
    {
        writer.String("axis");
        writer.Int64(kdnode->cut_feature_index_);

        writer.String("points");
        kdnode->serialize(writer);

        // continue the recursion if the current node is not leaf
        if (!kdnode->is_leaf()) {
            writer.String("left");
            serialize_kdtree(kdnode->left_, writer);

            // The right pointer might be nullptr when a node had 2 samples. The median computation chooses the
            // second sample as the pivot because the median of 2 samples will output index 1. The other index will
            // be 0 and thus the left child
            if (kdnode->right_) {
                writer.String("right");
                serialize_kdtree(kdnode->right_, writer);
            }
        }
    }
    writer.EndObject();
}

template <typename Iterator>
void KDTree<Iterator>::serialize(const fs::path& filepath) const {
    using DataType = DataType<Iterator>;

    static_assert(std::is_floating_point_v<DataType> || std::is_integral_v<DataType>,
                  "Unsupported type during kdtree serialization");

    rapidjson::Document                        document;
    rapidjson::StringBuffer                    buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

    writer.StartObject();
    {
        writer.String("n_samples");
        writer.Int64(
            common::utils::get_n_samples(std::get<0>(iterator_pair_), std::get<1>(iterator_pair_), n_features_));

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
        serialize_kdtree(root_, writer);
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