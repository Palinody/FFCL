#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/tree/kdtree/KDNodeView.hpp"
#include "ffcl/datastruct/tree/kdtree/policy/AxisSelectionPolicy.hpp"
#include "ffcl/datastruct/tree/kdtree/policy/SplittingRulePolicy.hpp"

#include "ffcl/common/math/random/Distributions.hpp"

#include <algorithm>
#include <array>
#include <cstddef>  // std::size_t
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
    using IndicesIteratorType = IndicesIterator;
    using SamplesIteratorType = SamplesIterator;

    static_assert(common::is_iterator<IndicesIteratorType>::value, "IndicesIteratorType is not an iterator");
    static_assert(common::is_iterator<SamplesIteratorType>::value, "SamplesIteratorType is not an iterator");

    using IndexType = typename std::iterator_traits<IndicesIterator>::value_type;
    using DataType  = typename std::iterator_traits<SamplesIterator>::value_type;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    using NodeType = typename KDNodeView<IndicesIterator, DataType>::NodeType;
    using NodePtr  = typename KDNodeView<IndicesIterator, DataType>::NodePtr;

    using HyperIntervalType = HyperInterval<DataType>;

    struct Options {
        Options()
          : bucket_size_{40}
          , max_depth_{common::infinity<std::size_t>()}
          , axis_selection_policy_ptr_{std::make_shared<
                kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>>()}
          , splitting_rule_policy_ptr_{
                std::make_shared<kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>>()} {}

        Options(const Options&) = default;

        Options& operator=(const Options&) = default;

        Options& bucket_size(std::size_t bucket_size) {
            bucket_size_ = bucket_size;
            return *this;
        }

        Options& max_depth(std::size_t max_depth) {
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
        std::size_t max_depth_;
        // the policy that will be responsible for selecting the axis at a given kdtree depth
        std::shared_ptr<kdtree::policy::AxisSelectionPolicy<IndicesIterator, SamplesIterator>>
            axis_selection_policy_ptr_;
        // the policy that will be responsible for splitting the selected axis around a pivot sample
        std::shared_ptr<kdtree::policy::SplittingRulePolicy<IndicesIterator, SamplesIterator>>
            splitting_rule_policy_ptr_;
    };

  public:
    explicit KDTree(IndicesIterator indices_range_first,
                    IndicesIterator indices_range_last,
                    SamplesIterator samples_range_first,
                    SamplesIterator samples_range_last,
                    std::size_t     n_features,
                    const Options&  options = Options());

    KDTree(const KDTree& other);

    KDTree(KDTree&& other) noexcept;

    std::size_t n_samples() const;

    std::size_t n_features() const;

    constexpr auto begin() const;

    constexpr auto end() const;

    constexpr auto cbegin() const;

    constexpr auto cend() const;

    constexpr auto root() const;

    constexpr auto operator[](std::size_t sample_index) const;

    constexpr auto features_range_first(std::size_t sample_index) const;

    constexpr auto features_range_last(std::size_t sample_index) const;

    // serialization

    void serialize(const NodePtr& kdnode, rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    void serialize(const fs::path& filepath) const;

  private:
    NodePtr build(const IndicesIterator& indices_range_first,
                  const IndicesIterator& indices_range_last,
                  std::size_t            feature_cut_index,
                  std::size_t            depth,
                  HyperIntervalType&     hyper_interval);

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
    HyperIntervalType hyper_interval_;
    // The root node of the indexing structure.
    NodePtr root_;
};

//  Class Template Argument Deduction (CTAD) guide
template <typename IndicesIterator, typename SamplesIterator>
KDTree(IndicesIterator,
       IndicesIterator,
       SamplesIterator,
       SamplesIterator,
       std::size_t,
       const typename KDTree<IndicesIterator, SamplesIterator>::Options&) -> KDTree<IndicesIterator, SamplesIterator>;

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
  , hyper_interval_{make_hyper_interval(indices_range_first,
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
                                                       hyper_interval_),
                0,
                hyper_interval_)} {}

template <typename IndicesIterator, typename SamplesIterator>
KDTree<IndicesIterator, SamplesIterator>::KDTree(const KDTree& other)
  : options_{other.options_}
  , samples_range_first_{other.samples_range_first_}
  , samples_range_last_{other.samples_range_last_}
  , n_features_{other.n_features_}
  , hyper_interval_{other.hyper_interval_}
  , root_{other.root_} {}

template <typename IndicesIterator, typename SamplesIterator>
KDTree<IndicesIterator, SamplesIterator>::KDTree(KDTree&& other) noexcept
  : options_{std::move(other.options_)}
  , samples_range_first_{std::move(other.samples_range_first_)}
  , samples_range_last_{std::move(other.samples_range_last_)}
  , n_features_{std::move(other.n_features_)}
  , hyper_interval_{std::move(other.hyper_interval_)}
  , root_{std::move(other.root_)} {}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t KDTree<IndicesIterator, SamplesIterator>::n_samples() const {
    return common::get_n_samples(samples_range_first_, samples_range_last_, n_features_);
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t KDTree<IndicesIterator, SamplesIterator>::n_features() const {
    return n_features_;
}

template <typename IndicesIterator, typename SamplesIterator>
constexpr auto KDTree<IndicesIterator, SamplesIterator>::begin() const {
    return samples_range_first_;
}

template <typename IndicesIterator, typename SamplesIterator>
constexpr auto KDTree<IndicesIterator, SamplesIterator>::end() const {
    return samples_range_last_;
}

template <typename IndicesIterator, typename SamplesIterator>
constexpr auto KDTree<IndicesIterator, SamplesIterator>::cbegin() const {
    return samples_range_first_;
}

template <typename IndicesIterator, typename SamplesIterator>
constexpr auto KDTree<IndicesIterator, SamplesIterator>::cend() const {
    return samples_range_last_;
}

template <typename IndicesIterator, typename SamplesIterator>
constexpr auto KDTree<IndicesIterator, SamplesIterator>::root() const {
    return root_;
}

template <typename IndicesIterator, typename SamplesIterator>
constexpr auto KDTree<IndicesIterator, SamplesIterator>::operator[](std::size_t sample_index) const {
    return features_range_first(sample_index);
}

template <typename IndicesIterator, typename SamplesIterator>
constexpr auto KDTree<IndicesIterator, SamplesIterator>::features_range_first(std::size_t sample_index) const {
    return samples_range_first_ + sample_index * n_features_;
}

template <typename IndicesIterator, typename SamplesIterator>
constexpr auto KDTree<IndicesIterator, SamplesIterator>::features_range_last(std::size_t sample_index) const {
    return samples_range_first_ + sample_index * n_features_ + n_features_;
}

template <typename IndicesIterator, typename SamplesIterator>
typename KDTree<IndicesIterator, SamplesIterator>::NodePtr KDTree<IndicesIterator, SamplesIterator>::build(
    const IndicesIterator& indices_range_first,
    const IndicesIterator& indices_range_last,
    std::size_t            feature_cut_index,
    std::size_t            depth,
    HyperIntervalType&     hyper_interval) {
    NodePtr kdnode;
    // number of samples in the current node
    const std::size_t n_node_samples = std::distance(indices_range_first, indices_range_last);
    // if the current number of samples is greater than the target bucket size, the node is not leaf
    if (depth < options_.max_depth_ && n_node_samples > options_.bucket_size_) {
        // select the feature_cut_index according to the one with the most spread (min-max values)
        feature_cut_index = (*options_.axis_selection_policy_ptr_)(
            /**/ indices_range_first,
            /**/ indices_range_last,
            /**/ samples_range_first_,
            /**/ samples_range_last_,
            /**/ n_features_,
            /**/ depth,
            /**/ hyper_interval);

        // left_partition_range, middle_partition_range, right_partition_range
        auto [cut_index, left_indices_range, cut_indices_range, right_indices_range] =
            (*options_.splitting_rule_policy_ptr_)(
                /**/ indices_range_first,
                /**/ indices_range_last,
                /**/ samples_range_first_,
                /**/ samples_range_last_,
                /**/ n_features_,
                /**/ feature_cut_index);

        common::ignore_parameters(cut_index);

        const auto feature_cut_value =
            samples_range_first_[cut_indices_range.first[0] * n_features_ + feature_cut_index];

        // make the current node
        {
            kdnode = std::make_shared<NodeType>(/**/ cut_indices_range,
                                                /**/ feature_cut_index,
                                                /**/ hyper_interval[feature_cut_index]);
        }
        // make the left node
        {
            // set the right bound of the left child to the cut value
            hyper_interval[feature_cut_index].second() = feature_cut_value;

            kdnode->left_ = build(/**/ left_indices_range.first,
                                  /**/ left_indices_range.second,
                                  /**/ feature_cut_index,
                                  /**/ depth + 1,
                                  /**/ hyper_interval);
            // provide a parent pointer to the child node for reversed traversal of the kdtree
            kdnode->left_->parent_ = kdnode;

            // reset the right bound of the bounding box to the current kdnode right bound
            hyper_interval[feature_cut_index].second() = kdnode->axis_interval_.second();
        }
        // make the right node
        {
            // set the left bound of the right child to the cut value
            hyper_interval[feature_cut_index].first() = feature_cut_value;

            kdnode->right_ = build(/**/ right_indices_range.first,
                                   /**/ right_indices_range.second,
                                   /**/ feature_cut_index,
                                   /**/ depth + 1,
                                   /**/ hyper_interval);
            // provide a parent pointer to the child node for reversed traversal of the kdtree
            kdnode->right_->parent_ = kdnode;

            // reset the left bound of the bounding box to the current kdnode left bound
            hyper_interval[feature_cut_index].first() = kdnode->axis_interval_.first();
        }
    } else {
        kdnode = std::make_shared<NodeType>(std::make_pair(indices_range_first, indices_range_last),
                                            hyper_interval[feature_cut_index]);
    }
    return kdnode;
}

template <typename IndicesIterator, typename SamplesIterator>
void KDTree<IndicesIterator, SamplesIterator>::serialize(const NodePtr&                              kdnode,
                                                         rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
    writer.StartObject();
    {
        writer.String("axis");
        writer.Int64(kdnode->cut_axis_feature_index_);

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
                writer.Int64(hyper_interval_[feature_index].first);
                writer.Int64(hyper_interval_[feature_index].second);

            } else if constexpr (std::is_floating_point_v<DataType>) {
                writer.Double(hyper_interval_[feature_index].first);
                writer.Double(hyper_interval_[feature_index].second);
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