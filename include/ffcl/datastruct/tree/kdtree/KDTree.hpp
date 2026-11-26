#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/tree/kdtree/KDNodeView.hpp"

#include "ffcl/datastruct/bounds/StaticBoundWithCentroid.hpp"

#include "ffcl/datastruct/bounds/AABBWithCentroid.hpp"
#include "ffcl/datastruct/bounds/segment/LowerBoundAndUpperBound.hpp"

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

#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>

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

    using BoundType = bounds::AABBWithCentroid<bounds::segment::LowerBoundAndUpperBound<DataType>>;

    static_assert(common::is_crtp_of<BoundType, bounds::StaticBoundWithCentroid>::value,
                  "Provided a BoundType that does not inherit from StaticBoundWithCentroid<Derived>");

    using NodeType = typename KDNodeView<IndicesIterator, BoundType>::NodeType;
    using NodePtr  = typename KDNodeView<IndicesIterator, BoundType>::NodePtr;

    struct Options {
        Options()
          : bucket_size_{40}  // `infinity` is 0 for integer types, which used to make a default-constructed tree a
                              // single leaf and every query a brute-force scan. Keep `infinity` itself as the sentinel
                              // other code expects and use the largest depth here instead.
          , max_depth_{std::numeric_limits<std::size_t>::max()}
          , axis_selection_policy_ptr_{std::make_shared<
                kdtree::policy::HighestVarianceBuild<IndicesIteratorType, SamplesIteratorType, BoundType>>()}
          , splitting_rule_policy_ptr_{
                std::make_shared<kdtree::policy::QuickselectMedianRange<IndicesIteratorType, SamplesIteratorType>>()} {}

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
            static_assert(
                std::is_base_of_v<kdtree::policy::AxisSelectionPolicy<IndicesIteratorType, SamplesIteratorType>,
                                  AxisSelectionPolicy>,
                "The provided axis selection policy must be derived from "
                "kdtree::policy::AxisSelectionPolicy<IndicesIteratorType, SamplesIteratorType>");

            axis_selection_policy_ptr_ = std::make_shared<AxisSelectionPolicy>(axis_selection_policy);
            return *this;
        }

        template <typename SplittingRulePolicy>
        Options& splitting_rule_policy(const SplittingRulePolicy& splitting_rule_policy) {
            static_assert(
                std::is_base_of_v<kdtree::policy::SplittingRulePolicy<IndicesIteratorType, SamplesIteratorType>,
                                  SplittingRulePolicy>,
                "The provided splitting rule policy must be derived from "
                "kdtree::policy::SplittingRulePolicy<IndicesIteratorType, SamplesIteratorType>");

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
        std::shared_ptr<kdtree::policy::AxisSelectionPolicy<IndicesIteratorType, SamplesIteratorType, BoundType>>
            axis_selection_policy_ptr_;
        // the policy that will be responsible for splitting the selected axis around a pivot sample
        std::shared_ptr<kdtree::policy::SplittingRulePolicy<IndicesIteratorType, SamplesIteratorType>>
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

    std::size_t n_nodes() const;

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
    // Recompute every node's bound from its subtree's samples, children before parents.
    void tighten_bounds(NodePtr kdnode);

    NodePtr build(const IndicesIterator& indices_range_first,
                  const IndicesIterator& indices_range_last,
                  std::size_t            feature_cut_index,
                  std::size_t            depth,
                  BoundType&             bound);

    // Options used to configure the indexing structure.
    Options options_;
    // Iterator pointing to the first element of the dataset.
    SamplesIteratorType samples_range_first_;
    // Iterator pointing to the last element of the dataset.
    SamplesIteratorType samples_range_last_;
    // The number of samples deduced from the indices range span.
    std::size_t n_samples_;
    // The number of features in the dataset, used to represent data as a vectorized 2D array.
    std::size_t n_features_;
    // Number of nodes of the tree, used to size the flat per-node caches of the search buffers.
    std::size_t n_nodes_;
    // Every node of the tree, in one container whose addresses stay valid as it grows. Shared so that
    // copying a tree keeps the same nodes, the way copies used to share the root.
    std::shared_ptr<std::deque<NodeType>> nodes_;
    // A hyperrectangle (bounding box) specifying the value bounds of the subset of data represented by the index array
    // from the entire dataset. This hyperrectangle is defined with respect to each dimension.
    BoundType bound_;
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
  , n_samples_{static_cast<std::size_t>(std::distance(indices_range_first, indices_range_last))}
  , n_features_{n_features}
  , n_nodes_{0}
  , nodes_{std::make_shared<std::deque<NodeType>>()}
  , bound_{make_tight_bound<IndicesIteratorType, SamplesIteratorType, BoundType>(indices_range_first,
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
                                                       bound_),
                0,
                bound_)} {
    // Replace every node's cell by the bounding box of its own samples. The cell is what the split produced
    // and it is loose: it spans the parent's extent on every axis but the cut one. The tight box is a valid
    // (it still contains the subtree's samples) and strictly smaller bound, which strengthens the box
    // distance used to prune node pairs in the dual-tree traversals and the radius visits of DBSCAN.
    tighten_bounds(root_);
    bound_ = root_->bound_;
}

template <typename IndicesIterator, typename SamplesIterator>
KDTree<IndicesIterator, SamplesIterator>::KDTree(const KDTree& other)
  : options_{other.options_}
  , samples_range_first_{other.samples_range_first_}
  , samples_range_last_{other.samples_range_last_}
  , n_samples_{other.n_samples_}
  , n_features_{other.n_features_}
  , n_nodes_{other.n_nodes_}
  , nodes_{other.nodes_}
  , bound_{other.bound_}
  , root_{other.root_} {}

template <typename IndicesIterator, typename SamplesIterator>
KDTree<IndicesIterator, SamplesIterator>::KDTree(KDTree&& other) noexcept
  : options_{std::move(other.options_)}
  , samples_range_first_{std::move(other.samples_range_first_)}
  , samples_range_last_{std::move(other.samples_range_last_)}
  , n_samples_{std::move(other.n_samples_)}
  , n_features_{std::move(other.n_features_)}
  , n_nodes_{std::move(other.n_nodes_)}
  , nodes_{std::move(other.nodes_)}
  , bound_{std::move(other.bound_)}
  , root_{std::move(other.root_)} {}

template <typename IndicesIterator, typename SamplesIterator>
void KDTree<IndicesIterator, SamplesIterator>::tighten_bounds(NodePtr kdnode) {
    if (!kdnode) {
        return;
    }
    if (kdnode->is_leaf()) {
        kdnode->bound_ =
            make_tight_bound<IndicesIteratorType, SamplesIteratorType, BoundType>(kdnode->indices_range_.first,
                                                                                  kdnode->indices_range_.second,
                                                                                  samples_range_first_,
                                                                                  samples_range_last_,
                                                                                  n_features_);
    } else {
        tighten_bounds(kdnode->left_);
        tighten_bounds(kdnode->right_);

        using SegmentType = typename BoundType::SegmentType;

        auto segments = std::vector<SegmentType>{};
        segments.reserve(n_features_);

        for (std::size_t feature_index = 0; feature_index < n_features_; ++feature_index) {
            const auto& left_segment  = kdnode->left_->bound_.segment_at(feature_index);
            const auto& right_segment = kdnode->right_->bound_.segment_at(feature_index);

            segments.emplace_back(std::min(left_segment.lower_bound(), right_segment.lower_bound()),
                                  std::max(left_segment.upper_bound(), right_segment.upper_bound()));
        }
        kdnode->bound_ = BoundType{std::move(segments)};
    }
    kdnode->diameter_ = kdnode->bound_.diameter();
}

template <typename IndicesIterator, typename SamplesIterator>
std::size_t KDTree<IndicesIterator, SamplesIterator>::n_nodes() const {
    return n_nodes_;
}

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
    BoundType&             bound) {
    NodePtr kdnode;
    // number of samples in the current node
    const std::size_t n_node_samples = std::distance(indices_range_first, indices_range_last);
    // if the current number of samples is greater than the target bucket size, the node is not leaf
    if (depth < options_.max_depth_ && n_node_samples > options_.bucket_size_) {
        // select the feature_cut_index according to the one with the most spread (min-max values)
        feature_cut_index = (*options_.axis_selection_policy_ptr_)(indices_range_first,
                                                                   indices_range_last,
                                                                   samples_range_first_,
                                                                   samples_range_last_,
                                                                   n_features_,
                                                                   depth,
                                                                   bound);

        // left_partition_range, middle_partition_range, right_partition_range
        auto [cut_index, left_indices_range, cut_indices_range, right_indices_range] =
            (*options_.splitting_rule_policy_ptr_)(indices_range_first,
                                                   indices_range_last,
                                                   samples_range_first_,
                                                   samples_range_last_,
                                                   n_features_,
                                                   feature_cut_index);

        common::ignore_parameters(cut_index);

        const auto feature_cut_value =
            samples_range_first_[cut_indices_range.first[0] * n_features_ + feature_cut_index];

        // make the current node. Its pivot sample is owned by the left child (so that the leaves of the
        // tree cover every sample exactly once) and only its index is kept here for the query routing.
        {
            kdnode = &nodes_->emplace_back(std::make_pair(indices_range_first, indices_range_first),
                                           feature_cut_index,
                                           bound);
            kdnode->set_pivot_index(static_cast<std::size_t>(cut_indices_range.first[0]));
            kdnode->set_node_id(n_nodes_++);
        }
        // make the left node
        {
            // set the right bound of the left child to the cut value
            bound.segment_at(feature_cut_index).update_upper_bound(feature_cut_value);

            kdnode->left_ =
                build(left_indices_range.first, left_indices_range.second, feature_cut_index, depth + 1, bound);
            // provide a parent pointer to the child node for reversed traversal of the kdtree
            kdnode->left_->parent_ = kdnode;

            // reset the right bound of the bounding box to the current kdnode right bound
            bound.segment_at(feature_cut_index)
                .update_upper_bound(kdnode->bound_.segment_at(feature_cut_index).upper_bound());
        }
        // make the right node
        {
            // set the left bound of the right child to the cut value
            bound.segment_at(feature_cut_index).update_lower_bound(feature_cut_value);

            kdnode->right_ =
                build(right_indices_range.first, right_indices_range.second, feature_cut_index, depth + 1, bound);
            // provide a parent pointer to the child node for reversed traversal of the kdtree
            kdnode->right_->parent_ = kdnode;

            // reset the left bound of the bounding box to the current kdnode left bound
            bound.segment_at(feature_cut_index)
                .update_lower_bound(kdnode->bound_.segment_at(feature_cut_index).lower_bound());
        }
    } else {
        kdnode = &nodes_->emplace_back(std::make_pair(indices_range_first, indices_range_last), bound);
        kdnode->set_node_id(n_nodes_++);
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
        writer.Int64(n_samples_);

        writer.String("n_features");
        writer.Int64(n_features_);

        options_.serialize(writer);

        writer.String("bounding_box");
        writer.StartArray();
        for (std::size_t feature_index = 0; feature_index < n_features_; ++feature_index) {
            writer.StartArray();

            if constexpr (std::is_integral_v<DataType>) {
                writer.Int64(bound_.segment_at(feature_index).lower_bound());
                writer.Int64(bound_.segment_at(feature_index).upper_bound());

            } else if constexpr (std::is_floating_point_v<DataType>) {
                writer.Double(bound_.segment_at(feature_index).lower_bound());
                writer.Double(bound_.segment_at(feature_index).upper_bound());
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