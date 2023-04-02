#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDNode.hpp"
#include "ffcl/containers/kdtree/KDTreeUtils.hpp"
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
        Options& bucket_size(std::size_t bucket_size) {
            bucket_size_ = bucket_size;
            return *this;
        }

        Options& max_depth(ssize_t max_depth) {
            max_depth_ = max_depth;
            return *this;
        }

        Options& n_samples_fraction_for_variance_computation(
            const std::pair<std::size_t, std::size_t>& n_samples_fraction_for_variance_computation) {
            n_samples_fraction_for_variance_computation_ = n_samples_fraction_for_variance_computation;
            return *this;
        }

        Options& operator=(const Options& options) {
            bucket_size_                                 = options.bucket_size_;
            max_depth_                                   = options.max_depth_;
            n_samples_fraction_for_variance_computation_ = options.n_samples_fraction_for_variance_computation_;
            return *this;
        }

        void serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
            writer.String("options");

            writer.StartObject();

            writer.String("bucket_size");
            writer.Int64(bucket_size_);

            writer.String("max_depth");
            writer.Int64(max_depth_);

            writer.String("n_samples_fraction_for_variance_computation");
            writer.Double(n_samples_fraction_for_variance_computation_);

            writer.EndObject();
        }
        // the maximum number of samples per leaf node
        std::size_t bucket_size_ = 40;
        // the maximum recursion depth. Defaults to infinity
        ssize_t max_depth_ = common::utils::infinity<ssize_t>();
        // number of samples used to compute the variance for the pivot axis selection
        double n_samples_fraction_for_variance_computation_ = 0.1;
    };

  public:
    KDTree(const IteratorPairType<Iterator>& iterator_pair, std::size_t n_features);

    KDTree(const KDTree&) = delete;

    void serialize_kdtree(const std::shared_ptr<KDNode<Iterator>>&    kdnode,
                          rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    void serialize(const fs::path& filepath) const;

  private:
    std::shared_ptr<KDNode<Iterator>> cycle_through_axes_build(const IteratorPairType<Iterator>& iterator_pair,
                                                               ssize_t                           cut_feature_index,
                                                               ssize_t                           depth,
                                                               BoundingBoxKDType<Iterator>&      kd_bounding_box);

    std::shared_ptr<KDNode<Iterator>> highest_variance_build(const IteratorPairType<Iterator>& iterator_pair,
                                                             ssize_t                           cut_feature_index,
                                                             ssize_t                           depth,
                                                             BoundingBoxKDType<Iterator>&      kd_bounding_box);

    std::shared_ptr<KDNode<Iterator>> maximum_spread_build(const IteratorPairType<Iterator>& iterator_pair,
                                                           ssize_t                           cut_feature_index,
                                                           ssize_t                           depth,
                                                           BoundingBoxKDType<Iterator>&      kd_bounding_box);

    IteratorPairType<Iterator> iterator_pair_;

    std::size_t n_features_;
    // bounding box hyper rectangle (w.r.t. each dimension)
    BoundingBoxKDType<Iterator> kd_bounding_box_;

    Options options_;

    std::shared_ptr<KDNode<Iterator>> root_;
};
/*
template <typename Iterator>
KDTree<Iterator>::KDTree(const IteratorPairType<Iterator>& iterator_pair, std::size_t n_features)
  : iterator_pair_{iterator_pair}
  , n_features_{n_features}
  , kd_bounding_box_{kdtree::utils::make_kd_bounding_box(std::get<0>(iterator_pair_),
                                                         std::get<1>(iterator_pair_),
                                                         n_features_)}
  , root_{cycle_through_axes_build(iterator_pair_, 0, 0, kd_bounding_box_)} {}
*/
// /*
template <typename Iterator>
KDTree<Iterator>::KDTree(const IteratorPairType<Iterator>& iterator_pair, std::size_t n_features)
  : iterator_pair_{iterator_pair}
  , n_features_{n_features}
  , kd_bounding_box_{kdtree::utils::make_kd_bounding_box(std::get<0>(iterator_pair_),
                                                         std::get<1>(iterator_pair_),
                                                         n_features_)}
  , root_{highest_variance_build(iterator_pair_,
                                 kdtree::utils::select_axis_with_largest_variance<Iterator>(
                                     std::get<0>(iterator_pair_),
                                     std::get<1>(iterator_pair_),
                                     n_features_,
                                     options_.n_samples_fraction_for_variance_computation_),
                                 0,
                                 kd_bounding_box_)} {}

// */
/*
template <typename Iterator>
KDTree<Iterator>::KDTree(const IteratorPairType<Iterator>& iterator_pair, std::size_t n_features)
  : iterator_pair_{iterator_pair}
  , n_features_{n_features}
  , kd_bounding_box_{kdtree::utils::make_kd_bounding_box(std::get<0>(iterator_pair_),
                                                         std::get<1>(iterator_pair_),
                                                         n_features_)}
  , root_{maximum_spread_build(
        iterator_pair_,
        kdtree::utils::select_axis_with_largest_bounding_box_difference<Iterator>(kd_bounding_box_),
        0,
        kd_bounding_box_)} {}
*/

template <typename Iterator>
std::shared_ptr<KDNode<Iterator>> KDTree<Iterator>::cycle_through_axes_build(
    const IteratorPairType<Iterator>& iterator_pair,
    ssize_t                           cut_feature_index,
    ssize_t                           depth,
    BoundingBoxKDType<Iterator>&      kd_bounding_box) {
    const auto& [samples_first, samples_last] = iterator_pair;

    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features_);

    if (n_samples == 0) {
        return nullptr;
    }
    std::shared_ptr<KDNode<Iterator>> node;

    // the current node is not leaf
    if (n_samples > options_.bucket_size_ && depth < options_.max_depth_) {
        // cycle through the cut_feature_index (dimension) according to the current depth & post-increment depth
        // select the cut_feature_index according to the one with the most variance
        cut_feature_index = depth % n_features_;

        node = std::make_shared<KDNode<Iterator>>(
            samples_first, samples_last, n_features_, cut_feature_index, kd_bounding_box);

        const std::size_t        median_index = n_samples / 2;
        const DataType<Iterator> median_value = *(samples_first + median_index * n_features_ + cut_feature_index);

        // all the points at the left of the pivot point
        auto left_samples_iterator_pair = std::make_pair(samples_first, samples_first + median_index * n_features_);

        // set the right bound of the left child to the cut value
        kd_bounding_box[cut_feature_index].second = median_value;

        node->left_ =
            cycle_through_axes_build(left_samples_iterator_pair, cut_feature_index, depth + 1, kd_bounding_box);

        // reset the right bound of the bounding box to the current node right bound
        kd_bounding_box[cut_feature_index].second = node->kd_bounding_box_[cut_feature_index].second;

        if (n_samples > 2) {
            // all the points at the right of the pivot point
            auto right_samples_iterator_pair =
                std::make_pair(samples_first + median_index * n_features_ + n_features_, samples_last);

            // set the left bound of the right child to the cut value
            kd_bounding_box[cut_feature_index].first = median_value;

            node->right_ =
                cycle_through_axes_build(right_samples_iterator_pair, cut_feature_index, depth + 1, kd_bounding_box);

            // reset the left bound of the bounding box to the current node left bound
            kd_bounding_box[cut_feature_index].first = node->kd_bounding_box_[cut_feature_index].first;
        }
    } else {
        node = std::make_shared<KDNode<Iterator>>(samples_first, samples_last, n_features_, kd_bounding_box);
    }
    return node;
}

template <typename Iterator>
std::shared_ptr<KDNode<Iterator>> KDTree<Iterator>::highest_variance_build(
    const IteratorPairType<Iterator>& iterator_pair,
    ssize_t                           cut_feature_index,
    ssize_t                           depth,
    BoundingBoxKDType<Iterator>&      kd_bounding_box) {
    const auto& [samples_first, samples_last] = iterator_pair;

    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features_);

    if (n_samples == 0) {
        return nullptr;
    }
    std::shared_ptr<KDNode<Iterator>> node;

    // the current node is not leaf
    if (n_samples > options_.bucket_size_ && depth < options_.max_depth_) {
        // select the cut_feature_index according to the one with the most variance
        cut_feature_index = kdtree::utils::select_axis_with_largest_variance<Iterator>(
            samples_first, samples_last, n_features_, options_.n_samples_fraction_for_variance_computation_);

        node = std::make_shared<KDNode<Iterator>>(
            samples_first, samples_last, n_features_, cut_feature_index, kd_bounding_box);

        const std::size_t        median_index = n_samples / 2;
        const DataType<Iterator> median_value = *(samples_first + median_index * n_features_ + cut_feature_index);

        // all the points at the left of the pivot point
        auto left_samples_iterator_pair = std::make_pair(samples_first, samples_first + median_index * n_features_);

        // set the right bound of the left child to the cut value
        kd_bounding_box[cut_feature_index].second = median_value;

        node->left_ = highest_variance_build(left_samples_iterator_pair, cut_feature_index, depth + 1, kd_bounding_box);

        // reset the right bound of the bounding box to the current node right bound
        kd_bounding_box[cut_feature_index].second = node->kd_bounding_box_[cut_feature_index].second;

        if (n_samples > 2) {
            // all the points at the right of the pivot point
            auto right_samples_iterator_pair =
                std::make_pair(samples_first + median_index * n_features_ + n_features_, samples_last);

            // set the left bound of the right child to the cut value
            kd_bounding_box[cut_feature_index].first = median_value;

            node->right_ =
                highest_variance_build(right_samples_iterator_pair, cut_feature_index, depth + 1, kd_bounding_box);

            // reset the left bound of the bounding box to the current node left bound
            kd_bounding_box[cut_feature_index].first = node->kd_bounding_box_[cut_feature_index].first;
        }
    } else {
        node = std::make_shared<KDNode<Iterator>>(samples_first, samples_last, n_features_, kd_bounding_box);
    }
    return node;
}

template <typename Iterator>
std::shared_ptr<KDNode<Iterator>> KDTree<Iterator>::maximum_spread_build(
    const IteratorPairType<Iterator>& iterator_pair,
    ssize_t                           cut_feature_index,
    ssize_t                           depth,
    BoundingBoxKDType<Iterator>&      kd_bounding_box) {
    const auto& [samples_first, samples_last] = iterator_pair;

    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features_);

    if (n_samples == 0) {
        return nullptr;
    }
    std::shared_ptr<KDNode<Iterator>> node;

    // the current node is not leaf
    if (n_samples > options_.bucket_size_ && depth < options_.max_depth_) {
        // select the cut_feature_index according to the one with the most spread (min-max values)
        cut_feature_index = kdtree::utils::select_axis_with_largest_bounding_box_difference<Iterator>(kd_bounding_box_);

        node = std::make_shared<KDNode<Iterator>>(
            samples_first, samples_last, n_features_, cut_feature_index, kd_bounding_box);

        const std::size_t        median_index = n_samples / 2;
        const DataType<Iterator> median_value = *(samples_first + median_index * n_features_ + cut_feature_index);

        // all the points at the left of the pivot point
        auto left_samples_iterator_pair = std::make_pair(samples_first, samples_first + median_index * n_features_);

        // set the right bound of the left child to the cut value
        kd_bounding_box[cut_feature_index].second = median_value;

        node->left_ = maximum_spread_build(left_samples_iterator_pair, cut_feature_index, depth + 1, kd_bounding_box);

        // reset the right bound of the bounding box to the current node right bound
        kd_bounding_box[cut_feature_index].second = node->kd_bounding_box_[cut_feature_index].second;

        if (n_samples > 2) {
            // all the points at the right of the pivot point
            auto right_samples_iterator_pair =
                std::make_pair(samples_first + median_index * n_features_ + n_features_, samples_last);

            // set the left bound of the right child to the cut value
            kd_bounding_box[cut_feature_index].first = median_value;

            node->right_ =
                maximum_spread_build(right_samples_iterator_pair, cut_feature_index, depth + 1, kd_bounding_box);

            // reset the left bound of the bounding box to the current node left bound
            kd_bounding_box[cut_feature_index].first = node->kd_bounding_box_[cut_feature_index].first;
        }
    } else {
        node = std::make_shared<KDNode<Iterator>>(samples_first, samples_last, n_features_, kd_bounding_box);
    }
    return node;
}

template <typename Iterator>
void KDTree<Iterator>::serialize_kdtree(const std::shared_ptr<KDNode<Iterator>>&    kdnode,
                                        rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
    writer.StartObject();
    {
        writer.String("axis");
        writer.Int64(kdnode->cut_feature_index_);

        writer.String("points");
        kdnode->serialize_kdnode(writer);

        // continue the recursion if the current node is not leaf
        if (!kdnode->is_leaf()) {
            writer.String("left");
            serialize_kdtree(kdnode->left_, writer);

            // The right pointer might be nullptr when a node had 2 samples. The median computation chooses the second
            // sample as the pivot because the median of 2 samples will output index 1. The other index will be 0 and
            // thus the left child
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