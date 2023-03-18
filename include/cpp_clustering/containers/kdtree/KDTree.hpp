#pragma once

#include "cpp_clustering/common/Utils.hpp"
#include "cpp_clustering/containers/kdtree/KDNode.hpp"
#include "cpp_clustering/containers/kdtree/KDTreeUtils.hpp"
#include "cpp_clustering/math/random/Distributions.hpp"

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

namespace cpp_clustering::containers {

namespace fs = std::filesystem;

template <typename Iterator>
class KDTree {
  public:
    struct Options {
        Options& bucket_size(std::size_t bucket_size) {
            bucket_size_ = bucket_size;
            return *this;
        }

        Options& operator=(const Options& options) {
            bucket_size_ = options.bucket_size_;
            return *this;
        }

        void serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
            writer.String("options");

            writer.StartObject();

            writer.String("bucket_size");
            writer.Int64(bucket_size_);

            writer.String("n_samples_for_variance_computation");
            writer.Int64(n_samples_for_variance_computation_);

            writer.EndObject();
        }
        // the maximum number of samples per leaf node
        std::size_t bucket_size_ = 10;
        // number of samples used to compute the variance for the pivot axis selection
        std::size_t n_samples_for_variance_computation_ = 100;
    };

  public:
    // {samples_first_, samples_last_, n_features_}
    using IteratorPairType = std::tuple<Iterator, Iterator>;

    KDTree(const IteratorPairType& iterator_pair, std::size_t n_features);

    KDTree(const KDTree&) = delete;

    void serialize_kdtree(const std::shared_ptr<KDNode<Iterator>>&    kdnode,
                          rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    void serialize(const fs::path& filepath) const;

    std::shared_ptr<KDNode<Iterator>> deserialize(const rapidjson::Value& rapidjson_kdnode);

    std::shared_ptr<KDNode<Iterator>> deserialize(const fs::path& filepath);

  private:
    std::shared_ptr<KDNode<Iterator>> cycle_through_depth_build(const IteratorPairType&      iterator_pair,
                                                                ssize_t                      cut_feature_index,
                                                                std::size_t                  depth,
                                                                BoundingBoxKDType<Iterator>& kd_bounding_box);

    IteratorPairType iterator_pair_;

    std::size_t n_features_;
    // bounding box hyper rectangle (w.r.t. each dimension)
    BoundingBoxKDType<Iterator> kd_bounding_box_;

    Options options_;

    std::shared_ptr<KDNode<Iterator>> root_;
};

template <typename Iterator>
KDTree<Iterator>::KDTree(const IteratorPairType& iterator_pair, std::size_t n_features)
  : iterator_pair_{iterator_pair}
  , n_features_{n_features}
  , kd_bounding_box_{kdtree::utils::make_kd_bounding_box(std::get<0>(iterator_pair_),
                                                         std::get<1>(iterator_pair_),
                                                         n_features_)}
  , root_{cycle_through_depth_build(iterator_pair_, 0, 0, kd_bounding_box_)} {}

template <typename Iterator>
std::shared_ptr<KDNode<Iterator>> KDTree<Iterator>::cycle_through_depth_build(
    const IteratorPairType&      iterator_pair,
    ssize_t                      cut_feature_index,
    std::size_t                  depth,
    BoundingBoxKDType<Iterator>& kd_bounding_box) {
    const auto& [samples_first, samples_last] = iterator_pair;

    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features_);

    std::shared_ptr<KDNode<Iterator>> node;

    // the current node is not leaf
    if (n_samples > options_.bucket_size_) {
        // cycle through the cut_feature_index (dimension) according to the current depth & post-increment depth
        cut_feature_index = (depth++) % n_features_;

        node = std::make_shared<KDNode<Iterator>>(
            samples_first, samples_last, n_features_, cut_feature_index, kd_bounding_box);

        const std::size_t        median_index = n_samples / 2;
        const DataType<Iterator> median_value = *(samples_first + median_index * n_features_ + cut_feature_index);

        auto left_samples_first = samples_first;
        auto left_samples_last  = samples_first + median_index * n_features_;
        // get the value of the median value right after the cut (new right bound according to the current cut axis)
        kd_bounding_box[cut_feature_index].second = median_value;

        node->left_ = cycle_through_depth_build(
            {left_samples_first, left_samples_last}, cut_feature_index, depth, kd_bounding_box);
        // restore the bounding box that was cut at the median value to the previous one
        kd_bounding_box[cut_feature_index].second = node->kd_bounding_box_[cut_feature_index].second;

        auto right_samples_first = samples_first + median_index * n_features_ + n_features_;
        auto right_samples_last  = samples_last;
        // get the value of the median value right before the cut (new right bound according to the current cut axis)
        kd_bounding_box[cut_feature_index].first = median_value;

        node->right_ = cycle_through_depth_build(
            {right_samples_first, right_samples_last}, cut_feature_index, depth, kd_bounding_box);
        // restore the bounding box that was cut at the median value to the previous one
        kd_bounding_box[cut_feature_index].first = node->kd_bounding_box_[cut_feature_index].first;

    } else {
        node = std::make_shared<KDNode<Iterator>>(samples_first, samples_last, n_features_, kd_bounding_box);
    }
    return node;
}

template <typename Iterator>
void KDTree<Iterator>::serialize_kdtree(const std::shared_ptr<KDNode<Iterator>>&    kdnode,
                                        rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
    writer.StartObject();

    writer.String("axis");
    writer.Int64(kdnode->cut_feature_index_);

    writer.String("value");
    writer.Double(kdnode->samples_.first[kdnode->cut_feature_index_]);

    writer.String("left");
    if (!kdnode->is_leaf()) {
        serialize_kdtree(kdnode->left_, writer);

    } else {
        kdnode->serialize_kdnode(writer);
    }

    writer.String("right");
    if (!kdnode->is_leaf()) {
        serialize_kdtree(kdnode->right_, writer);

    } else {
        kdnode->serialize_kdnode(writer);
    }
    writer.EndObject();
}

template <typename Iterator>
void KDTree<Iterator>::serialize(const fs::path& filepath) const {
    rapidjson::Document                        document;
    rapidjson::StringBuffer                    buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

    writer.StartObject();

    writer.String("n_samples");
    writer.Int64(common::utils::get_n_samples(std::get<0>(iterator_pair_), std::get<1>(iterator_pair_), n_features_));

    writer.String("n_features");
    writer.Int64(n_features_);

    options_.serialize(writer);

    writer.String("bounding_box");
    writer.StartArray();
    for (std::size_t feature_index = 0; feature_index < n_features_; ++feature_index) {
        writer.StartArray();
        writer.Double(kd_bounding_box_[feature_index].first);
        writer.Double(kd_bounding_box_[feature_index].second);
        writer.EndArray();
    }
    writer.EndArray();

    writer.String("tree");
    serialize_kdtree(root_, writer);

    writer.EndObject();

    document.Parse(buffer.GetString());

    std::ofstream                                output_file(filepath);
    rapidjson::OStreamWrapper                    output_stream_wrapper(output_file);
    rapidjson::Writer<rapidjson::OStreamWrapper> filewriter(output_stream_wrapper);
    document.Accept(filewriter);
    output_file.close();
}

template <typename Iterator>
std::shared_ptr<KDNode<Iterator>> KDTree<Iterator>::deserialize(const rapidjson::Value& rapidjson_kdnode) {
    const std::size_t axis = rapidjson_kdnode.GetInt64();
    // if the node is leaf
    if (axis == -1) {
        //

    } else {
        //
    }
    return nullptr;
}

template <typename Iterator>
std::shared_ptr<KDNode<Iterator>> KDTree<Iterator>::deserialize(const fs::path& filepath) {
    std::ifstream             input_file(filepath);
    rapidjson::IStreamWrapper input_stream_wrapper(input_file);
    rapidjson::Document       document;
    document.ParseStream(input_stream_wrapper);
    input_file.close();
    return deserialize(document);
}

}  // namespace cpp_clustering::containers