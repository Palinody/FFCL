#pragma once

#include "cpp_clustering/common/Utils.hpp"
#include "cpp_clustering/containers/kdtree/KDNode.hpp"
#include "cpp_clustering/containers/kdtree/KDTreeUtils.hpp"
#include "cpp_clustering/math/random/Distributions.hpp"

#include <sys/types.h>  // ssize_t
#include <algorithm>
#include <array>
#include <memory>

namespace cpp_clustering::containers {

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

        std::size_t bucket_size_ = 10;
    };

  public:
    KDTree(Iterator samples_first, Iterator samples_last, std::size_t n_features);

    KDTree(const KDTree&) = delete;

    void print_kdtree(const std::shared_ptr<KDNode<Iterator>>& kdnode) const;
    void print() const;

  private:
    std::shared_ptr<KDNode<Iterator>> cycle_through_depth_build(Iterator                     samples_first,
                                                                Iterator                     samples_last,
                                                                ssize_t                      cut_feature_index,
                                                                std::size_t                  depth,
                                                                BoundingBoxKDType<Iterator>& kd_bounding_box);

    std::size_t n_features_;
    // bounding box hyper rectangle (w.r.t. each dimension)
    BoundingBoxKDType<Iterator> kd_bounding_box_;

    Options options_;

    std::shared_ptr<KDNode<Iterator>> root_;
};

template <typename Iterator>
KDTree<Iterator>::KDTree(Iterator samples_first, Iterator samples_last, std::size_t n_features)
  : n_features_{n_features}
  , kd_bounding_box_{kdtree::utils::make_kd_bounding_box(samples_first, samples_last, n_features_)}
  , root_{cycle_through_depth_build(samples_first, samples_last, 0, 0, kd_bounding_box_)} {}

template <typename Iterator>
std::shared_ptr<KDNode<Iterator>> KDTree<Iterator>::cycle_through_depth_build(
    Iterator                     samples_first,
    Iterator                     samples_last,
    ssize_t                      cut_feature_index,
    std::size_t                  depth,
    BoundingBoxKDType<Iterator>& kd_bounding_box) {
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

        node->left_ =
            cycle_through_depth_build(left_samples_first, left_samples_last, cut_feature_index, depth, kd_bounding_box);
        // restore the bounding box that was cut at the median value to the previous one
        kd_bounding_box[cut_feature_index].second = node->kd_bounding_box_[cut_feature_index].second;

        auto right_samples_first = samples_first + median_index * n_features_ + n_features_;
        auto right_samples_last  = samples_last;
        // get the value of the median value right before the cut (new right bound according to the current cut axis)
        kd_bounding_box[cut_feature_index].first = median_value;

        node->right_ = cycle_through_depth_build(
            right_samples_first, right_samples_last, cut_feature_index, depth, kd_bounding_box);
        // restore the bounding box that was cut at the median value to the previous one
        kd_bounding_box[cut_feature_index].first = node->kd_bounding_box_[cut_feature_index].first;

    } else {
        node = std::make_shared<KDNode<Iterator>>(samples_first, samples_last, n_features_, kd_bounding_box);
    }
    return node;
}

#include <iostream>

template <typename Iterator>
void print_range(Iterator first, Iterator last) {
    while (first != last) {
        std::cout << *(first++) << " ";
    }
    std::cout << "\n";
}

template <typename Iterator>
void print_ranges(Iterator first, Iterator last, std::size_t n_features) {
    for (; first != last; std::advance(first, n_features)) {
        print_range(first, first + n_features);
    }
}

template <typename Iterator>
void KDTree<Iterator>::print_kdtree(const std::shared_ptr<KDNode<Iterator>>& kdnode) const {
    // static std::size_t counter = 0;

    const bool is_leaf = kdnode->cut_feature_index_ == -1 ? true : false;

    if (is_leaf) {
        if (kdnode->is_empty()) {
            std::cout << "Leaf(empty):\n";

        } else {
            std::cout << "Leaf:\n";
        }
        // counter += common::utils::get_n_samples(kdnode->samples_.first, kdnode->samples_.second, n_features_);
        print_ranges<Iterator>(kdnode->samples_.first, kdnode->samples_.second, n_features_);
    } else {
        std::cout << "Node:\n";
        print_range<Iterator>(kdnode->samples_.first, kdnode->samples_.second);

        // counter += common::utils::get_n_samples(kdnode->samples_.first, kdnode->samples_.second, n_features_);

        print_kdtree(kdnode->left_);
        print_kdtree(kdnode->right_);
    }
    // std::cout << counter << "\n";
}

template <typename Iterator>
void KDTree<Iterator>::print() const {
    print_kdtree(root_);
}

}  // namespace cpp_clustering::containers