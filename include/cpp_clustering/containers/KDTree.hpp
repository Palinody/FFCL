#pragma once

#include "cpp_clustering/math/random/Distributions.hpp"

#include <algorithm>
#include <array>
#include <memory>

namespace cpp_clustering::containers {

template <typename Type, std::size_t KDims>
struct KDTreeNode {
    std::array<Type, KDims>                  point;
    std::size_t                              axis;
    std::unique_ptr<KDTreeNode<Type, KDims>> left;
    std::unique_ptr<KDTreeNode<Type, KDims>> right;
};

template <typename Type, std::size_t KDims>
class KDTree {
  public:
    template <typename SamplesIterator>
    KDTree(const SamplesIterator& samples_first, const SamplesIterator& samples_last);

    template <typename SamplesIterator>
    std::vector<std::array<Type, KDims>> search(const SamplesIterator& samples_first,
                                                const SamplesIterator& samples_last,
                                                const Type&            epsilon);

  private:
    template <typename SamplesIterator>
    std::unique_ptr<KDTreeNode<Type, KDims>> buildTree(const SamplesIterator& samples_first,
                                                       const SamplesIterator& samples_last,
                                                       std::size_t            depth);

    void search(KDTreeNode<Type, KDims>*              node,
                const std::array<Type, KDims>&        point,
                Type                                  distance_sq,
                std::vector<std::array<Type, KDims>>& result,
                std::size_t                           depth);

    std::unique_ptr<KDTreeNode<Type, KDims>> root_;
};

template <typename Type, int KDims>
template <typename SamplesIterator>
std::unique_ptr<KDTreeNode<T, K>> KDTree<T, K>::buildTree(const SamplesIterator& samples_first,
                                                          const SamplesIterator& samples_last,
                                                          std::size_t            depth) {
    const auto n_elements = common::utils::get_n_elements(samples_first, samples_last, KDims);

    if (n_elements == 0) {
        return nullptr;
    }
    std::size_t axis        = depth % KDims;
    std::size_t split_index = n_elements / 2;

    auto compare = [&](const std::array<T, K>& a, const std::array<T, K>& b) { return a[axis] < b[axis]; };

    std::nth_element(samples_first, samples_first + split_index, samples_last, compare);

    std::unique_ptr<KDTreeNode<T, K>> node = std::make_unique<KDTreeNode<T, K>>();
    node->point                            = points[split_index];
    node->axis                             = axis;
    node->left                             = buildTree(points, start, split_index, depth + 1);
    node->right                            = buildTree(points, split_index + 1, end, depth + 1);
    return node;
}

template <typename Type, int KDims>
void KDTree<T, K>::search(KDTreeNode<T, K>*              node,
                          const std::array<T, K>&        point,
                          T                              distance_sq,
                          std::vector<std::array<T, K>>& result,
                          int                            depth) {
    if (!node) {
        return;
    }

    T diff = node->point[depth % K] - point[depth % K];
    T dist = diff * diff;

    if (dist <= distance_sq) {
        T dist_point = 0;
        for (int i = 0; i < K; i++) {
            T d = node->point[i] - point[i];
            dist_point += d * d;
        }

        if (dist_point <= distance_sq) {
            result.push_back(node->point);
        }
    }

    int axis = depth % K;
    if (point[axis] - node->point[axis] <= 0) {
        search(node->left.get(), point, distance_sq, result, depth + 1);
        search(node->right.get(), point, distance_sq, result, depth + 1);
    } else {
        search(node->right.get(), point, distance_sq, result, depth + 1);
        search(node->left.get(), point, distance_sq, result, depth + 1);
    }
}

}  // namespace cpp_clustering::containers