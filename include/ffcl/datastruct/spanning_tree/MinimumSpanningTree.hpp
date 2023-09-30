#pragma once

#include <algorithm>
#include <tuple>
#include <vector>

#include <iostream>

namespace ffcl::mst {

template <typename IndexType, typename ValueType>
using Edge = std::tuple<IndexType, IndexType, ValueType>;

template <typename IndexType, typename ValueType>
using MinimumSpanningTree = std::vector<Edge<IndexType, ValueType>>;

template <typename IndexType, typename ValueType>
auto sort(MinimumSpanningTree<IndexType, ValueType>&& mst) {
    auto comparator = [](const Edge<IndexType, ValueType>& lhs, const Edge<IndexType, ValueType>& rhs) {
        return std::get<2>(lhs) < std::get<2>(rhs);
    };

    std::sort(mst.begin(), mst.end(), comparator);

    return mst;
}

template <typename IndexType, typename ValueType>
auto sort_copy(const MinimumSpanningTree<IndexType, ValueType>& mst) {
    auto mst_copy = mst;

    auto comparator = [](const Edge<IndexType, ValueType>& lhs, const Edge<IndexType, ValueType>& rhs) {
        return std::get<2>(lhs) < std::get<2>(rhs);
    };

    std::sort(mst_copy.begin(), mst_copy.end(), comparator);

    return mst_copy;
}

template <typename IndexType, typename ValueType>
void print(const MinimumSpanningTree<IndexType, ValueType>& mst) {
    std::cout << "Minimum Spanning Tree (MST):\n";
    for (const auto& edge : mst) {
        std::cout << "(" << std::get<0>(edge) << ", " << std::get<1>(edge) << ", " << std::get<2>(edge) << "), \n";
    }
    std::cout << "\n";
}

}  // namespace ffcl::mst