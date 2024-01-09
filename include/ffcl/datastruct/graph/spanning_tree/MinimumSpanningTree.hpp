#pragma once

#include <algorithm>
#include <tuple>
#include <vector>

#include <iostream>

namespace ffcl::datastruct::mst {

template <typename Index, typename Distance>
using Edge = std::tuple<Index, Index, Distance>;

template <typename Index, typename Distance>
using EdgesList = std::vector<Edge<Index, Distance>>;

template <typename Index, typename Distance>
auto sort(EdgesList<Index, Distance>&& mst) {
    auto edge_comparator = [](const auto& edge_1, const auto& edge_2) {
        return std::get<2>(edge_1) < std::get<2>(edge_2);
    };

    std::sort(mst.begin(), mst.end(), edge_comparator);

    return mst;
}

template <typename Index, typename Distance>
auto sort_copy(const EdgesList<Index, Distance>& mst) {
    auto mst_copy = mst;

    auto edge_comparator = [](const auto& edge_1, const auto& edge_2) {
        return std::get<2>(edge_1) < std::get<2>(edge_2);
    };

    std::sort(mst_copy.begin(), mst_copy.end(), edge_comparator);

    return mst_copy;
}

template <typename Index, typename Distance>
void print(const EdgesList<Index, Distance>& mst) {
    std::cout << "Minimum Spanning Tree (MST):\n";

    for (const auto& edge : mst) {
        std::cout << "(" << std::get<0>(edge) << ", " << std::get<1>(edge) << ", " << std::get<2>(edge) << "), \n";
    }
    std::cout << "\n";
}

}  // namespace ffcl::datastruct::mst