#pragma once

#include <algorithm>
#include <tuple>
#include <vector>

#include <iostream>

namespace ffcl::mst {

template <typename Index, typename Value>
using Edge = std::tuple<Index, Index, Value>;

template <typename Index, typename Value>
using MinimumSpanningTree = std::vector<Edge<Index, Value>>;

template <typename Index, typename Value>
auto sort(MinimumSpanningTree<Index, Value>&& mst) {
    auto edge_comparator = [](const auto& edge_1, const auto& edge_2) {
        return std::get<2>(edge_1) < std::get<2>(edge_2);
    };

    std::sort(mst.begin(), mst.end(), edge_comparator);

    return mst;
}

template <typename Index, typename Value>
auto sort_copy(const MinimumSpanningTree<Index, Value>& mst) {
    auto mst_copy = mst;

    auto edge_comparator = [](const auto& edge_1, const auto& edge_2) {
        return std::get<2>(edge_1) < std::get<2>(edge_2);
    };

    std::sort(mst_copy.begin(), mst_copy.end(), edge_comparator);

    return mst_copy;
}

template <typename Index, typename Value>
void print(const MinimumSpanningTree<Index, Value>& mst) {
    std::cout << "Minimum Spanning Tree (MST):\n";

    for (const auto& edge : mst) {
        std::cout << "(" << std::get<0>(edge) << ", " << std::get<1>(edge) << ", " << std::get<2>(edge) << "), \n";
    }
    std::cout << "\n";
}

}  // namespace ffcl::mst