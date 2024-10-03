#pragma once

#include "ffcl/common/Utils.hpp"

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
constexpr auto make_edge(const Index& index_1, const Index& index_2, const Distance& distance) {
    return std::make_tuple(index_1, index_2, distance);
}

template <typename Index, typename Distance>
constexpr auto make_default_edge() {
    return make_edge(common::infinity<Index>(), common::infinity<Index>(), common::infinity<Distance>());
}

}  // namespace ffcl::datastruct::mst

template <typename Index, typename Distance>
struct std::less<std::tuple<Index, Index, Distance>> {
    bool operator()(const ffcl::datastruct::mst::Edge<Index, Distance>& edge1,
                    const ffcl::datastruct::mst::Edge<Index, Distance>& edge2) const {
        // Only compare based on the third element (distance)
        return std::get<2>(edge1) < std::get<2>(edge2);
    }
};

namespace ffcl::datastruct::mst {

template <typename Index, typename Distance>
auto sort(EdgesList<Index, Distance>&& mst) {
    std::sort(mst.begin(), mst.end(), std::less<Edge<Index, Distance>>());

    return mst;
}

template <typename Index, typename Distance>
auto sort_copy(const EdgesList<Index, Distance>& mst) {
    auto mst_copy = mst;

    std::sort(mst_copy.begin(), mst_copy.end(), std::less<Edge<Index, Distance>>());

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
