#pragma once

#include <algorithm>
#include <tuple>
#include <vector>

#include <iostream>

namespace ffcl::mst {

template <typename SampleIndexType, typename SampleValueType>
using EdgeType = std::tuple<SampleIndexType, SampleIndexType, SampleValueType>;

template <typename SampleIndexType, typename SampleValueType>
using MinimumSpanningTreeType = std::vector<EdgeType<SampleIndexType, SampleValueType>>;

template <typename SampleIndexType, typename SampleValueType>
auto sort(MinimumSpanningTreeType<SampleIndexType, SampleValueType>&& mst) {
    auto comparator = [](const EdgeType<SampleIndexType, SampleValueType>& lhs,
                         const EdgeType<SampleIndexType, SampleValueType>& rhs) {
        return std::get<2>(lhs) < std::get<2>(rhs);
    };

    std::sort(mst.begin(), mst.end(), comparator);

    return mst;
}

template <typename SampleIndexType, typename SampleValueType>
auto sort_copy(const MinimumSpanningTreeType<SampleIndexType, SampleValueType>& mst) {
    auto mst_copy = mst;

    auto comparator = [](const EdgeType<SampleIndexType, SampleValueType>& lhs,
                         const EdgeType<SampleIndexType, SampleValueType>& rhs) {
        return std::get<2>(lhs) < std::get<2>(rhs);
    };

    std::sort(mst_copy.begin(), mst_copy.end(), comparator);

    return mst_copy;
}

template <typename SampleIndexType, typename SampleValueType>
void print(const MinimumSpanningTreeType<SampleIndexType, SampleValueType>& mst) {
    std::cout << "Minimum Spanning Tree (MST):\n";
    for (const auto& edge : mst) {
        std::cout << "(" << std::get<0>(edge) << ", " << std::get<1>(edge) << ", " << std::get<2>(edge) << "), \n";
    }
    std::cout << "\n";
}

}  // namespace ffcl::mst