#pragma once

#include <cstdio>
#include <memory>
#include <numeric>

#include "rapidjson/writer.h"

namespace ffcl {

template <typename IndexType, typename ValueType>
struct SingleLinkageClusterNode {
    using NodeType = SingleLinkageClusterNode<IndexType, ValueType>;
    using NodePtr  = std::shared_ptr<NodeType>;

    SingleLinkageClusterNode(const IndexType& representative, const ValueType& level = 0);

    bool is_leaf() const;

    // void serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer,
    //                SamplesIterator                             samples_first,
    //                SamplesIterator                             samples_last,
    //                std::size_t                                 n_features) const;

    IndexType representative_;

    ValueType level_;

    NodePtr parent_, left_, right_;
};

template <typename IndexType, typename ValueType>
SingleLinkageClusterNode<IndexType, ValueType>::SingleLinkageClusterNode(const IndexType& representative,
                                                                         const ValueType& level)
  : representative_{representative}
  , level_{level} {}

template <typename IndexType, typename ValueType>
bool SingleLinkageClusterNode<IndexType, ValueType>::is_leaf() const {
    // could do level_ == 0 but that might require performing float equality
    return (left_ == nullptr && right_ == nullptr);
}

// template <typename IndexType, typename ValueType>
// void SingleLinkageClusterNode<IndexType, ValueType>::serialize(
//     rapidjson::Writer<rapidjson::StringBuffer>& writer,
//     SamplesIterator                             samples_first,
//     SamplesIterator                             samples_last,
//     std::size_t                                 n_features) const {
//     using DataType = bbox::DataType<SamplesIterator>;

//     static_assert(std::is_floating_point_v<DataType> || std::is_integral_v<DataType>,
//                   "Unsupported type during kdnode serialization");

//     writer.StartArray();
//     const auto [indices_range_first, indices_range_last] = indices_iterator_pair_;

//     common::utils::ignore_parameters(samples_last);

//     const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

//     for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
//         // sample (feature vector) array
//         writer.StartArray();
//         for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
//             if constexpr (std::is_integral_v<DataType>) {
//                 writer.Int64(samples_first[indices_range_first[sample_index] * n_features + feature_index]);

//             } else if constexpr (std::is_floating_point_v<DataType>) {
//                 writer.Double(samples_first[indices_range_first[sample_index] * n_features + feature_index]);
//             }
//         }
//         writer.EndArray();
//     }
//     writer.EndArray();
// }

}  // namespace ffcl