#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDTreeUtils.hpp"
#include "ffcl/math/random/Distributions.hpp"

#include <sys/types.h>  // ssize_t
#include <algorithm>
#include <array>
#include <memory>

#include "rapidjson/writer.h"

namespace ffcl::containers {

template <typename Iterator>
struct KDNodeView {
    KDNodeView(IteratorPairType<Iterator>         iterator_pair,
               std::size_t                        n_features,
               const BoundingBoxKDType<Iterator>& kd_bounding_box);

    KDNodeView(IteratorPairType<Iterator>         iterator_pair,
               std::size_t                        n_features,
               ssize_t                            cut_feature_index,
               const BoundingBoxKDType<Iterator>& kd_bounding_box);

    KDNodeView(const KDNodeView&) = delete;

    bool is_empty() const;

    bool is_leaf() const;

    void serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    // might contain [0, bucket_size] samples if the node is leaf, else only 1
    IteratorPairType<Iterator> samples_iterator_pair_;
    std::size_t                n_features_;
    ssize_t                    cut_feature_index_;
    // bounding box hyper rectangle (w.r.t. each dimension)
    BoundingBoxKDType<Iterator>           kd_bounding_box_;
    std::shared_ptr<KDNodeView<Iterator>> left_;
    std::shared_ptr<KDNodeView<Iterator>> right_;
};

template <typename Iterator>
KDNodeView<Iterator>::KDNodeView(IteratorPairType<Iterator>         iterator_pair,
                                 std::size_t                        n_features,
                                 const BoundingBoxKDType<Iterator>& kd_bounding_box)
  : samples_iterator_pair_{iterator_pair}
  , n_features_{n_features}
  , cut_feature_index_{-1}
  , kd_bounding_box_{kd_bounding_box} {}

template <typename Iterator>
KDNodeView<Iterator>::KDNodeView(IteratorPairType<Iterator>         iterator_pair,
                                 std::size_t                        n_features,
                                 ssize_t                            cut_feature_index,
                                 const BoundingBoxKDType<Iterator>& kd_bounding_box)
  : samples_iterator_pair_{iterator_pair}
  , n_features_{n_features}
  , cut_feature_index_{cut_feature_index}
  , kd_bounding_box_{kd_bounding_box} {}

template <typename Iterator>
bool KDNodeView<Iterator>::is_empty() const {
    return std::distance(samples_iterator_pair_.first, samples_iterator_pair_.second) == static_cast<std::ptrdiff_t>(0);
}

template <typename Iterator>
bool KDNodeView<Iterator>::is_leaf() const {
    return cut_feature_index_ == -1;
}

template <typename Iterator>
void KDNodeView<Iterator>::serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
    using DataType = DataType<Iterator>;

    static_assert(std::is_floating_point_v<DataType> || std::is_integral_v<DataType>,
                  "Unsupported type during kdnode serialization");

    writer.StartArray();
    // upper-left and lower-right (with sentinel) iterators
    const auto [range_first, range_last] = samples_iterator_pair_;

    const std::size_t n_samples = common::utils::get_n_samples(range_first, range_last, n_features_);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // sample (feature vector) array
        writer.StartArray();
        for (std::size_t feature_index = 0; feature_index < n_features_; ++feature_index) {
            if constexpr (std::is_integral_v<DataType>) {
                writer.Int64(samples_iterator_pair_.first[sample_index * n_features_ + feature_index]);

            } else if constexpr (std::is_floating_point_v<DataType>) {
                writer.Double(samples_iterator_pair_.first[sample_index * n_features_ + feature_index]);
            }
        }
        writer.EndArray();
    }
    writer.EndArray();
}

}  // namespace ffcl::containers