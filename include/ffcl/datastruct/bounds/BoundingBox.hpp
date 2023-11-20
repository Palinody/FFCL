#pragma once

#include <vector>

#include "ffcl/datastruct/bounds/Vertex.hpp"

namespace ffcl::datastruct::bounds {

template <typename Segment>
class BoundingBox {
  public:
    using ValueType    = typename Segment::ValueType;
    using SegmentsType = std::vector<Segment>;

    using CentroidType        = Vertex<ValueType, 0>;
    using LengthsFromCentroid = CentroidType;

    BoundingBox(const SegmentsType& segments)
      : center_point_{std::vector<ValueType>(segments.size())}
      , lengths_from_center_{std::vector<ValueType>(segments.size())} {
        for (std::size_t feature_index = 0; feature_index < segments.size(); ++feature_index) {
            center_point_[feature_index]        = segments[feature_index].centroid();
            lengths_from_center_[feature_index] = segments[feature_index].length_from_centroid();
        }
    }

    std::size_t n_features() const {
        return center_point_.size();
    }

    template <typename FeaturesIterator>
    bool is_in_bounds(const FeaturesIterator& features_range_first, const FeaturesIterator& features_range_last) const {
        const std::size_t n_features = std::distance(features_range_first, features_range_last);

        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            // A sample is inside the bounding box if p is in [lo, hi]
            if (features_range_first[feature_index] <
                    center_point_[feature_index] - lengths_from_center_[feature_index] ||
                features_range_first[feature_index] >
                    center_point_[feature_index] + lengths_from_center_[feature_index]) {
                return false;
            }
        }
        return true;
    }

    template <typename FeaturesIterator>
    ValueType distance(const FeaturesIterator& features_range_first,
                       const FeaturesIterator& features_range_last) const {
        assert(center_point_.size() == std::distance(features_range_first, features_range_last));

        return common::math::heuristics::auto_distance(
            features_range_first, features_range_last, center_point_.begin());
    }

    ValueType length_from_centroid() const {
        throw std::runtime_error("No half length to return if no feature dimension is specified for this bound.");
        return ValueType{};
    }

    constexpr ValueType length_from_centroid(std::size_t feature_index) const {
        return lengths_from_center_[feature_index];
    }

    const CentroidType& centroid() const {
        return center_point_;
    }

    CentroidType make_centroid() const {
        return center_point_;
    }

  private:
    // a bounding box represented as a center point coordinate and and the relative lengths from that point to the axis
    // aligned bounds w.r.t. each feature dimension
    CentroidType        center_point_;
    LengthsFromCentroid lengths_from_center_;
};

}  // namespace ffcl::datastruct::bounds