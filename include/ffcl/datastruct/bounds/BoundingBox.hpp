#pragma once

#include <vector>

#include "ffcl/datastruct/bounds/Vertex.hpp"
#include "ffcl/datastruct/bounds/segment_representation/MiddleAndLength.hpp"
#include "ffcl/datastruct/bounds/segment_representation/MinAndMax.hpp"
#include "ffcl/datastruct/bounds/segment_representation/PositionAndLength.hpp"

namespace ffcl::datastruct::bounds {

template <typename SegmentType>
class BoundingBox {
  public:
    using ValueType    = typename SegmentType::ValueType;
    using SegmentsType = std::vector<SegmentType>;

    using CentroidType = std::vector<ValueType>;

    BoundingBox(const SegmentsType& segments)
      : segments_{segments} {}

    BoundingBox(SegmentsType&& segments) noexcept
      : segments_{std::move(segments)} {}

    std::size_t n_features() const {
        return segments_.size();
    }

    ValueType length_from_centroid() const {
        throw std::runtime_error("No half length to return if no feature dimension is specified for this bound.");
        return ValueType{};
    }

    constexpr ValueType length_from_centroid(std::size_t feature_index) const {
        return segments_[feature_index].length_from_centroid();
    }

    CentroidType centroid() const {
        auto result = CentroidType(n_features());

        for (std::size_t feature_index = 0; feature_index < n_features(); ++feature_index) {
            result[feature_index] = segments_[feature_index].centroid();
        }
        return result;
    }

  private:
    SegmentsType segments_;
};

}  // namespace ffcl::datastruct::bounds