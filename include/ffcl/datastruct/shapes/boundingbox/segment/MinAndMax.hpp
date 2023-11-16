#pragma once

namespace ffcl::datastruct::boundingbox::segment {

template <typename ValueType>
class MinAndMax {
  public:
    using SegmentType = std::pair<ValueType, ValueType>;

    MinAndMax(const SegmentType& segment_representation)
      : segment_representation_{segment_representation} {}

    MinAndMax(SegmentType&& segment_representation) noexcept
      : segment_representation_{std::move(segment_representation)} {}

    constexpr ValueType length_from_centroid() const {
        return (segment_representation_.second - segment_representation_.first) / 2;
    }

    constexpr ValueType centroid() const {
        return (segment_representation_.first + segment_representation_.second) / 2;
    }

  private:
    // segment represented as a minimum and a maximum value
    SegmentType segment_representation_;
};

}  // namespace ffcl::datastruct::boundingbox::segment