#pragma once

namespace ffcl::datastruct::bounds::segment_representation {

template <typename Value>
class MinAndMax {
  public:
    using ValueType   = Value;
    using SegmentType = std::pair<Value, Value>;

    MinAndMax(const ValueType& min, const ValueType& max)
      : MinAndMax(std::make_pair(min, max)) {}

    MinAndMax(const SegmentType& segment_representation)
      : segment_representation_{segment_representation} {}

    MinAndMax(SegmentType&& segment_representation) noexcept
      : segment_representation_{std::move(segment_representation)} {}

    constexpr Value length_from_centroid() const {
        return (segment_representation_.second - segment_representation_.first) / 2;
    }

    constexpr Value centroid() const {
        return (segment_representation_.first + segment_representation_.second) / 2;
    }

  private:
    // segment_representation represented as a minimum and a maximum value
    SegmentType segment_representation_;
};

}  // namespace ffcl::datastruct::bounds::segment_representation