#pragma once

namespace ffcl::datastruct::bounds::segment_representation {

template <typename ValueType>
class PositionAndLength {
  public:
    using SegmentType = std::pair<ValueType, ValueType>;

    PositionAndLength(const SegmentType& segment_representation)
      : segment_representation_{segment_representation} {}

    PositionAndLength(SegmentType&& segment_representation) noexcept
      : segment_representation_{std::move(segment_representation)} {}

    constexpr ValueType length_from_middle() const {
        return segment_representation_.second / 2;
    }

    constexpr ValueType middle() const {
        return segment_representation_.first + length_from_middle();
    }

  private:
    // segment_representation represented as a reference position and a length relative to that position
    SegmentType segment_representation_;
};

}  // namespace ffcl::datastruct::bounds::segment_representation