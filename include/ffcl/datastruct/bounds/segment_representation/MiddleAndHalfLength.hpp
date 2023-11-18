#pragma once

namespace ffcl::datastruct::bounds::segment_representation {

template <typename ValueType>
class MiddleAndHalfLength {
  public:
    using SegmentType = std::pair<ValueType, ValueType>;

    MiddleAndHalfLength(const SegmentType& segment_representation)
      : segment_representation_{segment_representation} {}

    MiddleAndHalfLength(SegmentType&& segment_representation) noexcept
      : segment_representation_{std::move(segment_representation)} {}

    constexpr ValueType length_from_middle() const {
        return segment_representation_.second;
    }

    constexpr ValueType middle() const {
        return segment_representation_.first;
    }

  private:
    // segment_representation represented as a length and a middle point that cuts it in half
    // this class actually stores half of the length
    SegmentType segment_representation_;
};

}  // namespace ffcl::datastruct::bounds::segment_representation