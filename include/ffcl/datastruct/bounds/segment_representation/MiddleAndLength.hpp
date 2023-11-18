#pragma once

namespace ffcl::datastruct::bounds::segment_representation {

template <typename ValueType>
class MiddleAndLength {
  public:
    using SegmentType = std::pair<ValueType, ValueType>;

    MiddleAndLength(const SegmentType& segment_representation)
      : segment_representation_{std::make_pair(/**/ segment_representation.first,
                                               /**/ segment_representation.second / 2)} {}

    MiddleAndLength(SegmentType&& segment_representation) noexcept
      : segment_representation_{std::make_pair(/**/ std::move(segment_representation.first),
                                               /**/ std::move(segment_representation.second) / 2)} {}

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