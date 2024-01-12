#pragma once

#include "ffcl/datastruct/bounds/segment_representation/StaticSegmentRepresentation.hpp"

namespace ffcl::datastruct::bounds::segment_representation {

template <typename Value>
class MiddleAndLength : public StaticSegmentRepresentation<MiddleAndLength<Value>> {
  public:
    using ValueType   = Value;
    using SegmentType = std::pair<ValueType, ValueType>;

    MiddleAndLength(const ValueType& middle, const ValueType& length)
      : MiddleAndLength(std::make_pair(middle, length)) {}

    MiddleAndLength(const SegmentType& segment_representation)
      : segment_representation_{segment_representation.second} {
        segment_representation_.second /= 2;
    }

    MiddleAndLength(SegmentType&& segment_representation) noexcept
      : segment_representation_{std::move(segment_representation)} {
        segment_representation_.second /= 2;
    }

    constexpr auto read_only_first_impl() const {
        return segment_representation_.first;
    }

    auto& read_write_first_impl() {
        return segment_representation_.first;
    }

    constexpr auto read_only_second_impl() const {
        return segment_representation_.second;
    }

    auto& read_write_second_impl() {
        return segment_representation_.second;
    }

    constexpr auto length_from_centroid_impl() const {
        return segment_representation_.second;
    }

    constexpr auto centroid_impl() const {
        return segment_representation_.first;
    }

  private:
    // segment_representation represented as a length and a middle point that cuts it in half
    // this class actually stores half of the length
    SegmentType segment_representation_;
};

}  // namespace ffcl::datastruct::bounds::segment_representation