#pragma once

#include "ffcl/datastruct/bounds/segment_representation/StaticSegmentRepresentation.hpp"

namespace ffcl::datastruct::bounds::segment_representation {

template <typename Value>
class MinAndMax : public StaticSegmentRepresentation<MinAndMax<Value>> {
  public:
    using ValueType   = Value;
    using SegmentType = std::pair<ValueType, ValueType>;

    MinAndMax(const ValueType& min, const ValueType& max)
      : MinAndMax(std::make_pair(min, max)) {}

    MinAndMax(const SegmentType& segment_representation)
      : segment_representation_{segment_representation} {}

    MinAndMax(SegmentType&& segment_representation) noexcept
      : segment_representation_{std::move(segment_representation)} {}

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
        return (segment_representation_.second - segment_representation_.first) / 2;
    }

    constexpr auto centroid_impl() const {
        return (segment_representation_.first + segment_representation_.second) / 2;
    }

  private:
    // segment_representation represented as a minimum and a maximum value
    SegmentType segment_representation_;
};

}  // namespace ffcl::datastruct::bounds::segment_representation