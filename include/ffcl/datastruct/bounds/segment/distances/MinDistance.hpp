#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/bounds/segment/StaticSegment.hpp"

#include "ffcl/datastruct/bounds/segment/LowerBoundAndLength.hpp"
#include "ffcl/datastruct/bounds/segment/LowerBoundAndUpperBound.hpp"

namespace ffcl::datastruct::bounds {

template <typename FirstSegment, typename SecondSegment>
constexpr auto min_distance(const FirstSegment& first_segment, const SecondSegment& second_segment) {
    static_assert(common::is_crtp_of<FirstSegment, segment::StaticSegment>::value,
                  "Provided a FirstSegment that does not inherit from StaticSegment<Derived>");
    static_assert(common::is_crtp_of<SecondSegment, segment::StaticSegment>::value,
                  "Provided a SecondSegment that does not inherit from StaticSegment<Derived>");

    using ValueType = std::common_type_t<typename FirstSegment::ValueType, typename SecondSegment::ValueType>;

    if (first_segment.upper_bound() < second_segment.lower_bound()) {
        return second_segment.lower_bound() - first_segment.upper_bound();

    } else if (second_segment.upper_bound() < first_segment.lower_bound()) {
        return first_segment.lower_bound() - second_segment.upper_bound();

    } else {
        return static_cast<ValueType>(0);
    }
}

}  // namespace ffcl::datastruct::bounds