#pragma once

#include "ffcl/common/Utils.hpp"

#include <algorithm>
#include <vector>

namespace ffcl::common::math::linear_algebra {

template <typename LeftFeaturesIterator, typename RightFeaturesIterator>
auto translate_right_range_to_origin(const LeftFeaturesIterator&  left_features_range_first,
                                     const LeftFeaturesIterator&  left_features_range_last,
                                     const RightFeaturesIterator& right_features_range_first,
                                     const RightFeaturesIterator& right_features_range_last) {
    common::ignore_parameters(left_features_range_last);

    using ValueType = typename RightFeaturesIterator::value_type;

    const auto n_features = std::distance(right_features_range_first, right_features_range_last);

    auto right_features_centered_at_origin = std::vector<ValueType>(n_features);

    std::transform(right_features_range_first,
                   right_features_range_last,
                   left_features_range_first,
                   right_features_centered_at_origin.begin(),
                   std::minus<>());

    return right_features_centered_at_origin;
}

}  // namespace ffcl::common::math::linear_algebra