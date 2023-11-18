#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/bounds/Vertex.hpp"

namespace ffcl::datastruct::bounds {

template <typename ValueType>
class Ball {
  public:
    using CentroidType = Vertex<ValueType>;

    Ball(const CentroidType& centroid, const ValueType& radius)
      : centroid_{centroid}
      , radius_{radius} {}

    Ball(CentroidType&& centroid, const ValueType& radius) noexcept
      : centroid_{std::move(centroid)}
      , radius_{radius} {}

    std::size_t n_features() const {
        return centroid_.size();
    }

    ValueType length_from_centroid() const {
        return radius_;
    }

    constexpr ValueType length_from_centroid(std::size_t feature_index) const {
        common::ignore_parameters(feature_index);
        return radius_;
    }

    CentroidType centroid() const {
        auto result = CentroidType(n_features());

        for (std::size_t feature_index = 0; feature_index < n_features(); ++feature_index) {
            result[feature_index] = centroid_[feature_index].centroid();
        }
        return result;
    }

  private:
    // a ball can be represented as a single segment
    CentroidType centroid_;
    ValueType    radius_;
};

}  // namespace ffcl::datastruct::bounds