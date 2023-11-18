#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/bounds/Vertex.hpp"

namespace ffcl::datastruct::bounds {

template <typename ValueType, std::size_t Size = 0>
class Ball {
  public:
    using CentroidType = Vertex<ValueType, Size>;

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

    const CentroidType& centroid() const {
        return centroid_;
    }

    const CentroidType make_centroid() const {
        return centroid_;
    }

  private:
    // a ball can be represented as a single segment
    CentroidType centroid_;
    ValueType    radius_;
};

}  // namespace ffcl::datastruct::bounds