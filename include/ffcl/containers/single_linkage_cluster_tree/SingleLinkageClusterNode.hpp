#pragma once

#include "ffcl/common/Utils.hpp"

#include <memory>
#include <numeric>

namespace ffcl {

template <typename SampleIndexType, typename SampleValueType>
class SingleLinkageClusterNode {
  public:
    using NodeType = SingleLinkageClusterNode<SampleIndexType, SampleValueType>;
    using NodePtr  = std::shared_ptr<NodeType>;

    SingleLinkageClusterNode(const SampleIndexType& representative, const SampleValueType& level = 0)
      : representative_{representative}
      , level_{level} {}

    bool is_leaf() const {
        // could do level_ == 0 but that might require performing float equality
        return (left_ == nullptr && right_ == nullptr);
    }

  public:
    SampleIndexType representative_;

    SampleValueType level_;

    NodePtr parent_, left_, right_;
};

}  // namespace ffcl