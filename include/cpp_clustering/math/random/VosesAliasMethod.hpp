#pragma once

#include "cpp_clustering/math/random/Distributions.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace math::random {

class VosesAliasMethod {
  public:
    explicit VosesAliasMethod(const std::vector<double>& weights);

    VosesAliasMethod(const VosesAliasMethod&) = delete;

    std::int64_t sample();
    std::int64_t operator()();

  private:
    void init(const std::vector<double>& weights);

    std::int64_t                    n_weights_;
    std::unique_ptr<std::int64_t[]> alias_;
    std::unique_ptr<double[]>       probs_;
    // uniform random number generator
    uniform_distribution<double> rand_{0.0, 1.0};
};

}  // namespace math::random
