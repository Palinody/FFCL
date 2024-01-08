#pragma once

#include "ffcl/common/math/random/Distributions.hpp"

#include <sys/types.h>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

// For loaded integer, use: https://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

namespace ffcl::common::math::random {

template <typename Weight = float>
class VosesAliasMethod {
    static_assert(std::is_trivial_v<Weight>, "Weight must be trivial.");

  public:
    using FloatType = std::conditional_t<std::is_floating_point<Weight>::value, Weight, float>;

    explicit VosesAliasMethod(const std::vector<Weight>& weights);

    VosesAliasMethod(const VosesAliasMethod&) = delete;

    inline std::size_t sample();

    inline std::size_t operator()();

  private:
    void init(const std::vector<Weight>& weights);

    std::size_t                    n_weights_;
    std::unique_ptr<std::size_t[]> alias_;
    std::unique_ptr<FloatType[]>   probabilities_;
    // uniform random number generator
    uniform_distribution<FloatType> rand_;
};

template <typename Weight>
VosesAliasMethod<Weight>::VosesAliasMethod(const std::vector<Weight>& weights)
  : n_weights_{static_cast<std::size_t>(weights.size())}
  , alias_{std::make_unique<std::size_t[]>(n_weights_)}
  , probabilities_{std::make_unique<FloatType[]>(n_weights_)}
  , rand_{0, 1} {
    // the input weights vector shouldn't be empty (wouldnt make sense anyway)
    if (weights.empty()) {
        throw std::invalid_argument("Weights distribution vector shouldn't be empty.\n");
    }
    // initialize the lookup tables
    init(weights);
}

template <typename Weight>
void VosesAliasMethod<Weight>::init(const std::vector<Weight>& weights) {
    // normalized probabilities that do not sum to one
    auto fake_probabilities = std::make_unique<FloatType[]>(n_weights_);
    auto large              = std::make_unique<std::size_t[]>(n_weights_);
    auto small              = std::make_unique<std::size_t[]>(n_weights_);

    std::size_t n_small = 0;
    std::size_t n_large = 0;

    // normalization factor: n_weights / sum(weights)
    const FloatType norm_fact = n_weights_ / std::accumulate(weights.begin(), weights.end(), static_cast<FloatType>(0));
    // renormalization of the weights (sum(fake_probabilities) != 1, it's normal)
    std::transform(weights.begin(),
                   weights.end(),
                   fake_probabilities.get(),
                   std::bind(std::multiplies<FloatType>(), std::placeholders::_1, norm_fact));

    // Use k as a size then shift by one in the loop body to get the index so that an unsighed type can be used.
    for (std::size_t weight_index = n_weights_; weight_index > 0; --weight_index) {
        if (fake_probabilities[weight_index - 1] < static_cast<FloatType>(1)) {
            small[n_small++] = weight_index - 1;

        } else {
            large[n_large++] = weight_index - 1;
        }
    }
    while (n_small && n_large) {
        const std::size_t small_index = small[--n_small];
        const std::size_t large_index = large[--n_large];

        probabilities_[small_index] = fake_probabilities[small_index];
        alias_[small_index]         = large_index;
        fake_probabilities[large_index] += fake_probabilities[small_index] - 1;

        if (fake_probabilities[large_index] < static_cast<FloatType>(1)) {
            small[n_small++] = large_index;

        } else {
            large[n_large++] = large_index;
        }
    }
    while (n_large) {
        probabilities_[large[--n_large]] = static_cast<FloatType>(1);
    }
    while (n_small) {
        probabilities_[small[--n_small]] = static_cast<FloatType>(1);
    }
}

template <typename Weight>
std::size_t VosesAliasMethod<Weight>::sample() {
    const auto random_index = static_cast<std::size_t>(n_weights_ * rand_());
    return rand_() < probabilities_[random_index] ? random_index : alias_[random_index];
}

template <typename Weight>
std::size_t VosesAliasMethod<Weight>::operator()() {
    return sample();
}

}  // namespace ffcl::common::math::random
