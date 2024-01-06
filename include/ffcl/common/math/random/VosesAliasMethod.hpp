#pragma once

#include "ffcl/common/math/random/Distributions.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

// For loaded integer, use: https://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

namespace ffcl::common::math::random {

template <typename FloatType = float>
class VosesAliasMethod {
    static_assert(std::is_floating_point_v<FloatType>, "Invalid type specified. Should be float.");

  public:
    explicit VosesAliasMethod(const std::vector<FloatType>& weights);

    VosesAliasMethod(const VosesAliasMethod&) = delete;

    inline std::int64_t sample();

    inline std::int64_t operator()();

  private:
    void init(const std::vector<FloatType>& weights);

    std::int64_t                    n_weights_;
    std::unique_ptr<std::int64_t[]> alias_;
    std::unique_ptr<FloatType[]>    probabilities_;
    // uniform random number generator
    uniform_distribution<FloatType> rand_;
};

template <typename FloatType>
VosesAliasMethod<FloatType>::VosesAliasMethod(const std::vector<FloatType>& weights)
  : n_weights_{static_cast<std::int64_t>(weights.size())}
  , alias_{std::make_unique<std::int64_t[]>(n_weights_)}
  , probabilities_{std::make_unique<FloatType[]>(n_weights_)}
  , rand_{static_cast<FloatType>(0), static_cast<FloatType>(1)} {
    // the input weights vector shouldn't be empty (wouldnt make sense anyway)
    if (weights.empty()) {
        throw std::invalid_argument("Weights distribution vector shouldn't be empty.\n");
    }
    // initialize the lookup tables
    init(weights);
}

template <typename FloatType>
void VosesAliasMethod<FloatType>::init(const std::vector<FloatType>& weights) {
    // normalized probabilities that do not sum to one
    auto fake_probabilities = std::make_unique<FloatType[]>(n_weights_);
    auto large              = std::make_unique<std::int64_t[]>(n_weights_);
    auto small              = std::make_unique<std::int64_t[]>(n_weights_);

    std::int64_t n_small = 0, n_large = 0;

    // normalization factor: n_weights / sum(weights)
    const FloatType norm_fact = n_weights_ / std::accumulate(weights.begin(), weights.end(), static_cast<FloatType>(0));
    // renormalization of the weights (sum(fake_probabilities) != 1, it's normal)
    std::transform(weights.begin(),
                   weights.end(),
                   fake_probabilities.get(),
                   std::bind(std::multiplies<FloatType>(), std::placeholders::_1, norm_fact));

    for (std::int64_t k = n_weights_ - 1; k >= 0; --k) {
        if (fake_probabilities[k] < static_cast<FloatType>(1)) {
            small[n_small++] = k;

        } else {
            large[n_large++] = k;
        }
    }
    while (n_small && n_large) {
        const std::int64_t small_index = small[--n_small];
        const std::int64_t large_index = large[--n_large];

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

template <typename FloatType>
std::int64_t VosesAliasMethod<FloatType>::sample() {
    const auto random_index = static_cast<std::int64_t>(n_weights_ * rand_());
    return rand_() < probabilities_[random_index] ? random_index : alias_[random_index];
}

template <typename FloatType>
std::int64_t VosesAliasMethod<FloatType>::operator()() {
    return sample();
}

}  // namespace ffcl::common::math::random
