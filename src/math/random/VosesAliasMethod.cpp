#include "cpp_clustering/math/random/VosesAliasMethod.hpp"

namespace math::random {

VosesAliasMethod::VosesAliasMethod(const std::vector<double>& weights)
  : n_weights_{static_cast<std::int64_t>(weights.size())}
  , alias_{std::make_unique<std::int64_t[]>(n_weights_)}
  , probs_{std::make_unique<double[]>(n_weights_)} {
    // the input weights vector shouldn't be empty (wouldnt make sense anyway)
    if (weights.empty()) {
        throw std::invalid_argument("Weights distribution vector shouldn't be empty.\n");
    }
    // initialize the lookup tables
    init(weights);
}

void VosesAliasMethod::init(const std::vector<double>& weights) {
    // normalized probabilities that do not sum to one
    auto fake_probs = std::make_unique<double[]>(n_weights_);
    auto large      = std::make_unique<std::int64_t[]>(n_weights_);
    auto small      = std::make_unique<std::int64_t[]>(n_weights_);

    std::int64_t n_small = 0, n_large = 0;

    // normalization factor: n_weights / sum(weights)
    const double norm_fact = n_weights_ / std::accumulate(weights.begin(), weights.end(), 0.0);
    // renormalization of the weights (sum(fake_probs) != 1, it's normal)
    std::transform(weights.begin(),
                   weights.end(),
                   fake_probs.get(),
                   std::bind(std::multiplies<double>(), std::placeholders::_1, norm_fact));

    for (std::int64_t k = n_weights_ - 1; k >= 0; --k) {
        if (fake_probs[k] < 1.0) {
            small[n_small++] = k;
        } else {
            large[n_large++] = k;
        }
    }
    while (n_small && n_large) {
        const std::int64_t small_idx = small[--n_small];
        const std::int64_t large_idx = large[--n_large];

        probs_[small_idx] = fake_probs[small_idx];
        alias_[small_idx] = large_idx;
        fake_probs[large_idx] += fake_probs[small_idx] - 1;

        if (fake_probs[large_idx] < 1.0) {
            small[n_small++] = large_idx;
        } else {
            large[n_large++] = large_idx;
        }
    }
    while (n_large) {
        probs_[large[--n_large]] = 1.0;
    }
    while (n_small) {
        probs_[small[--n_small]] = 1.0;
    }
}

std::int64_t VosesAliasMethod::sample() {
    const auto k = static_cast<std::int64_t>(n_weights_ * rand_());
    return rand_() < probs_[k] ? k : alias_[k];
}

std::int64_t VosesAliasMethod::operator()() {
    return sample();
}

}  // namespace math::random
