#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <chrono>
#include <memory>
#include <random>
#include <thread>

namespace math::random {

/**
 * @brief seed that should be provided only as rvalue
 *
 * @return std::mt19937*
 */
static inline std::mt19937* seed() {
#if defined(_OPENMP) && THREADS_ENABLED == true
    auto thread_num = omp_get_thread_num();
#endif
    static thread_local std::unique_ptr<std::mt19937> instance;
    if (!instance) {
        const auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
#if defined(_OPENMP) && THREADS_ENABLED == true
        instance = std::make_unique<std::mt19937>(seed + thread_num);
#else
        instance = std::make_unique<std::mt19937>(seed);
#endif
    }
    return instance.get();
}
/**
 * @brief Generates random numbers with a normal distribution.
 *
 * @tparam T The type of numbers to generate.
 */
template <typename T>
class uniform_distribution {
  public:
    uniform_distribution(T inf, T sup)
      : distribution_(inf, sup) {}

    inline T operator()() {
        return distribution_(*seed());
    }

  private:
    template <typename dType>
    using UniformDistributionType = std::conditional_t<std::is_integral<dType>::value,
                                                       std::uniform_int_distribution<dType>,
                                                       std::uniform_real_distribution<dType> >;
    UniformDistributionType<T> distribution_;
};
/**
 * @brief
 * std::uniform_int_distribution<> d(n, p)
 * n: number of trials
 * p: success rate (e.g.: 0.5)
 *
 * std::normal_distribution<> d(mu, sd)
 * mu: mean of the distr.
 * sd: variance of the distr.
 * @tparam T
 */
template <typename T>
class normal_distribution {
  public:
    normal_distribution(T mean, double sd)
      : distribution_(mean, sd) {}

    inline T operator()() {
        return distribution_(*seed());
    }

  private:
    template <typename dType>
    using NormalDistributionType = std::conditional_t<std::is_integral<dType>::value,
                                                      std::binomial_distribution<dType>,
                                                      std::normal_distribution<dType> >;
    NormalDistributionType<T> distribution_;
};

}  // namespace math::random