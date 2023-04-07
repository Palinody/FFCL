#pragma once

#if defined(_OPENMP) && THREADS_ENABLED == true
#include <omp.h>
#elif !defined(_OPENMP) && THREADS_ENABLED == true
#include <thread>
#endif

#include <memory>
#include <random>

namespace math::random {

static inline std::mt19937& thread_local_mersenne_engine() {
    static thread_local std::random_device rnd_device;
    // static thread_local std::mt19937 mersienne_engine_thread_instance{rnd_device()};
    static thread_local std::mt19937 mersienne_engine_thread_instance(rnd_device());

    return mersienne_engine_thread_instance;
}

static inline std::mt19937& thread_local_mersenne_engine(std::size_t value) {
#if defined(_OPENMP) && THREADS_ENABLED == true
    value += omp_get_thread_num();

#elif !defined(_OPENMP) && THREADS_ENABLED == true
    value += std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
    static thread_local std::mt19937 mersienne_engine_thread_instance(value);

    return mersienne_engine_thread_instance;
}

/**
 * @brief Generates random numbers with a normal distribution.
 *
 * @tparam DataType The type of numbers to generate.
 */
template <typename DataType>
class uniform_distribution {
  public:
    uniform_distribution(DataType lower_bound, DataType upper_bound)
      : distribution_(lower_bound, upper_bound) {}

    inline DataType operator()() {
        return distribution_(thread_local_mersenne_engine());
    }

    inline DataType operator()(std::size_t value) {
        return distribution_(thread_local_mersenne_engine(value));
    }

  private:
    using UniformDistributionType = std::conditional_t<std::is_integral<DataType>::value,
                                                       std::uniform_int_distribution<DataType>,
                                                       std::uniform_real_distribution<DataType>>;
    UniformDistributionType distribution_;
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
 * @tparam DataType
 */
template <typename DataType>
class normal_distribution {
  public:
    normal_distribution(DataType mean, DataType sd)
      : distribution_(mean, sd) {}

    normal_distribution(DataType n_trials, double success_rate)
      : distribution_(n_trials, success_rate) {}

    inline DataType operator()() {
        return distribution_(thread_local_mersenne_engine());
    }

    inline DataType operator()(std::size_t value) {
        return distribution_(thread_local_mersenne_engine(value));
    }

  private:
    using NormalDistributionType = std::conditional_t<std::is_integral<DataType>::value,
                                                      std::binomial_distribution<DataType>,
                                                      std::normal_distribution<DataType>>;

    NormalDistributionType distribution_;
};

}  // namespace math::random