#pragma once

#include <stdio.h>
#include <chrono>
#include <thread>

#if defined(_OPENMP) && THREADS_ENABLED == true
#include <omp.h>
#endif

namespace ffcl::common {

using Seconds      = std::chrono::seconds;
using Milliseconds = std::chrono::milliseconds;
using Microseconds = std::chrono::microseconds;
using Nanoseconds  = std::chrono::nanoseconds;

template <typename DurationType = std::chrono::seconds>
class Timer {
    using ClockType = std::chrono::system_clock;

  public:
    Timer();

    inline std::uint64_t get_now() const;

    inline void reset();

    inline std::uint64_t elapsed() const;

    template <typename SleepDurationType>
    inline void sleep(std::uint64_t duration) const;

    inline void print_elapsed_seconds(std::uint8_t n_decimals = 3) const;

  private:
    std::uint64_t timestamp_;
};

template <typename DurationType>
Timer<DurationType>::Timer()
  : timestamp_{get_now()} {}

template <typename DurationType>
std::uint64_t Timer<DurationType>::get_now() const {
    return std::chrono::duration_cast<DurationType>(ClockType::now().time_since_epoch()).count();
}

template <typename DurationType>
void Timer<DurationType>::reset() {
    timestamp_ = get_now();
}

template <typename DurationType>
std::uint64_t Timer<DurationType>::elapsed() const {
    return get_now() - timestamp_;
}

template <typename DurationType>
template <typename SleepDurationType>
void Timer<DurationType>::sleep(std::uint64_t duration) const {
    std::this_thread::sleep_for(static_cast<SleepDurationType>(duration));
}

template <typename DurationType>
void Timer<DurationType>::print_elapsed_seconds(std::uint8_t n_decimals) const {
    const std::uint64_t elapsed = this->elapsed();

    if constexpr (std::is_same_v<DurationType, Seconds>) {
        printf("Elapsed (s): %.*f\n", n_decimals, static_cast<float>(elapsed));

    } else if constexpr (std::is_same_v<DurationType, Milliseconds>) {
        printf("Elapsed (s): %.*f\n", n_decimals, (elapsed * 1e-3f));

    } else if constexpr (std::is_same_v<DurationType, Microseconds>) {
        printf("Elapsed (s): %.*f\n", n_decimals, (elapsed * 1e-6f));

    } else if constexpr (std::is_same_v<DurationType, Nanoseconds>) {
        printf("Elapsed (s): %.*f\n", n_decimals, (elapsed * 1e-9f));

    } else {
        printf("[WARN] DurationType not supported, default: Milliseconds\n");
        printf("Elapsed (s): %.*f\n", n_decimals, (elapsed * 1e-3f));
    }
}

}  // namespace ffcl::common