#pragma once

#include <stdio.h>
#include <chrono>
#include <thread>

#if defined(_OPENMP) && THREADS_ENABLED == true
#include <omp.h>
#endif

namespace common::timer {

using Seconds      = std::chrono::seconds;
using Milliseconds = std::chrono::milliseconds;
using Microseconds = std::chrono::microseconds;
using Nanoseconds  = std::chrono::nanoseconds;

template <typename DurationType = std::chrono::seconds>
class Timer {
    using clock_t = std::chrono::system_clock;

  public:
    Timer();

    inline std::uint64_t getNow();

    inline void reset();

    inline std::uint64_t elapsed();

    template <typename SleepDurationType>
    inline void sleep(std::uint64_t duration);

    inline void print_elapsed_seconds(const std::uint8_t n_decimals = 3);

  private:
    std::uint64_t now_;
};

template <typename DurationType>
Timer<DurationType>::Timer()
  : now_{getNow()} {}

template <typename DurationType>
std::uint64_t Timer<DurationType>::getNow() {
    return std::chrono::duration_cast<DurationType>(clock_t::now().time_since_epoch()).count();
}

template <typename DurationType>
void Timer<DurationType>::reset() {
    now_ = getNow();
}

template <typename DurationType>
std::uint64_t Timer<DurationType>::elapsed() {
    return getNow() - now_;
}

template <typename DurationType>
template <typename SleepDurationType>
void Timer<DurationType>::sleep(std::uint64_t duration) {
    std::this_thread::sleep_for(static_cast<SleepDurationType>(duration));
}

template <typename DurationType>
void Timer<DurationType>::print_elapsed_seconds(const std::uint8_t n_decimals) {
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

}  // namespace common::timer