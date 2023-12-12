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

    inline auto get_now() const;

    inline void reset();

    inline auto elapsed() const;

    template <typename SleepDurationType>
    inline void sleep(std::chrono::seconds::rep duration) const;

    inline void print_elapsed_seconds(std::uint8_t n_decimals = get_num_decimals()) const;

  private:
    static constexpr std::uint8_t get_num_decimals();

    std::chrono::seconds::rep timestamp_;
};

template <typename DurationType>
Timer<DurationType>::Timer()
  : timestamp_{get_now()} {}

template <typename DurationType>
auto Timer<DurationType>::get_now() const {
    return std::chrono::duration_cast<DurationType>(ClockType::now().time_since_epoch()).count();
}

template <typename DurationType>
void Timer<DurationType>::reset() {
    timestamp_ = get_now();
}

template <typename DurationType>
auto Timer<DurationType>::elapsed() const {
    return get_now() - timestamp_;
}

template <typename DurationType>
template <typename SleepDurationType>
void Timer<DurationType>::sleep(std::chrono::seconds::rep duration) const {
    std::this_thread::sleep_for(static_cast<SleepDurationType>(duration));
}

template <typename DurationType>
void Timer<DurationType>::print_elapsed_seconds(std::uint8_t n_decimals) const {
    const auto elapsed = this->elapsed();

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

template <typename DurationType>
constexpr std::uint8_t Timer<DurationType>::get_num_decimals() {
    if constexpr (std::is_same_v<DurationType, std::chrono::seconds>) {
        return 0;

    } else if constexpr (std::is_same_v<DurationType, std::chrono::milliseconds>) {
        return 3;

    } else if constexpr (std::is_same_v<DurationType, std::chrono::microseconds>) {
        return 6;

    } else if constexpr (std::is_same_v<DurationType, std::chrono::nanoseconds>) {
        return 9;

    } else {
        // Default number of decimals for unsupported types
        return 3;
    }
}

}  // namespace ffcl::common

/*
void example() {
    // set a timer object with a millisecond time precision
    ffcl::common::Timer<ffcl::common::Milliseconds> timer;
    // set the amount of time to wait w.r.t. the time unit. Accepts only integer so choose the right unit depending on
    // the precision needed: Seconds, Milliseconds, Microseconds, Nanoseconds
    timer.sleep<ffcl::common::Milliseconds>(4500);
    timer.print_elapsed_seconds(9);
}
*/