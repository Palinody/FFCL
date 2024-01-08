#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/random/Distributions.hpp"

#include <map>

template <typename DataType>
void print_sequence(std::size_t sequence_length) {
    ffcl::common::math::random::normal_distribution<DataType> uniform_distribution_number_generator(5, 0.5);

    for (std::size_t i = 0; i < sequence_length; ++i) {
        std::cout << uniform_distribution_number_generator() << " ";
    }
    std::cout << "\n";
}

template <typename DataType>
void print_sequence(std::size_t sequence_length, std::size_t seed) {
    ffcl::common::math::random::normal_distribution<DataType> uniform_distribution_number_generator(5, 0.5);

    for (std::size_t i = 0; i < sequence_length; ++i) {
        std::cout << uniform_distribution_number_generator(seed) << " ";
    }
    std::cout << "\n";
}

template <typename DataType>
void print_thread_sequence(std::size_t thread_id, std::size_t sequence_length) {
    ffcl::common::math::random::normal_distribution<DataType> uniform_distribution_number_generator(5, 0.5);

    // std::cout << thread_id << ": ";

    for (std::size_t i = 0; i < sequence_length; ++i) {
        std::cout << uniform_distribution_number_generator() << " ";
    }
    std::cout << "\n";
}

template <typename DataType>
void print_thread_sequence(std::size_t thread_id, std::size_t sequence_length, std::size_t seed) {
    ffcl::common::math::random::normal_distribution<DataType> uniform_distribution_number_generator(5, 0.5);

    // std::cout << thread_id << ": ";
    static_cast<void>(thread_id);

    for (std::size_t i = 0; i < sequence_length; ++i) {
        std::cout << uniform_distribution_number_generator(seed) << " ";
    }
    std::cout << "\n";
}

template <typename DataType>
void map_hist() {
    ffcl::common::math::random::normal_distribution<DataType> uniform_distribution_number_generator(10, 0.1);

    std::map<DataType, DataType> hist;

    std::size_t n_values = 1e4;

    for (std::size_t n = 0; n != n_values; ++n)
        ++hist[uniform_distribution_number_generator()];

    for (auto const& [x, y] : hist)
        std::cout << x << ' ' << std::string(y / 100, '*') << '\n';
}

template <typename DataType>
void thread_func(std::size_t thread_id) {
    ffcl::common::math::random::normal_distribution<DataType> uniform_distribution_number_generator(10, 0.1);

    for (int i = 0; i < 10; ++i) {
        std::cout << "Thread " << thread_id << ": " << uniform_distribution_number_generator() << "\n";
    }
}

TEST(Test, STDThreadsTest) {
    constexpr std::size_t sequence_length = 1;
    constexpr std::size_t seed            = 0;

#if !defined(_OPENMP) && THREADS_ENABLED == true
    const std::size_t num_threads = 4;

    std::thread threads[num_threads];

    for (std::size_t i = 0; i < num_threads; ++i) {
        threads[i] =
            std::thread(static_cast<void (*)(std::size_t, std::size_t, std::size_t)>(print_thread_sequence<float>),
                        i,
                        sequence_length,
                        seed);

        // threads[i] = std::thread(static_cast<void (*)(std::size_t, std::size_t)>(print_thread_sequence<float>), i,
        // 1sequence_length);
    }

    for (std::size_t i = 0; i < num_threads; ++i) {
        threads[i].join();
    }
#else
    print_sequence<float>(sequence_length, seed);
#endif
}

TEST(Test2, OpenMPThreadsTest) {
    constexpr std::size_t sequence_length = 1;
    constexpr std::size_t seed            = 0;

#if defined(_OPENMP) && THREADS_ENABLED == true

    const std::size_t num_threads = 4;

#pragma omp parallel num_threads(num_threads)
    {
        std::size_t tid = omp_get_thread_num();
        print_thread_sequence<float>(tid, sequence_length, seed);
    }
#else
    print_sequence<float>(sequence_length, seed);
#endif
}

/*
#include "ffcl/common/math/random/Sampling.hpp"

#include <chrono>
#include <iostream>
#include <utility>

template <typename Func, typename... Args>
auto measure_execution_time(Func&& func, Args&&... args) {
    using namespace std::chrono;

    auto start = high_resolution_clock::now();
    std::forward<Func>(func)(std::forward<Args>(args)...);  // Forwarding the function and its arguments
    auto end = high_resolution_clock::now();

    return duration_cast<microseconds>(end - start).count();  // You can change the time unit here
}

TEST(Test2, SelectFromRangeTest) {
    for (std::size_t range_size = 9990; range_size < 10000; ++range_size) {
        std::pair<std::size_t, std::size_t> indices_range = {0, range_size};
        for (std::size_t n_choices = 9990; n_choices < range_size; ++n_choices) {
            auto elapsed_time = measure_execution_time(
                ffcl::common::math::random::select_from_range_deprecated, n_choices, indices_range);

            std::cout << "range_size, n_choices: " << range_size << ", " << n_choices << " | " << elapsed_time
                      << " microseconds\n";
        }
    }

    const auto indices = ffcl::common::math::random::select_from_range(15, {5, 20});

    for (const auto& index : indices) {
        std::cout << index << ", ";
    }
    std::cout << "\n";
}
*/

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}