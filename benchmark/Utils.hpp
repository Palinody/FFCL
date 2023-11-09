#pragma once

#include "ffcl/common/Utils.hpp"

#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

namespace utils {

void print_progress_bar(std::size_t current, std::size_t total, std::size_t bar_width = 40) {
    float       progress = static_cast<float>(current) / total;
    std::size_t position = static_cast<std::size_t>(bar_width * progress);

    std::cout << "[";
    for (std::size_t i = 0; i < bar_width; ++i) {
        if (i < position)
            std::cout << "=";
        else if (i == position)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setw(6) << std::setprecision(2) << (progress * 100.0) << "%";

    // carriage return to go back to the beginning of the line
    std::cout << '\r';
    std::cout.flush();
}

struct DurationsSummary {
    DurationsSummary() = default;

    DurationsSummary(const DurationsSummary& other) = default;

    DurationsSummary& operator=(const DurationsSummary& other) {
        if (this != &other) {  // Check for self-assignment
            n_samples              = other.n_samples;
            n_features             = other.n_features;
            indexer_build_duration = other.indexer_build_duration;
            indexer_query_duration = other.indexer_query_duration;
            total_duration         = other.total_duration;
        }
        return *this;
    }

    DurationsSummary& operator+=(const DurationsSummary& other) {
        n_samples += other.n_samples;
        n_features += other.n_features;
        indexer_build_duration += other.indexer_build_duration;
        indexer_query_duration += other.indexer_query_duration;
        total_duration += other.total_duration;
        return *this;
    }

    DurationsSummary& operator-=(const DurationsSummary& other) {
        n_samples -= other.n_samples;
        n_features -= other.n_features;
        indexer_build_duration -= other.indexer_build_duration;
        indexer_query_duration -= other.indexer_query_duration;
        total_duration -= other.total_duration;
        return *this;
    }

    DurationsSummary& operator/=(long double divisor) {
        if (ffcl::common::utils::inequality(divisor, 0)) {
            n_samples /= divisor;
            n_features /= divisor;
            indexer_build_duration /= divisor;
            indexer_query_duration /= divisor;
            total_duration /= divisor;
        }
        return *this;
    }

    DurationsSummary& apply_timer_multiplier(long double multiplier) {
        indexer_build_duration *= multiplier;
        indexer_query_duration *= multiplier;
        total_duration *= multiplier;
        return *this;
    }

    DurationsSummary& operator*=(const DurationsSummary& other) {
        n_samples *= other.n_samples;
        n_features *= other.n_features;
        indexer_build_duration *= other.indexer_build_duration;
        indexer_query_duration *= other.indexer_query_duration;
        total_duration *= other.total_duration;
        return *this;
    }

    DurationsSummary operator*(const DurationsSummary& summary) {
        DurationsSummary result;
        result.n_samples              = n_samples * summary.n_samples;
        result.n_features             = n_features * summary.n_features;
        result.indexer_build_duration = indexer_build_duration * summary.indexer_build_duration;
        result.indexer_query_duration = indexer_query_duration * summary.indexer_query_duration;
        result.total_duration         = total_duration * summary.total_duration;
        return result;
    }

    long double n_samples              = 0;
    long double n_features             = 0;
    long double indexer_build_duration = 0;
    long double indexer_query_duration = 0;
    long double total_duration         = 0;
};

std::vector<std::size_t> generate_indices(std::size_t n_samples) {
    std::vector<std::size_t> elements(n_samples);
    std::iota(elements.begin(), elements.end(), static_cast<std::size_t>(0));
    return elements;
}

template <typename SamplesIterator>
void cartesian_to_polar_inplace(const SamplesIterator& samples_first,
                                const SamplesIterator& samples_last,
                                std::size_t            n_features) {
    const std::size_t n_samples = ffcl::common::utils::get_n_samples(samples_first, samples_last, n_features);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        const auto x = samples_first[sample_index * n_features];
        const auto y = samples_first[sample_index * n_features + 1];
        // to radius
        samples_first[sample_index * n_features] = std::sqrt(x * x + y * y);
        // to angle
        samples_first[sample_index * n_features + 1] = std::atan2(y, x);
    }
}

template <typename Type>
void print_data(const std::vector<Type>& data, std::size_t n_features) {
    if (!n_features) {
        return;
    }
    const std::size_t n_samples = data.size() / n_features;

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            std::cout << data[sample_index * n_features + feature_index] << " ";
        }
        std::cout << "\n";
    }
}

}  // namespace utils