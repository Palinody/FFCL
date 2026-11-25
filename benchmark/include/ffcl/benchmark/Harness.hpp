#pragma once

// Shared harness for the FFCL benchmark.
//
// Every measurement is one JSON line, so the FFCL numbers and the reference-library numbers (produced
// by benchmark/scripts/) share a format and can be merged into the tables by
// benchmark/scripts/make_tables.py.
//
// Timing policy, applied to every single measurement: run the workload `kTries` times and report both
// the mean and the median. Three tries keeps a full run short while still being able to spot a first
// iteration that pays for lazy initialisation or a cold cache.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace ffcl::benchmark {

inline constexpr std::size_t kTries = 3;

struct Measurement {
    std::string library;
    std::string algorithm;
    std::string stage;
    std::string dataset;
    std::size_t n       = 0;
    std::size_t leaf    = 0;
    std::size_t threads = 1;
    std::size_t tries   = kTries;
    double      mean    = 0.0;
    double      median  = 0.0;
    // Anything the reader needs to judge comparability: metric, parameters, notes.
    std::string note;
};

// Runs `workload` kTries times and returns the mean and the median wall-clock duration in seconds.
inline auto time_workload(const std::function<void()>& workload, std::size_t tries = kTries)
    -> std::pair<double, double> {
    auto samples = std::vector<double>{};
    samples.reserve(tries);

    for (std::size_t try_index = 0; try_index < tries; ++try_index) {
        const auto start = std::chrono::steady_clock::now();
        workload();
        const auto stop = std::chrono::steady_clock::now();

        samples.emplace_back(std::chrono::duration<double>(stop - start).count());
    }
    std::sort(samples.begin(), samples.end());

    const auto mean   = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    const auto median = samples.size() % 2 == 1
                            ? samples[samples.size() / 2]
                            : 0.5 * (samples[samples.size() / 2 - 1] + samples[samples.size() / 2]);

    return {mean, median};
}

inline auto json_escape(const std::string& value) -> std::string {
    auto escaped = std::string{};
    for (const auto character : value) {
        switch (character) {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            default:
                escaped += character;
        }
    }
    return escaped;
}

inline void emit_json_line(std::ostream& stream, const Measurement& measurement) {
    stream << std::fixed << std::setprecision(6) << "{\"library\": \"" << json_escape(measurement.library)
           << "\", \"algorithm\": \"" << json_escape(measurement.algorithm) << "\", \"stage\": \""
           << json_escape(measurement.stage) << "\", \"dataset\": \"" << json_escape(measurement.dataset)
           << "\", \"n\": " << measurement.n << ", \"leaf\": " << measurement.leaf
           << ", \"threads\": " << measurement.threads << ", \"tries\": " << measurement.tries
           << ", \"mean\": " << measurement.mean << ", \"median\": " << measurement.median << ", \"note\": \""
           << json_escape(measurement.note) << "\"}\n";
}

// Times `workload` and writes the resulting measurement to `stream`.
inline void measure_and_emit(std::ostream&               stream,
                            const Measurement&          template_measurement,
                            const std::function<void()>& workload) {
    auto measurement = template_measurement;

    const auto [mean, median] = time_workload(workload, measurement.tries);

    measurement.mean   = mean;
    measurement.median = median;

    emit_json_line(stream, measurement);
}

}  // namespace ffcl::benchmark
