// Shared harness for the reference-library benchmarks: dataset loading, the timing policy and the
// JSON emission. Each library gets its own executable: pdbscan and pargeo hdbscan vendor different
// copies of the parlay scheduler (linking both segfaults), and linking mlpack/armadillo next to
// nanoflann measurably slows the nanoflann hot loop.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace refbench {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kTries          = 3;
constexpr std::size_t kNQueries       = 1000;
constexpr std::size_t kK              = 2;
constexpr double      kRadius         = 0.5;
constexpr std::size_t kMinSamples     = 5;
constexpr std::size_t kLeaf           = 40;

struct Dataset {
    std::string         name;
    std::size_t         n_features = 0;
    std::size_t         n_samples  = 0;
    std::vector<double> data;  // row major
};

inline auto load_dataset(const std::string& dir, const std::string& name) -> Dataset {
    auto dataset = Dataset{};
    dataset.name = name;

    auto stream = std::ifstream(dir + "/" + name + ".txt");
    if (!stream.is_open()) {
        throw std::runtime_error("cannot open " + dir + "/" + name + ".txt");
    }
    auto line = std::string{};
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        auto values = std::vector<double>{};
        auto item   = std::stringstream{line};
        auto value  = 0.0;
        while (item >> value) values.push_back(value);
        if (values.empty()) continue;
        dataset.n_features = values.size();
        dataset.data.insert(dataset.data.end(), values.begin(), values.end());
    }
    dataset.n_samples = dataset.data.size() / dataset.n_features;
    return dataset;
}

inline auto all_dataset_names() -> const std::vector<std::string>& {
    static const auto names = std::vector<std::string>{"noisy_circles", "noisy_moons", "varied", "aniso",
                                                       "blobs",         "no_structure", "unbalanced_blobs"};
    return names;
}

inline void emit(const char* library, const char* algorithm, const char* stage, const std::string& dataset,
                 std::size_t n, std::size_t leaf, std::size_t threads, std::size_t tries, double mean,
                 double median, const std::string& note) {
    std::printf(
        "{\"library\": \"%s\", \"algorithm\": \"%s\", \"stage\": \"%s\", \"dataset\": \"%s\", \"n\": %zu, "
        "\"leaf\": %zu, \"threads\": %zu, \"tries\": %zu, \"mean\": %.9f, \"median\": %.9f, \"note\": \"%s\"}\n",
        library, algorithm, stage, dataset.c_str(), n, leaf, threads, tries, mean, median, note.c_str());
}

// Runs `workload` kTries times and emits mean and median, like the project's own harness.
inline void measure(const char* library, const char* algorithm, const char* stage, const Dataset& dataset,
                    std::size_t leaf, std::size_t threads, const std::function<void()>& workload,
                    const std::string& note = {}) {
    auto times = std::vector<double>{};
    for (std::size_t try_index = 0; try_index < kTries; ++try_index) {
        const auto start = Clock::now();
        workload();
        times.push_back(std::chrono::duration<double>(Clock::now() - start).count());
    }
    std::sort(times.begin(), times.end());
    const auto mean = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    emit(library, algorithm, stage, dataset.name, dataset.n_samples, leaf, threads, kTries, mean,
         times[times.size() / 2], note);
}

struct Options {
    std::string datasets_dir;
    std::string library;
    std::string workload;
    std::size_t threads = 1;
};

inline auto parse_options(int argc, char** argv) -> Options {
    auto options = Options{};
    for (int index = 1; index < argc; ++index) {
        const auto argument = std::string{argv[index]};
        if (argument == "--datasets" && index + 1 < argc) options.datasets_dir = argv[++index];
        else if (argument == "--library" && index + 1 < argc) options.library = argv[++index];
        else if (argument == "--workload" && index + 1 < argc) options.workload = argv[++index];
        else if (argument == "--threads" && index + 1 < argc) options.threads = std::stoul(argv[++index]);
    }
    return options;
}

// Runs `body` for every dataset in the fixed order and returns 0, or 1 when the library was not
// compiled in. `body` receives the loaded dataset.
template <class Body>
auto for_each_dataset(const Options& options, Body&& body) -> int {
    for (const auto& name : all_dataset_names()) {
        const auto dataset = load_dataset(options.datasets_dir, name);
        body(dataset);
    }
    return 0;
}

inline void announce(const Options& options) {
    std::printf("{\"info\": \"library=%s workload=%s threads=%zu\"}\n", options.library.c_str(),
                options.workload.c_str(), options.threads);
}

}  // namespace refbench
