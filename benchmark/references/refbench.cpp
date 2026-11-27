// Reference-library benchmark. Emits one JSON line per measurement, in the same schema as the
// project's own harness so the two can be merged by the table generator.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#if defined(HAVE_NANOFLANN)
#include <nanoflann.hpp>
#endif

#if defined(HAVE_FLANN)
#include <flann/flann.hpp>
#endif

#if defined(HAVE_ARBORX)
#include <ArborX.hpp>
#include <Kokkos_Core.hpp>
#endif

using Clock = std::chrono::steady_clock;

namespace {

constexpr std::size_t kTries = 3;
constexpr std::size_t kNQueries = 1000;
constexpr std::size_t kK = 2;
constexpr double kRadius = 0.5;
constexpr std::size_t kMinSamples = 5;
constexpr std::size_t kMinClusterSize = 20;
constexpr std::size_t kLeaf = 40;

struct Dataset {
    std::string        name;
    std::size_t        n_features = 0;
    std::size_t        n_samples  = 0;
    std::vector<double> data;  // row major
};

auto load_dataset(const std::string& dir, const std::string& name) -> Dataset {
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

auto all_dataset_names() -> const std::vector<std::string>& {
    static const auto names = std::vector<std::string>{"noisy_circles", "noisy_moons", "varied", "aniso",
                                                       "blobs",         "no_structure", "unbalanced_blobs"};
    return names;
}

auto geometric_mean(const std::vector<double>& values) -> double {
    auto total = 0.0;
    for (const auto value : values) total += std::log(value);
    return std::exp(total / static_cast<double>(values.size()));
}

void emit(const char* library, const char* algorithm, const char* stage, const std::string& dataset,
          std::size_t n, std::size_t leaf, std::size_t threads, std::size_t tries, double mean,
          double median, const std::string& note) {
    std::printf(
        "{\"library\": \"%s\", \"algorithm\": \"%s\", \"stage\": \"%s\", \"dataset\": \"%s\", \"n\": %zu, "
        "\"leaf\": %zu, \"threads\": %zu, \"tries\": %zu, \"mean\": %.9f, \"median\": %.9f, \"note\": \"%s\"}\n",
        library, algorithm, stage, dataset.c_str(), n, leaf, threads, tries, mean, median, note.c_str());
}

// Runs `workload` kTries times and emits mean and median, like the project's own harness.
void measure(const char* library, const char* algorithm, const char* stage, const Dataset& dataset,
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

auto squared_distance(const double* a, const double* b, std::size_t n_features) -> double {
    auto total = 0.0;
    for (std::size_t feature = 0; feature < n_features; ++feature) {
        const auto difference = a[feature] - b[feature];
        total += difference * difference;
    }
    return total;
}


// ─── nanoflann ───────────────────────────────────────────────────────────────────────────────────
#if defined(HAVE_NANOFLANN)
struct NanoflannCloud {
    const std::vector<double>* data = nullptr;
    std::size_t                n_features = 0;

    auto kdtree_get_point_count() const -> std::size_t { return data->size() / n_features; }
    auto kdtree_get_pt(std::size_t index, std::size_t dimension) const -> double {
        return (*data)[index * n_features + dimension];
    }
    template <class BoundingBox>
    auto kdtree_get_bbox(BoundingBox&) const -> bool { return false; }
};

using NanoflannIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, NanoflannCloud>, NanoflannCloud, -1, std::size_t>;

void run_nanoflann(const Dataset& dataset, const char* workload, std::size_t threads) {
    auto cloud = NanoflannCloud{&dataset.data, dataset.n_features};

    measure("nanoflann", "kdtree", "tree_build", dataset, kLeaf, threads, [&] {
        auto index = NanoflannIndex(static_cast<int>(dataset.n_features), cloud,
                                    nanoflann::KDTreeSingleIndexAdaptorParams(kLeaf));
        index.buildIndex();
    });

    if (std::strcmp(workload, "knn") != 0) return;

    auto index = NanoflannIndex(static_cast<int>(dataset.n_features), cloud,
                                nanoflann::KDTreeSingleIndexAdaptorParams(kLeaf));
    index.buildIndex();

    measure("nanoflann", "kdtree", "knn", dataset, kLeaf, threads, [&] {
        auto indices   = std::vector<std::size_t>(kK);
        auto distances = std::vector<double>(kK);
        auto total     = 0.0;
        for (std::size_t query = 0; query < kNQueries; ++query) {
            index.knnSearch(&dataset.data[query * dataset.n_features], kK, indices.data(), distances.data());
            total += distances.back();
        }
        if (total < 0) std::abort();
    }, "k=2");
}
#endif

// ─── FLANN, exact single tree ───────────────────────────────────────────────────────────────────
#if defined(HAVE_FLANN)
void run_flann(const Dataset& dataset, const char* workload, std::size_t threads) {
    using flann_matrix = flann::Matrix<double>;
    using flann_result = flann::Matrix<int>;
    using flann_dist   = flann::Matrix<double>;

    const auto n = dataset.n_samples;
    const auto d = dataset.n_features;

    measure("flann", "kdtree", "tree_build", dataset, kLeaf, threads, [&] {
        auto data   = flann_matrix(const_cast<double*>(dataset.data.data()), n, d);
        auto params = flann::KDTreeSingleIndexParams(static_cast<int>(kLeaf));
        auto index  = flann::Index<flann::L2_Simple<double>>(data, params);
        index.buildIndex();
    });

    if (std::strcmp(workload, "knn") != 0) return;

    auto data   = flann_matrix(const_cast<double*>(dataset.data.data()), n, d);
    auto params = flann::KDTreeSingleIndexParams(static_cast<int>(kLeaf));
    auto index  = flann::Index<flann::L2_Simple<double>>(data, params);
    index.buildIndex();

    auto queries = flann_matrix(const_cast<double*>(dataset.data.data()), kNQueries, d);
    auto indices = flann_result(new int[kNQueries * kK], kNQueries, kK);
    auto dists   = flann_dist(new double[kNQueries * kK], kNQueries, kK);

    measure("flann", "kdtree", "knn", dataset, kLeaf, threads, [&] {
        index.knnSearch(queries, indices, dists, static_cast<int>(kK), flann::SearchParams(-1));
    }, "k=2 checks=exhaustive");

    delete[] indices.ptr();
    delete[] dists.ptr();
}
#endif

}  // namespace

int main(int argc, char** argv) {
    auto datasets_dir = std::string{};
    auto library      = std::string{};
    auto workload     = std::string{};
    auto threads      = std::size_t{1};

    for (int index = 1; index < argc; ++index) {
        const auto argument = std::string{argv[index]};
        if (argument == "--datasets" && index + 1 < argc) datasets_dir = argv[++index];
        else if (argument == "--library" && index + 1 < argc) library = argv[++index];
        else if (argument == "--workload" && index + 1 < argc) workload = argv[++index];
        else if (argument == "--threads" && index + 1 < argc) threads = std::stoul(argv[++index]);
    }
    if (datasets_dir.empty() || library.empty()) {
        std::fprintf(stderr, "usage: refbench --datasets DIR --library LIB [--workload W] [--threads N]\n");
        return 2;
    }
    std::printf("{\"info\": \"library=%s workload=%s threads=%zu\"}\n", library.c_str(), workload.c_str(), threads);

    for (const auto& name : all_dataset_names()) {
        const auto dataset = load_dataset(datasets_dir, name);

#if defined(HAVE_ARBORX) && defined(REFBENCH_ARBORX_READY)
        if (library == "arborx") {
            run_arborx(dataset, workload.c_str(), threads);
            continue;
        }
#endif
#if defined(HAVE_NANOFLANN)
        if (library == "nanoflann") {
            run_nanoflann(dataset, workload.c_str(), threads);
            continue;
        }
#endif
#if defined(HAVE_FLANN)
        if (library == "flann") {
            run_flann(dataset, workload.c_str(), threads);
            continue;
        }
#endif
        std::fprintf(stderr, "refbench: library %s not compiled in\n", library.c_str());
        return 1;
    }
    return 0;
}
