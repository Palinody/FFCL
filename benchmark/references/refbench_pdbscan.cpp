// pdbscan (Wang, Gu, Shun — "Theoretically-Efficient and Practical Parallel DBSCAN") adapter.
// Separate executable: its bundled parlay scheduler clashes with pargeo hdbscan's copy.
#include "refbench_common.hpp"

#include "dbscan/capi.h"
#include "dbscan/pbbs/parallel.h"

#include <cstring>

using namespace refbench;

namespace {

void run_pdbscan(const Dataset& dataset, const char* workload, std::size_t threads) {
    if (std::strcmp(workload, "dbscan") != 0) {
        std::fprintf(stderr, "refbench_pdbscan: unsupported workload %s\n", workload);
        std::exit(1);
    }

    const auto n   = dataset.n_samples;
    const auto dim = dataset.n_features;
    // The C API wants a mutable double buffer; it does not modify it, so a copy is enough.
    auto data  = dataset.data;
    auto core  = std::vector<char>(n);
    auto labels = std::vector<int>(n);

    measure("pdbscan", "dbscan", "full", dataset, 0, threads, [&] {
        const auto error = DBSCAN(static_cast<int>(dim), static_cast<int>(n), data.data(), kRadius,
                                  static_cast<int>(kMinSamples), reinterpret_cast<bool*>(core.data()),
                                  labels.data());
        if (error != 0) std::abort();
    }, "radius=0.5 min_pts=5 grid-based");
}

}  // namespace

int main(int argc, char** argv) {
    const auto options = parse_options(argc, argv);
    if (options.datasets_dir.empty() || options.library.empty()) {
        std::fprintf(stderr, "usage: refbench_pdbscan --datasets DIR --library pdbscan [--workload dbscan]\n");
        return 2;
    }

    parlay::internal::start_scheduler();
    announce(options);
    const auto status = for_each_dataset(options, [&](const Dataset& dataset) {
        run_pdbscan(dataset, options.workload.c_str(), options.threads);
    });
    parlay::internal::stop_scheduler();
    return status;
}
