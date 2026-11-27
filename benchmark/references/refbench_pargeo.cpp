// pargeo hdbscan (Wang, Yu, Gu, Shun — "Fast Parallel Algorithms for Euclidean Minimum Spanning
// Tree and Hierarchical Spatial Clustering") adapter. Separate executable: its parlay scheduler
// clashes with pdbscan's copy.
//
// pargeo's hdbscan() returns the mutual-reachability MST edges; dendrogram() turns those into the
// cluster hierarchy. It does not extract flat labels, so there is no "full HDBSCAN" row here — only
// MST and dendrogram.
#include "refbench_common.hpp"

#include "hdbscan/hdbscan.h"
#include "hdbscan/point.h"
#include "parlay/sequence.h"

#include <cstring>
#include <iostream>
#include <sstream>

using namespace refbench;

namespace {

// pargeo prints its own stage timings to stdout; keep them out of the JSON stream.
class CoutSilencer {
public:
    CoutSilencer() : previous_(std::cout.rdbuf(sink_.rdbuf())) {}
    ~CoutSilencer() { std::cout.rdbuf(previous_); }

private:
    std::stringstream sink_;
    std::streambuf*   previous_;
};

template <int Dim>
auto make_sequence(const Dataset& dataset) -> parlay::sequence<pargeo::point<Dim>> {
    auto points = parlay::sequence<pargeo::point<Dim>>(dataset.n_samples);
    for (std::size_t i = 0; i < dataset.n_samples; ++i) {
        auto point = pargeo::point<Dim>{};
        for (int j = 0; j < Dim; ++j) point[j] = dataset.data[i * dataset.n_features + j];
        points[i] = point;
    }
    return points;
}

template <int Dim>
void run_pargeo_dim(const Dataset& dataset, const char* workload, std::size_t threads) {
    auto points = make_sequence<Dim>(dataset);

    if (std::strcmp(workload, "mst") == 0) {
        auto guard = CoutSilencer{};
        measure("pargeo", "mst", "full", dataset, 0, threads, [&] {
            // minPts = 2: core distance is the distance to the 2nd nearest point including self,
            // which is FFCL's k_nearest_neighbors = 2.
            auto edges = pargeo::hdbscan<Dim>(points, kK);
            if (edges.size() == 0) std::abort();
        }, "mutual-reachability MST min_pts=2");
        return;
    }

    if (std::strcmp(workload, "dendrogram") == 0) {
        auto edges = parlay::sequence<pargeo::wghEdge>{};
        {
            auto guard = CoutSilencer{};
            edges      = pargeo::hdbscan<Dim>(points, kK);
        }
        auto guard = CoutSilencer{};
        measure("pargeo", "dendrogram", "full", dataset, 0, threads, [&] {
            // dendrogram() takes the edge list by non-const reference, so time it on a copy.
            auto edges_copy = edges;
            auto nodes      = pargeo::dendrogram(edges_copy, dataset.n_samples);
            if (nodes.size() == 0) std::abort();
        }, "from mutual-reachability MST min_pts=2");
        return;
    }

    std::fprintf(stderr, "refbench_pargeo: unsupported workload %s\n", workload);
    std::exit(1);
}

void run_pargeo(const Dataset& dataset, const char* workload, std::size_t threads) {
    switch (dataset.n_features) {
        case 2: run_pargeo_dim<2>(dataset, workload, threads); break;
        case 3: run_pargeo_dim<3>(dataset, workload, threads); break;
        case 4: run_pargeo_dim<4>(dataset, workload, threads); break;
        default: std::fprintf(stderr, "pargeo: unsupported dimension %zu\n", dataset.n_features); std::exit(1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const auto options = parse_options(argc, argv);
    if (options.datasets_dir.empty() || options.library.empty()) {
        std::fprintf(stderr, "usage: refbench_pargeo --datasets DIR --library pargeo [--workload mst|dendrogram]\n");
        return 2;
    }

    announce(options);
    return for_each_dataset(options, [&](const Dataset& dataset) {
        run_pargeo(dataset, options.workload.c_str(), options.threads);
    });
}
