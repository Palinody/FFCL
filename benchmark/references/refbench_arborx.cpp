// ArborX adapter: BVH build, kNN, radius, DBSCAN, mutual-reachability MST, dendrogram and the
// HDBSCAN dendrogram. Own executable so the other libraries' code cannot perturb its code layout.
#include "refbench_common.hpp"

#include <ArborX.hpp>
#include <ArborX_DBSCAN.hpp>
#include <ArborX_Dendrogram.hpp>
#include <ArborX_HDBSCAN.hpp>
#include <ArborX_MinimumSpanningTree.hpp>
#include <Kokkos_Core.hpp>

using namespace refbench;

namespace {

using ArbExecutionSpace = Kokkos::DefaultExecutionSpace;
using ArbMemorySpace    = ArbExecutionSpace::memory_space;

template <int Dim>
auto make_points(const Dataset& dataset) -> Kokkos::View<ArborX::Point<Dim>*, ArbMemorySpace> {
    auto points = Kokkos::View<ArborX::Point<Dim>*, ArbMemorySpace>("refbench::points", dataset.n_samples);
    auto host   = Kokkos::create_mirror_view(points);
    for (std::size_t i = 0; i < dataset.n_samples; ++i)
        for (int j = 0; j < Dim; ++j) host(i)[j] = static_cast<float>(dataset.data[i * dataset.n_features + j]);
    Kokkos::deep_copy(points, host);
    return points;
}

template <int Dim>
void run_dim(const Dataset& dataset, const char* workload, std::size_t threads) {
    auto exec   = ArbExecutionSpace{};
    auto points = make_points<Dim>(dataset);

    if (std::strcmp(workload, "build") == 0 || std::strcmp(workload, "knn") == 0 ||
        std::strcmp(workload, "radius") == 0) {
        // attach_indices makes the query results carry the original point indices.
        auto indexed = ArborX::Experimental::attach_indices(points);
        measure("arborx", "kdtree", "tree_build", dataset, kLeaf, threads, [&] {
            auto bvh = ArborX::BoundingVolumeHierarchy(exec, indexed);
            if (bvh.size() == 0) std::abort();
        }, "bvh (ArborX has no leaf-size parameter)");

        if (std::strcmp(workload, "build") == 0) return;

        auto bvh = ArborX::BoundingVolumeHierarchy(exec, indexed);

        auto queries      = Kokkos::View<ArborX::Point<Dim>*, ArbMemorySpace>("refbench::queries", kNQueries);
        auto queries_host = Kokkos::create_mirror_view(queries);
        for (std::size_t i = 0; i < kNQueries; ++i)
            for (int j = 0; j < Dim; ++j)
                queries_host(i)[j] = static_cast<float>(dataset.data[i * dataset.n_features + j]);
        Kokkos::deep_copy(queries, queries_host);

        if (std::strcmp(workload, "knn") == 0) {
            measure("arborx", "kdtree", "knn", dataset, kLeaf, threads, [&] {
                auto values  = Kokkos::View<int*, ArbMemorySpace>("refbench::knn_values", 0);
                auto offsets = Kokkos::View<int*, ArbMemorySpace>("refbench::knn_offsets", 0);
                bvh.query(exec, ArborX::Experimental::make_nearest(queries, static_cast<int>(kK)),
                          ArborX::Experimental::ExtractPairIndexCallback{}, values, offsets);
                if (offsets.size() == 0) std::abort();
            }, "k=2");
        } else {
            measure("arborx", "kdtree", "radius", dataset, kLeaf, threads, [&] {
                auto values  = Kokkos::View<int*, ArbMemorySpace>("refbench::radius_values", 0);
                auto offsets = Kokkos::View<int*, ArbMemorySpace>("refbench::radius_offsets", 0);
                bvh.query(exec, ArborX::Experimental::make_intersects(queries, kRadius),
                          ArborX::Experimental::ExtractPairIndexCallback{}, values, offsets);
                if (offsets.size() == 0) std::abort();
            }, "radius=0.5");
        }
        return;
    }

    if (std::strcmp(workload, "dbscan") == 0) {
        measure("arborx", "dbscan", "full", dataset, kLeaf, threads, [&] {
            auto labels = Kokkos::View<int*, ArbMemorySpace>("refbench::labels", 0);
            // core_min_size counts the point itself, so 5 matches sklearn's min_samples=5.
            ArborX::dbscan(exec, points, kRadius, static_cast<int>(kMinSamples), labels);
            if (labels.size() == 0) std::abort();
        }, "radius=0.5 core_min_size=5");
        return;
    }

    if (std::strcmp(workload, "mst") == 0) {
        measure("arborx", "mst", "full", dataset, kLeaf, threads, [&] {
            auto mst = ArborX::Experimental::MinimumSpanningTree<ArbMemorySpace>(exec, points,
                                                                                 static_cast<int>(kK));
            if (mst.edges.size() == 0) std::abort();
        }, "mutual-reachability core_min_size=2");
        return;
    }

    if (std::strcmp(workload, "dendrogram") == 0) {
        // The MST is built outside the timed region, so this row is the dendrogram construction only.
        auto mst = ArborX::Experimental::MinimumSpanningTree<ArbMemorySpace>(exec, points, static_cast<int>(kK));
        measure("arborx", "dendrogram", "full", dataset, kLeaf, threads, [&] {
            auto dendrogram = ArborX::Experimental::Dendrogram<ArbMemorySpace>(exec, mst.edges);
            if (dendrogram._parents.size() == 0) std::abort();
        }, "from mutual-reachability MST core_min_size=2");
        return;
    }

    if (std::strcmp(workload, "hdbscan") == 0) {
        measure("arborx", "hdbscan", "dendrogram", dataset, kLeaf, threads, [&] {
            // ArborX stops at the dendrogram: it does not extract flat clusters, so this is not a
            // full HDBSCAN row. Reported as hdbscan/dendrogram.
            auto dendrogram = ArborX::Experimental::hdbscan(exec, points, static_cast<int>(kK));
            if (dendrogram._parents.size() == 0) std::abort();
        }, "returns dendrogram, no flat labels; core_min_size=2");
        return;
    }

    std::fprintf(stderr, "arborx: unsupported workload %s\n", workload);
    std::exit(1);
}

void run(const Dataset& dataset, const char* workload, std::size_t threads) {
    switch (dataset.n_features) {
        case 2: run_dim<2>(dataset, workload, threads); break;
        case 3: run_dim<3>(dataset, workload, threads); break;
        case 4: run_dim<4>(dataset, workload, threads); break;
        default: std::fprintf(stderr, "arborx: unsupported dimension %zu\n", dataset.n_features); std::exit(1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const auto options = parse_options(argc, argv);
    if (options.datasets_dir.empty()) {
        std::fprintf(stderr, "usage: refbench_arborx --datasets DIR [--workload W] [--threads N]\n");
        return 2;
    }

    Kokkos::initialize(argc, argv);
    announce(options);
    const auto status = for_each_dataset(options, [&](const Dataset& dataset) {
        run(dataset, options.workload.c_str(), options.threads);
    });
    Kokkos::finalize();
    return status;
}
