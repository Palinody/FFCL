// mlpack adapter: KD-tree build, kNN, DBSCAN and the Euclidean EMST. Own executable so the other
// libraries' code cannot perturb its code layout (linking mlpack into the shared binary measurably
// slowed the nanoflann hot loop).
#include "refbench_common.hpp"

#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/methods/emst/emst.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace refbench;

namespace {

using MlpackTree = mlpack::KDTree<mlpack::EuclideanDistance,
                                  mlpack::NeighborSearchStat<mlpack::NearestNeighborSort>,
                                  arma::mat>;

auto to_arma(const Dataset& dataset) -> arma::mat {
    auto matrix = arma::mat(dataset.n_features, dataset.n_samples);
    for (std::size_t i = 0; i < dataset.n_samples; ++i)
        for (std::size_t j = 0; j < dataset.n_features; ++j)
            matrix(j, i) = dataset.data[i * dataset.n_features + j];
    return matrix;
}

void run(const Dataset& dataset, const char* workload, std::size_t threads) {
    const auto data = to_arma(dataset);

    if (std::strcmp(workload, "build") == 0 || std::strcmp(workload, "knn") == 0) {
        measure("mlpack", "kdtree", "tree_build", dataset, kLeaf, threads, [&] {
            auto tree = MlpackTree(data, kLeaf);
            if (tree.NumDescendants() == 0) std::abort();
        }, "leaf=40");

        if (std::strcmp(workload, "build") == 0) return;

        auto tree    = MlpackTree(data, kLeaf);
        auto knn     = mlpack::KNN(tree);
        auto queries = data.cols(0, kNQueries - 1);

        measure("mlpack", "kdtree", "knn", dataset, kLeaf, threads, [&] {
            auto neighbors = arma::Mat<size_t>{};
            auto distances = arma::mat{};
            knn.Search(queries, kK, neighbors, distances);
            if (neighbors.n_elem == 0) std::abort();
        }, "k=2 dual-tree");
        return;
    }

    if (std::strcmp(workload, "dbscan") == 0) {
        measure("mlpack", "dbscan", "full", dataset, 20, threads, [&] {
            // The internal range-search tree is built with mlpack's default leaf size (20); the
            // public DBSCAN constructor does not expose it.
            auto dbscan      = mlpack::DBSCAN<>(kRadius, kMinSamples);
            auto assignments = arma::Row<size_t>{};
            dbscan.Cluster(data, assignments);
            if (assignments.n_elem == 0) std::abort();
        }, "radius=0.5 min_points=5 leaf=20 (mlpack default)");
        return;
    }

    if (std::strcmp(workload, "mst") == 0) {
        measure("mlpack", "mst", "full", dataset, 20, threads, [&] {
            // Plain Euclidean MST, not mutual reachability: DualTreeBoruvka has no core-distance
            // notion. Reported with that caveat rather than silently compared to FFCL's MR-MST.
            auto boruvka = mlpack::DualTreeBoruvka<>(data);
            auto results = arma::mat{};
            boruvka.ComputeMST(results);
            if (results.n_cols == 0) std::abort();
        }, "euclidean MST (not mutual reachability) leaf=20 (mlpack default)");
        return;
    }

    std::fprintf(stderr, "mlpack: unsupported workload %s\n", workload);
    std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
    const auto options = parse_options(argc, argv);
    if (options.datasets_dir.empty()) {
        std::fprintf(stderr, "usage: refbench_mlpack --datasets DIR [--workload build|knn|dbscan|mst]\n");
        return 2;
    }
    announce(options);
    return for_each_dataset(options, [&](const Dataset& dataset) {
        run(dataset, options.workload.c_str(), options.threads);
    });
}
