// nanoflann adapter. Own executable so the other libraries' code cannot perturb its code layout.
#include "refbench_common.hpp"

#include <nanoflann.hpp>

using namespace refbench;

namespace {

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

void run(const Dataset& dataset, const char* workload, std::size_t threads) {
    auto cloud = NanoflannCloud{&dataset.data, dataset.n_features};

    measure("nanoflann", "kdtree", "tree_build", dataset, kLeaf, threads, [&] {
        auto index = NanoflannIndex(static_cast<int>(dataset.n_features), cloud,
                                    nanoflann::KDTreeSingleIndexAdaptorParams(kLeaf));
        index.buildIndex();
    }, "leaf=40");

    if (std::string{workload} != "knn") return;

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

}  // namespace

int main(int argc, char** argv) {
    const auto options = parse_options(argc, argv);
    if (options.datasets_dir.empty()) {
        std::fprintf(stderr, "usage: refbench_nanoflann --datasets DIR [--workload build|knn] [--threads N]\n");
        return 2;
    }
    announce(options);
    return for_each_dataset(options, [&](const Dataset& dataset) {
        run(dataset, options.workload.c_str(), options.threads);
    });
}
