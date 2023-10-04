#include <gtest/gtest.h>

#include "ffcl/common/Utils.hpp"
#include "ffcl/knn/NearestNeighbor.hpp"

#include "Range2DBaseFixture.hpp"

template <typename DataType>
class NearestNeighborTestFixture : public Range2DBaseFixture<DataType> {
  public:
    void SetUp() override {
        if constexpr (std::is_integral_v<DataType> && std::is_signed_v<DataType>) {
            lower_bound_ = -1;
            upper_bound_ = 1;

        } else if constexpr (std::is_integral_v<DataType> && std::is_unsigned_v<DataType>) {
            lower_bound_ = 0;
            upper_bound_ = 1;

        } else if constexpr (std::is_floating_point_v<DataType>) {
            lower_bound_ = -1;
            upper_bound_ = 1;
        }
        min_n_samples_  = 1;
        max_n_samples_  = 10;
        n_features_     = 1;
        n_random_tests_ = 5;
    }

  protected:
    DataType    lower_bound_;
    DataType    upper_bound_;
    std::size_t min_n_samples_;
    std::size_t max_n_samples_;
    std::size_t n_features_;
    std::size_t n_random_tests_;
};

using DataTypes = ::testing::Types<int, std::size_t, float, double>;
TYPED_TEST_SUITE(NearestNeighborTestFixture, DataTypes);

TYPED_TEST(NearestNeighborTestFixture, NearestNeighborsTest) {
    using DataType = TypeParam;

    std::vector<DataType> data = {/*0*/ 0,
                                  /*1*/ 1,
                                  /*2*/ 2,
                                  /*3*/ 3,
                                  /*4*/ 1,
                                  /*5*/ 4,
                                  /*6*/ 8,
                                  /*7*/ 9,
                                  /*8*/ 2,
                                  /*9*/ 4,
                                  /*10*/ 3};

    std::size_t n_features         = 1;
    std::size_t sample_index_query = 4;
    std::size_t n_neighbors        = 2;

    using SamplesIterator = typename std::vector<DataType>::iterator;

    std::vector<std::size_t> nn_indices = {5, 6, 7, 9};
    std::vector<DataType>    nn_distances(nn_indices.size());

    std::transform(nn_indices.begin(), nn_indices.end(), nn_distances.begin(), [&](std::size_t nn_index) {
        return math::heuristics::auto_distance(data.begin() + sample_index_query * n_features,
                                               data.begin() + sample_index_query * n_features + n_features,
                                               data.begin() + nn_index * n_features);
    });

    printf("Distances:\n");
    this->print_data(nn_distances, 1);

    ffcl::knn::NearestNeighborsBufferWithMemory<SamplesIterator> nn_buffer(nn_indices, nn_distances, n_neighbors);

    auto new_nn_buffer = nn_buffer;

    for (std::size_t i = 0; i < 5; ++i) {
        ffcl::knn::k_nearest_neighbors(
            /**/ data.begin(),
            /**/ data.end(),
            /**/ data.begin(),
            /**/ data.end(),
            /**/ n_features,
            /**/ sample_index_query,
            /**/ new_nn_buffer);
    }
    new_nn_buffer.print();
}

template <typename Indexer, typename IndexerFunction, typename... Args>
auto get_neighbors(std::size_t query_index, const Indexer& indexer, IndexerFunction&& func, Args&&... args) {
    using LabelType = ssize_t;

    const std::size_t n_samples = indexer.n_samples();

    // vector keeping track of the cluster label for each index specified by the global index range
    auto predictions = std::vector<LabelType>(n_samples);

    // initialize the initial cluster counter that's in [0, n_samples)
    LabelType cluster_label = static_cast<LabelType>(0);

    auto query_function = [&indexer = static_cast<const Indexer&>(indexer), func = std::forward<IndexerFunction>(func)](
                              std::size_t index, auto&&... funcArgs) mutable {
        return std::invoke(func, indexer, index, std::forward<decltype(funcArgs)>(funcArgs)...);
    };

    // the indices of the neighbors in the global dataset with their corresponding distances
    // the query sample is not included
    auto initial_neighbors_buffer = query_function(query_index, std::forward<Args>(args)...);

    ++cluster_label;

    predictions[query_index] = cluster_label;

    auto initial_neighbors_indices = initial_neighbors_buffer.extract_indices();

    // iterate over the samples that are assigned to the current cluster
    for (std::size_t cluster_sample_index = 0; cluster_sample_index < initial_neighbors_indices.size();
         ++cluster_sample_index) {
        const auto neighbor_index   = initial_neighbors_indices[cluster_sample_index];
        predictions[neighbor_index] = cluster_label;
    }

    predictions[query_index] = ++cluster_label;
    return predictions;
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
