#include <gtest/gtest.h>

#include "ffcl/common/Utils.hpp"

#include "ffcl/math/heuristics/Distances.hpp"

#include "ffcl/knn/buffer/WithMemory.hpp"

#include "ffcl/knn/search/KNearestNeighborsSearch.hpp"

#include "Range2DBaseFixture.hpp"

template <typename ValueType>
class KNearestNeighborsSearchTestFixture : public Range2DBaseFixture<ValueType> {
  public:
    void SetUp() override {
        if constexpr (std::is_integral_v<ValueType> && std::is_signed_v<ValueType>) {
            lower_bound_ = -1;
            upper_bound_ = 1;

        } else if constexpr (std::is_integral_v<ValueType> && std::is_unsigned_v<ValueType>) {
            lower_bound_ = 0;
            upper_bound_ = 1;

        } else if constexpr (std::is_floating_point_v<ValueType>) {
            lower_bound_ = -1;
            upper_bound_ = 1;
        }
        min_n_samples_  = 1;
        max_n_samples_  = 10;
        n_features_     = 1;
        n_random_tests_ = 5;
    }

  protected:
    ValueType   lower_bound_;
    ValueType   upper_bound_;
    std::size_t min_n_samples_;
    std::size_t max_n_samples_;
    std::size_t n_features_;
    std::size_t n_random_tests_;
};

using DataTypes = ::testing::Types<int, std::size_t, float, double>;
TYPED_TEST_SUITE(KNearestNeighborsSearchTestFixture, DataTypes);

TYPED_TEST(KNearestNeighborsSearchTestFixture, NearestNeighborsTest) {
    using IndexType           = std::size_t;
    using ValueType           = TypeParam;
    using IndicesType         = std::vector<IndexType>;
    using ValuesType          = std::vector<ValueType>;
    using IndicesIteratorType = typename IndicesType::iterator;
    using ValuesIteratorType  = typename ValuesType::iterator;

    std::vector<ValueType> data = {/*0*/ 0,
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

    IndicesType nn_indices = {5, 6, 7, 9};
    ValuesType  nn_distances(nn_indices.size());

    std::transform(nn_indices.begin(), nn_indices.end(), nn_distances.begin(), [&](const auto& nn_index) {
        return math::heuristics::auto_distance(data.begin() + sample_index_query * n_features,
                                               data.begin() + sample_index_query * n_features + n_features,
                                               data.begin() + nn_index * n_features);
    });

    printf("Distances:\n");
    this->print_data(nn_distances, 1);

    auto nn_buffer =
        ffcl::knn::buffer::WithMemory<IndicesIteratorType, ValuesIteratorType>(nn_indices, nn_distances, n_neighbors);

    auto new_nn_buffer = nn_buffer;

    for (std::size_t i = 0; i < 5; ++i) {
        new_nn_buffer(
            /**/ nn_indices.begin(),
            /**/ nn_indices.end(),
            /**/ data.begin(),
            /**/ data.end(),
            /**/ n_features,
            /**/ sample_index_query);
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
