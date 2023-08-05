#include <gtest/gtest.h>

#include "ffcl/common/Utils.hpp"
#include "ffcl/math/heuristics/NearestNeighbor.hpp"

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

/*
TYPED_TEST(NearestNeighborTestFixture, UpdateNearestNeighborIndicesBufferTest) {
    using DataType        = TypeParam;
    using SamplesIterator = typename std::vector<DataType>::iterator;

    // const std::size_t query_index = this->max_n_samples_ / 2;
    const std::size_t n_neighbors = 5;
    const std::size_t n_samples   = 10;

    const auto distances_buffer =
        this->generate_random_uniform_vector(n_samples, 1, this->lower_bound_, this->upper_bound_);

    NearestNeighborsBuffer<SamplesIterator> nn_buffer(n_neighbors);

    for (std::size_t index = 0; index < distances_buffer.size(); ++index) {
        std::cout << "\t(" << index << ", " << distances_buffer[index] << ")\n";

        nn_buffer.update(index, distances_buffer[index]);

        nn_buffer.print();
        printf("---\n");
    }
    printf("Data:\n");
    this->print_data(distances_buffer, this->n_features_);
    printf("---\n");
    nn_buffer.print();
    printf("---\n");
}
*/

TYPED_TEST(NearestNeighborTestFixture, NearestNeighborsTest) {
    using DataType = TypeParam;

    std::vector<DataType> data               = {0, 1, 2, 3, 1, 4, 8, 9, 2, 4, 3};
    std::size_t           n_features         = 1;
    std::size_t           sample_index_query = 4;

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

    NearestNeighborsBufferWithMemory<SamplesIterator> nn_buffer(nn_indices, nn_distances, 5);

    for (std::size_t i = 0; i < 5; ++i) {
        math::heuristics::k_nearest_neighbors_range(
            /**/ data.begin(),
            /**/ data.end(),
            /**/ data.begin(),
            /**/ data.end(),
            /**/ n_features,
            /**/ sample_index_query,
            /**/ nn_buffer);
    }
    nn_buffer.print();
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
