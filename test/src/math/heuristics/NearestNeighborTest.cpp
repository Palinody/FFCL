#include "ffcl/math/heuristics/NearestNeighbor.hpp"
#include "ffcl/common/Utils.hpp"

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

using DataTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NearestNeighborTestFixture, DataTypes);

TYPED_TEST(NearestNeighborTestFixture, UpdateNearestNeighborIndicesBufferTest) {
    // const std::size_t query_index                      = this->max_n_samples_ / 2;
    const std::size_t n_neighbors = 5;

    const auto distances_buffer = this->generate_random_uniform_vector(
        this->max_n_samples_, this->n_features_, this->lower_bound_, this->upper_bound_);

    this->print_data(distances_buffer, this->n_features_);

    using DataType        = TypeParam;
    using SamplesIterator = typename std::vector<DataType>::iterator;

    NearestNeighborsBuffer<SamplesIterator> nn_priority_queue(n_neighbors);

    auto nn_priority_queue_2 = nn_priority_queue;

    for (std::size_t index = 0; index < distances_buffer.size(); ++index) {
        nn_priority_queue_2.update(index, distances_buffer[index]);
    }
    nn_priority_queue_2.print();

    printf("Distances after update\n");
    this->print_data(distances_buffer, 1);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
