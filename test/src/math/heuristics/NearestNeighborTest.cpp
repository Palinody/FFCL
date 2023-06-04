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

using DataTypes = ::testing::Types<int, std::size_t, float, double>;
TYPED_TEST_SUITE(NearestNeighborTestFixture, DataTypes);

TYPED_TEST(NearestNeighborTestFixture, UpdateNearestNeighborIndicesBufferTest) {
    using DataType        = TypeParam;
    using SamplesIterator = typename std::vector<DataType>::iterator;

    // const std::size_t query_index = this->max_n_samples_ / 2;
    const std::size_t n_neighbors = 5;
    const std::size_t n_samples   = 10;

    const auto distances_buffer =
        this->generate_random_uniform_vector(n_samples, 1, this->lower_bound_, this->upper_bound_);

    NearestNeighborsBuffer<SamplesIterator> nn_priority_queue(n_neighbors);

    for (std::size_t index = 0; index < distances_buffer.size(); ++index) {
        std::cout << "\t(" << index << ", " << distances_buffer[index] << ")\n";

        nn_priority_queue.update(index, distances_buffer[index]);

        nn_priority_queue.print();
        printf("---\n");
    }
    printf("Data:\n");
    this->print_data(distances_buffer, this->n_features_);
    printf("---\n");
    nn_priority_queue.print();
    printf("---\n");
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
