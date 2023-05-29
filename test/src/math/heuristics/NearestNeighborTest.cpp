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
    const std::size_t query_index                      = this->max_n_samples_ / 2;
    const std::size_t candidate_nearest_neighbor_index = this->max_n_samples_ - 1;
    const std::size_t n_neighbors                      = 5;

    std::vector<TypeParam>   distances_buffer = {0.9, 1.1, 1.2, 1.4, 5.8};  // {1.9, 2.1, 3.6, 5.4, 7.8};
    std::vector<std::size_t> indices_buffer   = this->generate_indices(distances_buffer.size());

    const auto data = this->generate_random_uniform_vector(
        this->max_n_samples_, this->n_features_, this->lower_bound_, this->upper_bound_);

    printf("Candidate distance: %.5f\n",
           math::heuristics::auto_distance(data.begin() + query_index * this->n_features_,
                                           data.begin() + query_index * this->n_features_ + this->n_features_,
                                           data.begin() + candidate_nearest_neighbor_index * this->n_features_));

    printf("Indices before update\n");
    this->print_data(indices_buffer, 1);
    printf("Distances before update\n");
    this->print_data(distances_buffer, 1);

    this->print_data(data, this->n_features_);

    math::heuristics::update_nearest_neighbors_indices_buffer(data.begin(),
                                                              data.end(),
                                                              this->n_features_,
                                                              query_index,
                                                              candidate_nearest_neighbor_index,
                                                              n_neighbors,
                                                              indices_buffer,
                                                              distances_buffer);

    printf("Indices after update\n");
    this->print_data(indices_buffer, 1);
    printf("Distances after update\n");
    this->print_data(distances_buffer, 1);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
