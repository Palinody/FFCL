#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/tree/kdtree/KDTree.hpp"

#include "ffcl/search/Search.hpp"
#include "ffcl/search/buffer/Unsorted.hpp"

#include "ffcl/datastruct/bounds/segment_representation/MiddleAndLength.hpp"
#include "ffcl/datastruct/bounds/segment_representation/MinAndMax.hpp"
#include "ffcl/datastruct/bounds/segment_representation/PositionAndLength.hpp"

#include "ffcl/datastruct/bounds/Ball.hpp"
#include "ffcl/datastruct/bounds/BoundingBox.hpp"
#include "ffcl/datastruct/bounds/UnboundedBall.hpp"
#include "ffcl/datastruct/bounds/Vertex.hpp"

#include <sys/types.h>  // std::ssize_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace fs = std::filesystem;

class SearcherErrorsTest : public ::testing::Test {
  public:
    using dType = float;

  protected:
    ssize_t get_num_features_in_file(const fs::path& filepath, char delimiter = ' ') {
        std::ifstream file(filepath);
        ssize_t       n_features = -1;
        if (file.is_open()) {
            std::string line;

            std::getline(file, line);
            // count the number of values at the first line, delimited by the specified delimiter
            n_features = std::count(line.begin(), line.end(), delimiter) + 1;

            file.close();

        } else {
            throw std::ios_base::failure("Unable to open file: " + filepath.string());
        }
        return n_features;
    }

    template <typename T = float>
    std::vector<T> load_data(const fs::path& filename, char delimiter = ' ') {
        std::ifstream  filestream(filename);
        std::vector<T> data;

        if (filestream.is_open()) {
            // temporary string data
            std::string row_str, elem_str;

            while (std::getline(filestream, row_str, '\n')) {
                std::stringstream row_str_stream(row_str);

                while (std::getline(row_str_stream, elem_str, delimiter)) {
                    data.emplace_back(std::stof(elem_str));
                }
            }
            filestream.close();
        }
        return data;
    }

    void make_directories(const fs::path& directory_path) {
        try {
            if (!std::filesystem::exists(directory_path)) {
                std::filesystem::create_directories(directory_path);

#if defined(VERBOSE) && VERBOSE == true
                std::cout << "Directory created: " << directory_path << "\n";

            } else {
                std::cout << "Dir. already exists\n";
#endif
            }
        } catch (std::filesystem::filesystem_error& e) {
            std::cerr << e.what() << std::endl;
        }
    }

    template <typename T = float>
    void write_data(const std::vector<T>& data, std::size_t n_features, const fs::path& filename) {
        const auto parent_path = filename.parent_path();

        make_directories(parent_path);

        std::ofstream filestream(filename);

        if (filestream.is_open()) {
            std::size_t iter{};

            for (const auto& elem : data) {
                filestream << elem;

                ++iter;

                if (iter % n_features == 0 && iter != 0) {
                    filestream << '\n';

                } else {
                    filestream << ' ';
                }
            }
        }
    }

    static constexpr std::size_t n_iterations_global = 100;
    static constexpr std::size_t n_centroids_global  = 4;

    const fs::path folder_root_        = fs::path("../bin/clustering");
    const fs::path inputs_folder_      = folder_root_ / fs::path("inputs");
    const fs::path targets_folder_     = folder_root_ / fs::path("targets");
    const fs::path predictions_folder_ = folder_root_ / fs::path("predictions");
};

std::vector<std::size_t> generate_indices(std::size_t n_samples) {
    std::vector<std::size_t> elements(n_samples);
    std::iota(elements.begin(), elements.end(), static_cast<std::size_t>(0));
    return elements;
}

TEST_F(SearcherErrorsTest, NoisyCirclesTest) {
    fs::path filename = "no_structure.txt";

    using IndexType = std::size_t;
    using ValueType = dType;

    auto            data       = load_data<ValueType>(inputs_folder_ / filename, ' ');
    const IndexType n_features = get_num_features_in_file(inputs_folder_ / filename);
    const IndexType n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType{})
                                   .splitting_rule_policy(SplittingRulePolicyType{}));

    // using SegmentType = ffcl::datastruct::bounds::segment_representation::MinAndMax<ValueType>;
    // using BoundType   = ffcl::datastruct::bounds::BoundingBox<SegmentType>;
    // using BoundType = ffcl::datastruct::bounds::Ball<ValueType, 2>;
    // using BoundType = ffcl::datastruct::bounds::UnboundedBall<ValueType, 2>;
    // using BoundType = ffcl::datastruct::bounds::UnboundedBallView<SamplesIterator>;
    // using BoundType = ffcl::datastruct::bounds::BallView<SamplesIterator>;
    // using BoundType = ffcl::datastruct::bounds::BoundingBoxView<SamplesIterator>;

    // using BufferType = ffcl::search::buffer::Unsorted<SamplesIterator, BoundType>;

    // auto bound_ptr = std::make_shared<BoundType>(BoundType({{-15, -10}, {-15, -3}}));
    // auto bound_ptr = std::make_shared<BoundType>(BoundType{{-10, -10}, 10});
    // auto bound_ptr = std::make_shared<BoundType>(BoundType{{-10, -10}});

    auto center_point_query = std::vector<ValueType>{-10, -10};
    // const ValueType radius_query = 5;
    auto lengths_from_center_point_query = std::vector<ValueType>{2, 6};

    // auto bound_query = BoundType(center_point_query.begin(), center_point_query.end(), radius_query);
    auto bound_query = ffcl::datastruct::bounds::BoundingBoxView(
        /**/ center_point_query.begin(),
        /**/ center_point_query.end(),
        /**/ lengths_from_center_point_query);

    const IndexType max_capacity = ffcl::common::infinity<IndexType>();

    auto bounded_buffer_query = ffcl::search::buffer::Unsorted(std::move(bound_query), /*max_capacity=*/max_capacity);
    // auto bounded_buffer_query =
    // BufferType(center_point_query.begin(), center_point_query.end(), /*max_capacity=*/max_capacity);

    auto searcher = ffcl::search::Searcher(std::move(indexer));

    const auto returned_indices = searcher(std::move(bounded_buffer_query)).indices();

    auto predictions = std::vector<IndexType>(n_samples);

    for (const auto& index : returned_indices) {
        predictions[index] = 1;
    }

    write_data<IndexType>(predictions, 1, predictions_folder_ / fs::path(filename));
}

TEST_F(SearcherErrorsTest, NoisyCirclesBenchmarkTest) {
    fs::path filename = "no_structure.txt";

    using IndexType = std::size_t;
    using ValueType = dType;

    auto            data       = load_data<ValueType>(inputs_folder_ / filename, ' ');
    const IndexType n_features = get_num_features_in_file(inputs_folder_ / filename);
    const IndexType n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType{})
                                   .splitting_rule_policy(SplittingRulePolicyType{}));

    // using BufferType = ffcl::search::buffer::Unsorted<SamplesIterator>;

    auto searcher = ffcl::search::Searcher(std::move(indexer));

    constexpr IndexType max_capacity = 10;  // ffcl::common::infinity<IndexType>();

    std::size_t returned_indices_counter = 0;

    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        auto bounded_buffer_query =
            ffcl::search::buffer::Unsorted(data.begin() + sample_index_query * n_features,
                                           data.begin() + sample_index_query * n_features + n_features,
                                           max_capacity);

        const auto returned_indices = searcher(std::move(bounded_buffer_query)).indices();

        returned_indices_counter += returned_indices.size();
    }
    std::cout << "returned_indices_counter: " << returned_indices_counter << "\n";
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}