#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ffcl/common/Timer.hpp"
#include "ffcl/containers/kdtree/KDTreeIndexed.hpp"

#include <sys/types.h>  // std::ssize_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace fs = std::filesystem;

class KDTreeIndexedErrorsTest : public ::testing::Test {
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

    const fs::path kdtree_folder_root_     = fs::path("../bin/kdtree");
    const fs::path clustering_folder_root_ = fs::path("../bin/clustering");
    const fs::path inputs_folder_          = clustering_folder_root_ / fs::path("inputs");
    const fs::path targets_folder_         = clustering_folder_root_ / fs::path("targets");
};

std::vector<std::size_t> generate_indices(std::size_t n_samples) {
    std::vector<std::size_t> elements(n_samples);
    std::iota(elements.begin(), elements.end(), static_cast<std::size_t>(0));
    return elements;
}

TEST_F(KDTreeIndexedErrorsTest, MainTest) {
    fs::path filename = "noisy_circles.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = labels.size();

    const std::size_t sample_index_query = 1;

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    auto data_indices = generate_indices(n_samples);

    using IndicesIterator = decltype(data_indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    // IndexedHighestVarianceBuild, IndexedMaximumSpreadBuild, IndexedCycleThroughAxesBuild
    auto kdtree = ffcl::containers::KDTreeIndexed(
        data_indices.begin(),
        data_indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::containers::KDTreeIndexed<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(n_samples)
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::IndexedHighestVarianceBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    common::timer::Timer<common::timer::Nanoseconds> timer;

    timer.reset();
    const auto nn_index = kdtree.get_nearest_neighbor_index(1);
    timer.print_elapsed_seconds(/*n_decimals=*/6);

    printf("Query position: %.3f, %.3f\n",
           data.begin()[sample_index_query * n_features + 0],
           data.begin()[sample_index_query * n_features + 1]);

    printf("nn_index: %ld\n", nn_index);
    printf(
        "nn position: %.3f, %.3f\n", data.begin()[nn_index * n_features + 0], data.begin()[nn_index * n_features + 1]);

    timer.reset();
    const auto [nearest_neighbor_index, nearest_neighbor_distance] = math::heuristics::nearest_neighbor_indexed_range(
        data_indices.begin(), data_indices.end(), data.begin(), data.end(), n_features, sample_index_query);
    timer.print_elapsed_seconds(/*n_decimals=*/6);

    printf("CORRECT ANSWER: nn_index: %ld, nn_distance: %.3f\n", nearest_neighbor_index, nearest_neighbor_distance);
    printf("nn position: %.3f, %.3f\n",
           data.begin()[nearest_neighbor_index * n_features + 0],
           data.begin()[nearest_neighbor_index * n_features + 1]);
}

TEST_F(KDTreeIndexedErrorsTest, MNISTTest) {
    fs::path filename = "mnist.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = labels.size();

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    auto data_indices = generate_indices(n_samples);

    using IndicesIterator = decltype(data_indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    // IndexedHighestVarianceBuild, IndexedMaximumSpreadBuild, IndexedCycleThroughAxesBuild
    auto kdtree = ffcl::containers::KDTreeIndexed(
        data_indices.begin(),
        data_indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::containers::KDTreeIndexed<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(10)
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::IndexedHighestVarianceBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(/*n_decimals=*/6);
}

TEST_F(KDTreeIndexedErrorsTest, NoisyCirclesTest) {
    fs::path filename = "noisy_circles.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = labels.size();

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << labels.size() << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    auto data_indices = generate_indices(n_samples);

    using IndicesIterator = decltype(data_indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;

    auto kdtree = ffcl::containers::KDTreeIndexed(
        data_indices.begin(),
        data_indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::containers::KDTreeIndexed<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(10)
            .axis_selection_policy(kdtree::policy::IndexedMaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(/*n_decimals=*/6);

    make_directories(kdtree_folder_root_);

    fs::path kdtree_filename = filename.stem().string() + ".json";

    kdtree.serialize(kdtree_folder_root_ / kdtree_filename);
}

TEST_F(KDTreeIndexedErrorsTest, NoisyMoonsTest) {
    fs::path filename = "noisy_moons.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = labels.size();

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << labels.size() << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    auto data_indices = generate_indices(n_samples);

    using IndicesIterator = decltype(data_indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;

    auto kdtree = ffcl::containers::KDTreeIndexed(
        data_indices.begin(),
        data_indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::containers::KDTreeIndexed<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(10)
            .axis_selection_policy(kdtree::policy::IndexedMaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(/*n_decimals=*/6);

    make_directories(kdtree_folder_root_);

    fs::path kdtree_filename = filename.stem().string() + ".json";

    kdtree.serialize(kdtree_folder_root_ / kdtree_filename);
}

TEST_F(KDTreeIndexedErrorsTest, VariedTest) {
    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = labels.size();

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << labels.size() << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    auto data_indices = generate_indices(n_samples);

    using IndicesIterator = decltype(data_indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;

    auto kdtree = ffcl::containers::KDTreeIndexed(
        data_indices.begin(),
        data_indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::containers::KDTreeIndexed<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(10)
            .axis_selection_policy(kdtree::policy::IndexedMaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(/*n_decimals=*/6);

    make_directories(kdtree_folder_root_);

    fs::path kdtree_filename = filename.stem().string() + ".json";

    kdtree.serialize(kdtree_folder_root_ / kdtree_filename);
}

TEST_F(KDTreeIndexedErrorsTest, AnisoTest) {
    fs::path filename = "aniso.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = labels.size();

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << labels.size() << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    auto data_indices = generate_indices(n_samples);

    using IndicesIterator = decltype(data_indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;

    auto kdtree = ffcl::containers::KDTreeIndexed(
        data_indices.begin(),
        data_indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::containers::KDTreeIndexed<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(10)
            .axis_selection_policy(kdtree::policy::IndexedMaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(/*n_decimals=*/6);

    make_directories(kdtree_folder_root_);

    fs::path kdtree_filename = filename.stem().string() + ".json";

    kdtree.serialize(kdtree_folder_root_ / kdtree_filename);
}

TEST_F(KDTreeIndexedErrorsTest, BlobsTest) {
    fs::path filename = "blobs.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = labels.size();

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << labels.size() << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    auto data_indices = generate_indices(n_samples);

    using IndicesIterator = decltype(data_indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;

    auto kdtree = ffcl::containers::KDTreeIndexed(
        data_indices.begin(),
        data_indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::containers::KDTreeIndexed<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(10)
            .axis_selection_policy(kdtree::policy::IndexedMaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(/*n_decimals=*/6);

    make_directories(kdtree_folder_root_);

    fs::path kdtree_filename = filename.stem().string() + ".json";

    kdtree.serialize(kdtree_folder_root_ / kdtree_filename);
}

TEST_F(KDTreeIndexedErrorsTest, NoStructureTest) {
    fs::path filename = "no_structure.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = labels.size();

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << labels.size() << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    auto data_indices = generate_indices(n_samples);

    using IndicesIterator = decltype(data_indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;

    auto kdtree = ffcl::containers::KDTreeIndexed(
        data_indices.begin(),
        data_indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::containers::KDTreeIndexed<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(10)
            .axis_selection_policy(kdtree::policy::IndexedMaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(/*n_decimals=*/6);

    make_directories(kdtree_folder_root_);

    fs::path kdtree_filename = filename.stem().string() + ".json";

    kdtree.serialize(kdtree_folder_root_ / kdtree_filename);
}

TEST_F(KDTreeIndexedErrorsTest, UnbalancedBlobsTest) {
    fs::path filename = "unbalanced_blobs.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = labels.size();

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << labels.size() << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    auto data_indices = generate_indices(n_samples);

    using IndicesIterator = decltype(data_indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;

    auto kdtree = ffcl::containers::KDTreeIndexed(
        data_indices.begin(),
        data_indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::containers::KDTreeIndexed<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(10)
            .axis_selection_policy(kdtree::policy::IndexedMaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(/*n_decimals=*/6);

    make_directories(kdtree_folder_root_);

    fs::path kdtree_filename = filename.stem().string() + ".json";

    kdtree.serialize(kdtree_folder_root_ / kdtree_filename);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
