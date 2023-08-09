#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ffcl/common/Timer.hpp"
#include "ffcl/containers/kdtree/KDTree.hpp"

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

class KDTreeErrorsTest : public ::testing::Test {
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

template <typename DataType>
void print_kd_bounding_box(const std::vector<std::pair<DataType, DataType>>& kd_bounding_box, std::size_t n_features) {
    for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
        std::cout << kd_bounding_box[feature_index].first << ", " << kd_bounding_box[feature_index].second << "\n";
    }
}

TEST_F(KDTreeErrorsTest, KDBoundingBoxTest) {
    using DataType = float;

    std::vector<DataType> dataset = {/*row0*/
                                     1,
                                     2,
                                     -1,
                                     /*row1*/
                                     -300,
                                     1,
                                     2,
                                     /*row2*/
                                     2,
                                     8,
                                     0};

    const auto n_features = 3;

    const auto kd_bounding_box = bbox::make_kd_bounding_box(dataset.begin(), dataset.end(), n_features);

    print_kd_bounding_box<DataType>(kd_bounding_box, n_features);

    ssize_t axis = kdtree::algorithms::select_axis_with_largest_bounding_box_difference<decltype(dataset.begin())>(
        kd_bounding_box);

    std::cout << axis << "\n";
}

template <typename Type>
void print_data(const std::vector<Type>& data, std::size_t n_features) {
    if (!n_features) {
        return;
    }
    const std::size_t n_samples = data.size() / n_features;

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            std::cout << data[sample_index * n_features + feature_index] << " ";
        }
        std::cout << "\n";
    }
}

TEST_F(KDTreeErrorsTest, SequentialNearestNeighborIndexTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    ssize_t current_nearest_neighbor_index    = 0;
    dType   current_nearest_neighbor_distance = common::utils::infinity<dType>();

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        math::heuristics::nearest_neighbor_range(data.begin(),
                                                 data.end(),
                                                 data.begin(),
                                                 data.end(),
                                                 n_features,
                                                 sample_index_query,
                                                 current_nearest_neighbor_index,
                                                 current_nearest_neighbor_distance);

        // common::utils::ignore_parameters(current_nearest_neighbor_index, current_nearest_neighbor_distance);
    }
    timer.print_elapsed_seconds(9);

    printf("Dummy print (sequential): %ld, %.9f\n", current_nearest_neighbor_index, current_nearest_neighbor_distance);
}

TEST_F(KDTreeErrorsTest, NearestNeighborTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree =
        ffcl::containers::KDTree(data.begin(),
                                 data.end(),
                                 n_features,
                                 ffcl::containers::KDTree<SamplesIterator>::Options()
                                     .bucket_size(std::sqrt(n_samples))
                                     .max_depth(std::log2(n_samples))
                                     .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<SamplesIterator>())
                                     .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<SamplesIterator>()));
    timer.print_elapsed_seconds(9);

    std::size_t nn_index    = 0;
    auto        nn_distance = common::utils::infinity<dType>();

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        std::tie(nn_index, nn_distance) = kdtree.nearest_neighbor_around_query_index(sample_index_query);
    }
    timer.print_elapsed_seconds(9);

    printf("Dummy print (kdtree): %ld, %.3f\n", nn_index, nn_distance);
}

TEST_F(KDTreeErrorsTest, KNearestNeighborsTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree =
        ffcl::containers::KDTree(data.begin(),
                                 data.end(),
                                 n_features,
                                 ffcl::containers::KDTree<SamplesIterator>::Options()
                                     .bucket_size(std::sqrt(n_samples))
                                     .max_depth(std::log2(n_samples))
                                     .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<SamplesIterator>())
                                     .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<SamplesIterator>()));
    timer.print_elapsed_seconds(9);

    std::size_t nn_index;
    dType       nn_distance;

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        std::tie(nn_index, nn_distance) = kdtree.nearest_neighbor_around_query_index(sample_index_query);
    }
    timer.print_elapsed_seconds(9);

    std::vector<std::size_t> nn_indices;
    std::vector<dType>       nn_distances;

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        std::tie(nn_indices, nn_distances) = kdtree.k_nearest_neighbors_around_query_index(sample_index_query, 5);
    }
    timer.print_elapsed_seconds(9);

    std::cout << "nearest_neighbor_around_query_index (index, distance): (" << nn_index << ", " << nn_distance << ")\n";

    std::cout << "k_nearest_neighbors_around_query_index (indices, distances)\n";
    print_data(nn_indices, nn_indices.size());
    print_data(nn_distances, nn_distances.size());
}

TEST_F(KDTreeErrorsTest, RadiusCountTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree =
        ffcl::containers::KDTree(data.begin(),
                                 data.end(),
                                 n_features,
                                 ffcl::containers::KDTree<SamplesIterator>::Options()
                                     .bucket_size(std::sqrt(n_samples))
                                     .max_depth(std::log2(n_samples))
                                     .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<SamplesIterator>())
                                     .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<SamplesIterator>()));
    timer.print_elapsed_seconds(9);

    std::size_t radius_count = 0;
    dType       radius       = 1;  // common::utils::infinity<dType>();

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        radius_count = kdtree.radius_count_around_query_index(sample_index_query, radius);
    }
    timer.print_elapsed_seconds(9);

    printf("Dummy print (kdtree): number of neighbors: %ld, radius: %.3f\n", radius_count, radius);
}

TEST_F(KDTreeErrorsTest, KNearestNeighborsInRadiusTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree =
        ffcl::containers::KDTree(data.begin(),
                                 data.end(),
                                 n_features,
                                 ffcl::containers::KDTree<SamplesIterator>::Options()
                                     .bucket_size(std::sqrt(n_samples))
                                     .max_depth(std::log2(n_samples))
                                     .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<SamplesIterator>())
                                     .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<SamplesIterator>()));
    timer.print_elapsed_seconds(9);

    std::vector<std::size_t> nn_indices;
    std::vector<dType>       nn_distances;
    dType                    radius = 0.2;

    std::size_t total_neighbors = 0;

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        std::tie(nn_indices, nn_distances) = kdtree.radius_search_around_query_index(sample_index_query, radius);

        total_neighbors += nn_indices.size();
    }
    timer.print_elapsed_seconds(9);

    std::cout << "radius_search_around_query_index (indices, distances)\n";
    print_data(nn_indices, nn_indices.size());
    print_data(nn_distances, nn_distances.size());
    std::cout << "Total neighbors number: " << total_neighbors
              << ", average neighbors number: " << total_neighbors / float(n_samples) << "\n";
}

TEST_F(KDTreeErrorsTest, MNISTTest) {
    fs::path filename = "mnist.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree =
        ffcl::containers::KDTree(data.begin(),
                                 data.end(),
                                 n_features,
                                 ffcl::containers::KDTree<SamplesIterator>::Options()
                                     .bucket_size(std::sqrt(n_samples))
                                     .max_depth(std::log2(n_samples))
                                     .axis_selection_policy(kdtree::policy::HighestVarianceBuild<SamplesIterator>())
                                     .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<SamplesIterator>()));

    timer.print_elapsed_seconds(6);
}

TEST_F(KDTreeErrorsTest, NoisyCirclesTest) {
    fs::path filename = "noisy_circles.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    using SamplesIterator = decltype(data)::iterator;

    auto kdtree =
        ffcl::containers::KDTree(data.begin(),
                                 data.end(),
                                 n_features,
                                 ffcl::containers::KDTree<SamplesIterator>::Options()
                                     .bucket_size(std::sqrt(n_samples))
                                     .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<SamplesIterator>())
                                     .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<SamplesIterator>()));

    timer.print_elapsed_seconds(/*n_decimals=*/6);

    make_directories(kdtree_folder_root_);

    fs::path kdtree_filename = filename.stem().string() + ".json";

    kdtree.serialize(kdtree_folder_root_ / kdtree_filename);
}

TEST_F(KDTreeErrorsTest, NoisyMoonsTest) {
    fs::path filename = "noisy_moons.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    using SamplesIterator = decltype(data)::iterator;

    auto kdtree =
        ffcl::containers::KDTree(data.begin(),
                                 data.end(),
                                 n_features,
                                 ffcl::containers::KDTree<SamplesIterator>::Options()
                                     .bucket_size(std::sqrt(n_samples))
                                     .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<SamplesIterator>())
                                     .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<SamplesIterator>()));

    timer.print_elapsed_seconds(/*n_decimals=*/6);

    make_directories(kdtree_folder_root_);

    fs::path kdtree_filename = filename.stem().string() + ".json";

    kdtree.serialize(kdtree_folder_root_ / kdtree_filename);
}

TEST_F(KDTreeErrorsTest, VariedTest) {
    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    using SamplesIterator = decltype(data)::iterator;

    auto kdtree =
        ffcl::containers::KDTree(data.begin(),
                                 data.end(),
                                 n_features,
                                 ffcl::containers::KDTree<SamplesIterator>::Options()
                                     .bucket_size(std::sqrt(n_samples))
                                     .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<SamplesIterator>())
                                     .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<SamplesIterator>()));

    timer.print_elapsed_seconds(/*n_decimals=*/6);

    make_directories(kdtree_folder_root_);

    fs::path kdtree_filename = filename.stem().string() + ".json";

    kdtree.serialize(kdtree_folder_root_ / kdtree_filename);
}

TEST_F(KDTreeErrorsTest, AnisoTest) {
    fs::path filename = "aniso.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    using SamplesIterator = decltype(data)::iterator;

    auto kdtree =
        ffcl::containers::KDTree(data.begin(),
                                 data.end(),
                                 n_features,
                                 ffcl::containers::KDTree<SamplesIterator>::Options()
                                     .bucket_size(std::sqrt(n_samples))
                                     .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<SamplesIterator>())
                                     .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<SamplesIterator>()));

    timer.print_elapsed_seconds(/*n_decimals=*/6);

    make_directories(kdtree_folder_root_);

    fs::path kdtree_filename = filename.stem().string() + ".json";

    kdtree.serialize(kdtree_folder_root_ / kdtree_filename);
}

TEST_F(KDTreeErrorsTest, BlobsTest) {
    fs::path filename = "blobs.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    using SamplesIterator = decltype(data)::iterator;

    auto kdtree =
        ffcl::containers::KDTree(data.begin(),
                                 data.end(),
                                 n_features,
                                 ffcl::containers::KDTree<SamplesIterator>::Options()
                                     .bucket_size(std::sqrt(n_samples))
                                     .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<SamplesIterator>())
                                     .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<SamplesIterator>()));

    timer.print_elapsed_seconds(/*n_decimals=*/6);

    make_directories(kdtree_folder_root_);

    fs::path kdtree_filename = filename.stem().string() + ".json";

    kdtree.serialize(kdtree_folder_root_ / kdtree_filename);
}

TEST_F(KDTreeErrorsTest, NoStructureTest) {
    fs::path filename = "no_structure.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    using SamplesIterator = decltype(data)::iterator;

    auto kdtree =
        ffcl::containers::KDTree(data.begin(),
                                 data.end(),
                                 n_features,
                                 ffcl::containers::KDTree<SamplesIterator>::Options()
                                     .bucket_size(std::sqrt(n_samples))
                                     .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<SamplesIterator>())
                                     .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<SamplesIterator>()));

    timer.print_elapsed_seconds(/*n_decimals=*/6);

    make_directories(kdtree_folder_root_);

    fs::path kdtree_filename = filename.stem().string() + ".json";

    kdtree.serialize(kdtree_folder_root_ / kdtree_filename);
}

TEST_F(KDTreeErrorsTest, UnbalancedBlobsTest) {
    fs::path filename = "unbalanced_blobs.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    common::timer::Timer<common::timer::Nanoseconds> timer;

    using SamplesIterator = decltype(data)::iterator;

    auto kdtree =
        ffcl::containers::KDTree(data.begin(),
                                 data.end(),
                                 n_features,
                                 ffcl::containers::KDTree<SamplesIterator>::Options()
                                     .bucket_size(std::sqrt(n_samples))
                                     .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<SamplesIterator>())
                                     .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<SamplesIterator>()));

    timer.print_elapsed_seconds(/*n_decimals=*/6);

    make_directories(kdtree_folder_root_);

    fs::path kdtree_filename = filename.stem().string() + ".json";

    kdtree.serialize(kdtree_folder_root_ / kdtree_filename);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}