#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ffcl/common/Timer.hpp"
#include "ffcl/datastruct/kdtree/KDTree.hpp"
#include "ffcl/math/random/Distributions.hpp"

#include <sys/types.h>  // std::ssize_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

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

    const std::size_t n_neighbors_ = 5;
    const dType       radius_      = 1;
};

std::vector<std::size_t> generate_indices(std::size_t n_samples) {
    std::vector<std::size_t> elements(n_samples);
    std::iota(elements.begin(), elements.end(), static_cast<std::size_t>(0));
    return elements;
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

/*
TEST_F(KDTreeErrorsTest, SequentialNearestNeighborIndexTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    ssize_t current_nearest_neighbor_index    = -1;
    auto    current_nearest_neighbor_distance = common::utils::infinity<dType>();

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        ffcl::knn::nearest_neighbor(indices.begin(),
                                    indices.end(),
                                    data.begin(),
                                    data.end(),
                                    n_features,
                                    indices[sample_index_query],
                                    current_nearest_neighbor_index,
                                    current_nearest_neighbor_distance);
    }
    timer.print_elapsed_seconds(9);

    printf("Dummy print (sequential): %ld, %d\n", current_nearest_neighbor_index, current_nearest_neighbor_distance);
}
*/

TEST_F(KDTreeErrorsTest, NearestNeighborIndexTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

    ssize_t nn_index    = -1;
    auto    nn_distance = common::utils::infinity<dType>();

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        std::tie(nn_index, nn_distance) = kdtree.nearest_neighbor_around_query_index(indices[sample_index_query]);
    }
    timer.print_elapsed_seconds(9);

    std::cout << "nn_index: " << nn_index << ", nn_distance: " << nn_distance << "\n";
}

TEST_F(KDTreeErrorsTest, KNearestNeighborsIndexTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

    ssize_t nn_index    = -1;
    auto    nn_distance = common::utils::infinity<dType>();

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        std::tie(nn_index, nn_distance) = kdtree.nearest_neighbor_around_query_index(indices[sample_index_query]);
    }
    timer.print_elapsed_seconds(9);

    const std::size_t n_neighbors = n_neighbors_;

    std::vector<std::size_t> nn_histogram(n_samples);

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        auto nearest_neighbors_buffer =
            kdtree.k_nearest_neighbors_around_query_index(indices[sample_index_query], n_neighbors);

        nn_histogram[indices[sample_index_query]] = nearest_neighbors_buffer.size();
    }
    timer.print_elapsed_seconds(9);

    std::cout << "nearest_neighbor_around_query_index (index, distance): (" << nn_index << ", " << nn_distance << ")\n";

    // std::cout << "k_nearest_neighbors_around_query_index (histogram)\n";
    // print_data(nn_histogram, n_samples);
}

TEST_F(KDTreeErrorsTest, RadiusCountIndexTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

    std::size_t radius_count = 0;
    const dType radius       = radius_;

    std::vector<std::size_t> nn_histogram(n_samples);

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        radius_count = kdtree.radius_count_around_query_index(indices[sample_index_query], radius);

        nn_histogram[indices[sample_index_query]] = radius_count;
    }
    timer.print_elapsed_seconds(9);

    std::cout << "radius_count: " << radius_count << ", radius: " << radius << "\n";

    // std::cout << "radius_count_around_query_index (histogram)\n";
    // print_data(nn_histogram, n_samples);
}

TEST_F(KDTreeErrorsTest, KNearestNeighborsInRadiusIndexTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

    const dType radius = radius_;

    std::vector<std::size_t> nn_histogram(n_samples);

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        auto nearest_neighbors_buffer = kdtree.radius_search_around_query_index(indices[sample_index_query], radius);

        nn_histogram[indices[sample_index_query]] = nearest_neighbors_buffer.size();
    }
    timer.print_elapsed_seconds(9);

    // std::cout << "radius_search_around_query_index (histogram)\n";
    // print_data(nn_histogram, n_samples);
}

TEST_F(KDTreeErrorsTest, NearestNeighborIndexWithUnknownSampleTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

    ssize_t nn_index    = -1;
    auto    nn_distance = common::utils::infinity<dType>();

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        std::tie(nn_index, nn_distance) = kdtree.nearest_neighbor_around_query_sample(
            data.begin() + indices[sample_index_query] * n_features,
            data.begin() + indices[sample_index_query] * n_features + n_features);
    }
    timer.print_elapsed_seconds(9);

    std::cout << "nn_index: " << nn_index << ", nn_distance: " << nn_distance << "\n";
}

TEST_F(KDTreeErrorsTest, KNearestNeighborsSampleTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

    ssize_t nn_index    = -1;
    auto    nn_distance = common::utils::infinity<dType>();

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        std::tie(nn_index, nn_distance) = kdtree.nearest_neighbor_around_query_sample(
            data.begin() + indices[sample_index_query] * n_features,
            data.begin() + indices[sample_index_query] * n_features + n_features);
    }
    timer.print_elapsed_seconds(9);

    std::vector<std::size_t> nn_indices;
    std::vector<dType>       nn_distances;
    const std::size_t        n_neighbors = n_neighbors_;

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        auto nearest_neighbors_buffer = kdtree.k_nearest_neighbors_around_query_sample(
            data.begin() + indices[sample_index_query] * n_features,
            data.begin() + indices[sample_index_query] * n_features + n_features,
            n_neighbors);

        std::tie(nn_indices, nn_distances) = nearest_neighbors_buffer.move_data_to_indices_distances_pair();
    }
    timer.print_elapsed_seconds(9);

    std::cout << "nearest_neighbor_around_query_sample (index, distance): (" << nn_index << ", " << nn_distance
              << ")\n";

    std::cout << "k_nearest_neighbors_around_query_sample (indices, distances)\n";
    print_data(nn_indices, nn_indices.size());
    print_data(nn_distances, nn_distances.size());
}

TEST_F(KDTreeErrorsTest, RadiusCountSampleTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

    std::size_t radius_count = 0;
    const dType radius       = radius_;

    std::size_t total_neighbors = 0;

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        radius_count = kdtree.radius_count_around_query_sample(
            data.begin() + indices[sample_index_query] * n_features,
            data.begin() + indices[sample_index_query] * n_features + n_features,
            radius);

        total_neighbors += radius_count;
    }
    timer.print_elapsed_seconds(9);

    std::cout << "Radius: " << radius << "\nTotal neighbors number: " << total_neighbors
              << ", average neighbors number: " << total_neighbors / float(n_samples) << "\n";
}

TEST_F(KDTreeErrorsTest, KNearestNeighborsInRadiusSampleTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

    std::vector<std::size_t> nn_indices;
    std::vector<dType>       nn_distances;
    const dType              radius = radius_;

    std::size_t total_neighbors = 0;

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        auto nearest_neighbors_buffer = kdtree.radius_search_around_query_sample(
            data.begin() + indices[sample_index_query] * n_features,
            data.begin() + indices[sample_index_query] * n_features + n_features,
            radius);

        std::tie(nn_indices, nn_distances) = nearest_neighbors_buffer.move_data_to_indices_distances_pair();

        total_neighbors += nn_indices.size();
    }
    timer.print_elapsed_seconds(9);

    std::cout << "radius_search_around_query_index (indices, distances)\n";
    print_data(nn_indices, nn_indices.size());
    print_data(nn_distances, nn_distances.size());
    std::cout << "Total neighbors number: " << total_neighbors
              << ", average neighbors number: " << total_neighbors / float(n_samples) << "\n";
}

TEST_F(KDTreeErrorsTest, KDBoundingBoxCountIndexTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

    std::size_t radius_count    = 0;
    const auto  kd_bounding_box = ffcl::bbox::HyperRangeType<SamplesIterator>{{-0.1, 0.1}, {-0.1, 0.1}};

    std::size_t total_neighbors = 0;

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        radius_count = kdtree.range_count_around_query_index(indices[sample_index_query], kd_bounding_box);

        total_neighbors += radius_count;
    }
    timer.print_elapsed_seconds(9);

    std::cout << "Total neighbors number in bounding box: " << total_neighbors
              << ", average neighbors number: " << total_neighbors / float(n_samples) << "\n";
}

TEST_F(KDTreeErrorsTest, KDBoundingBoxSearchIndexTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

    std::vector<std::size_t> nn_indices;
    std::vector<dType>       nn_distances;

    const auto kd_bounding_box = ffcl::bbox::HyperRangeType<SamplesIterator>{{-0.1, 0.1}, {-0.1, 0.1}};

    std::size_t total_neighbors = 0;

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        auto nearest_neighbors_buffer =
            kdtree.range_search_around_query_index(indices[sample_index_query], kd_bounding_box);

        std::tie(nn_indices, nn_distances) = nearest_neighbors_buffer.move_data_to_indices_distances_pair();

        total_neighbors += nn_indices.size();
    }
    timer.print_elapsed_seconds(9);

    std::cout << "range_search_around_query_index (indices, distances)\n";
    print_data(nn_indices, nn_indices.size());
    print_data(nn_distances, nn_distances.size());
    std::cout << "Total neighbors number: " << total_neighbors
              << ", average neighbors number: " << total_neighbors / float(n_samples) << "\n";
}

TEST_F(KDTreeErrorsTest, KDBoundingBoxCountSampleTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

    std::size_t radius_count    = 0;
    const auto  kd_bounding_box = ffcl::bbox::HyperRangeType<SamplesIterator>{{-0.1, 0.1}, {-0.1, 0.1}};

    std::size_t total_neighbors = 0;

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        radius_count =
            kdtree.range_count_around_query_sample(data.begin() + indices[sample_index_query] * n_features,
                                                   data.begin() + indices[sample_index_query] * n_features + n_features,
                                                   kd_bounding_box);

        total_neighbors += radius_count;
    }
    timer.print_elapsed_seconds(9);

    std::cout << "Total neighbors number in bounding box: " << total_neighbors
              << ", average neighbors number: " << total_neighbors / float(n_samples) << "\n";
}

TEST_F(KDTreeErrorsTest, KDBoundingBoxSearchSampleTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const std::size_t n_samples = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    std::cout << "n_elements: " << data.size() << "\n";
    std::cout << "n_samples: " << n_samples << "\n";
    std::cout << "n_features: " << n_features << "\n";

    printf("Making the kdtree:\n");

    timer.reset();

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

    std::vector<std::size_t> nn_indices;
    std::vector<dType>       nn_distances;

    const auto kd_bounding_box = ffcl::bbox::HyperRangeType<SamplesIterator>{{-0.1, 0.1}, {-0.1, 0.1}};

    std::size_t total_neighbors = 0;

    timer.reset();
    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        auto nearest_neighbors_buffer = kdtree.range_search_around_query_sample(
            data.begin() + indices[sample_index_query] * n_features,
            data.begin() + indices[sample_index_query] * n_features + n_features,
            kd_bounding_box);

        std::tie(nn_indices, nn_distances) = nearest_neighbors_buffer.move_data_to_indices_distances_pair();

        total_neighbors += nn_indices.size();
    }
    timer.print_elapsed_seconds(9);

    std::cout << "range_search_around_query_sample (indices, distances)\n";
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

    auto indices         = generate_indices(n_samples);
    auto feature_indices = generate_indices(n_features);

    std::shuffle(feature_indices.begin(), feature_indices.end(), std::mt19937{std::random_device{}()});

    timer.reset();
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(
                kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>().feature_mask(feature_indices))
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);
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

    auto indices = generate_indices(n_samples);

    timer.reset();
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;

    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

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

    auto indices = generate_indices(n_samples);

    timer.reset();
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;

    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

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

    auto indices = generate_indices(n_samples);

    timer.reset();
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;

    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

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

    auto indices = generate_indices(n_samples);

    timer.reset();
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;

    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

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

    auto indices = generate_indices(n_samples);

    timer.reset();
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;

    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

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

    auto indices = generate_indices(n_samples);

    timer.reset();
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;

    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

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

    auto indices = generate_indices(n_samples);

    timer.reset();
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;

    auto kdtree = ffcl::datastruct::KDTree(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>::Options()
            .bucket_size(std::sqrt(n_samples))
            .max_depth(std::log2(n_samples))
            .axis_selection_policy(kdtree::policy::MaximumSpreadBuild<IndicesIterator, SamplesIterator>())
            .splitting_rule_policy(kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>()));

    timer.print_elapsed_seconds(9);

    make_directories(kdtree_folder_root_);

    fs::path kdtree_filename = filename.stem().string() + ".json";

    kdtree.serialize(kdtree_folder_root_ / kdtree_filename);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
