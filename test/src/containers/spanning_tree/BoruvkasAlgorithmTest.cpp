#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/LowerTriangleMatrix.hpp"
#include "ffcl/containers/kdtree/KDTreeIndexed.hpp"
#include "ffcl/containers/spanning_tree/BoruvkasAlgorithm.hpp"

#include <sys/types.h>  // std::ssize_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace fs = std::filesystem;

class BoruvkasAlgorithmErrorsTest : public ::testing::Test {
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

/*
TEST_F(BoruvkasAlgorithmErrorsTest, NoisyCirclesTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "noisy_circles.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator         = decltype(indices)::iterator;
    using SamplesIterator         = decltype(data)::iterator;
    using IndexerType             = ffcl::containers::KDTreeIndexed<IndicesIterator, SamplesIterator>;
    using OptionsType             = IndexerType::Options;
    using AxisSelectionPolicyType = kdtree::policy::IndexedHighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType = kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>;

    timer.reset();

    // IndexedHighestVarianceBuild, IndexedMaximumSpreadBuild, IndexedCycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType())
                                   .splitting_rule_policy(SplittingRulePolicyType()));

    timer.print_elapsed_seconds(9);

    auto boruvkas_algorithm = ffcl::BoruvkasAlgorithm<IndexerType>();

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(3));

    timer.reset();

    const auto minimum_spanning_tree = boruvkas_algorithm.make_tree(
        indexer, &IndexerType::k_nearest_neighbors_around_query_index, &IndexerType::k_mutual_reachability_distance, 3);

    for (const auto& edge : minimum_spanning_tree) {
        std::cout << std::get<0>(edge) << ", " << std::get<1>(edge) << ", " << std::get<2>(edge) << "\n";
    }
    std::cout << "MST size: " << minimum_spanning_tree.size() << "\n";
    timer.print_elapsed_seconds(9);
}
*/

using SampleIndexType = std::size_t;

class ForestPartition {
  public:
    ForestPartition(std::size_t n_samples)
      : n_samples_{n_samples}
      , component_labels_{std::vector<SampleIndexType>(n_samples)}
      , sample_indices_{std::vector<SampleIndexType>(n_samples)}
      , component_sizes_{std::vector<SampleIndexType>(n_samples, 1)}
      , component_offsets_{std::vector<SampleIndexType>(n_samples, 0)}
      , sorting_state_{true} {
        std::iota(component_labels_.begin(), component_labels_.end(), static_cast<SampleIndexType>(0));

        std::iota(sample_indices_.begin(), sample_indices_.end(), static_cast<SampleIndexType>(0));

        update_component_offsets();
    }

    std::size_t n_elements() const {
        return n_samples_;
    }

    SampleIndexType n_components() const {
        return component_sizes_.size();
    }

    auto component_sizes() const {
        return component_sizes_;
    }

    auto component_indices_range(const SampleIndexType& component_index) const {
        return std::make_pair(
            sample_indices_.begin() + component_offsets_[component_index],
            sample_indices_.begin() + component_offsets_[component_index] + component_sizes_[component_index]);
    }

    void update_sample_index_to_component_label(const SampleIndexType& sample_index,
                                                const SampleIndexType& new_component_label) {
        // decrement the number of samples in the previous component that sample_index was mapped with
        --component_sizes_[component_labels_[sample_index]];
        // remap the sample index to the new component label
        component_labels_[sample_index] = new_component_label;
        // increment the number of samples in the new component that sample_index is now mapped with
        ++component_sizes_[component_labels_[sample_index]];
        // sorting_state is now wrong
        sorting_state_ = false;
    }

    void update() {
        group_sample_indices_by_component();
        update_components();
    }

    void print() const {
        std::cout << "component_label:\n";
        for (const auto& component_label : component_labels_) {
            std::cout << component_label << ", ";
        }
        std::cout << "\n";

        std::cout << "sample_index:\n";
        for (const auto& sample_index : sample_indices_) {
            std::cout << sample_index << ", ";
        }
        std::cout << "\n";

        std::cout << "component_size:\n";
        for (const auto& component_size : component_sizes_) {
            std::cout << component_size << ", ";
        }
        std::cout << "\n";

        std::cout << "component_offset:\n";
        for (const auto& component_offset : component_offsets_) {
            std::cout << component_offset << ", ";
        }
        std::cout << "\n";
    }

  private:
    void group_sample_indices_by_component() {
        std::vector<SampleIndexType> indices(n_samples_);
        std::iota(indices.begin(), indices.end(), static_cast<SampleIndexType>(0));

        auto comparator = [this](const auto& index_1, const auto& index_2) {
            return component_labels_[index_1] < component_labels_[index_2];
        };

        std::sort(indices.begin(), indices.end(), comparator);

        auto sorted_component_labels = std::vector<SampleIndexType>(n_samples_);
        auto sorted_sample_indices   = std::vector<SampleIndexType>(n_samples_);

        for (std::size_t index = 0; index < n_samples_; ++index) {
            sorted_component_labels[index] = component_labels_[indices[index]];
            sorted_sample_indices[index]   = sample_indices_[indices[index]];
        }

        component_labels_ = std::move(sorted_component_labels);
        sample_indices_   = std::move(sorted_sample_indices);
    }

    void prune_component_sizes() {
        component_sizes_.erase(std::remove_if(component_sizes_.begin(),
                                              component_sizes_.end(),
                                              [](const auto& component_size) { return !component_size; }),
                               component_sizes_.end());
    }

    void update_component_offsets() {
        if (component_offsets_.size() != component_sizes_.size()) {
            // reset the vector to a default vector with the new number of elements
            component_offsets_ = decltype(component_offsets_)(component_sizes_.size());
        }
        // recompute the offsets
        std::exclusive_scan(component_sizes_.begin(), component_sizes_.end(), component_offsets_.begin(), 0);
    }

    void update_components() {
        prune_component_sizes();
        update_component_offsets();
    }

    std::size_t n_samples_;
    // the component class/label that can range in [0, n_samples) that partitions the sample indices
    std::vector<SampleIndexType> component_labels_;
    // the sample indices that will be rearranged based on the component labels order
    std::vector<SampleIndexType> sample_indices_;
    // the component size for each label that can range in [0, n_samples)
    std::vector<SampleIndexType> component_sizes_;
    // the cumulated sum of the components sizes to retrieve the beginning of each sequence of component
    std::vector<SampleIndexType> component_offsets_;
    // whether the sample indices are sorted w.r.t. the sorted component indices
    bool sorting_state_;
};

template <typename Type>
void print_data(const std::vector<Type>& data, std::size_t n_features) {
    const std::size_t n_samples = data.size() / n_features;

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            std::cout << data[sample_index * n_features + feature_index] << " ";
        }
        std::cout << "\n";
    }
}

#include <cstddef>  // For size_t
#include <iomanip>  // For setw and fixed
#include <iostream>

template <typename Matrix>
void print_matrix(const Matrix& matrix) {
    static constexpr std::size_t integral_cout_width = 3;
    static constexpr std::size_t decimal_cout_width  = 3;

    for (std::size_t sample_index = 0; sample_index < matrix.n_samples(); ++sample_index) {
        for (std::size_t other_sample_index = 0; other_sample_index < matrix.n_samples(); ++other_sample_index) {
            // Set the output format
            std::cout << std::setw(integral_cout_width + decimal_cout_width + 1) << std::fixed
                      << std::setprecision(decimal_cout_width) << matrix(sample_index, other_sample_index) << " ";
        }
        std::cout << "\n";
    }
}

TEST_F(BoruvkasAlgorithmErrorsTest, ForestPartitionTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    using SampleValueType = float;

    auto              data       = std::vector<SampleValueType>({1, 2, 1.5, 2.2, 2.5, 2.9, 2, 3, 4, 2, 3, 3, 3.5, 2.2});
    const std::size_t n_features = 2;
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);
    auto              distance_matrix = ffcl::containers::LowerTriangleMatrix(data.begin(), data.end(), n_features);

    auto nearest_neirbors_index_map = std::vector<SampleIndexType>({});
    print_data(data, n_features);
    std::cout << "---\n";
    print_matrix(distance_matrix);
    std::cout << "---\n";

    auto indices = generate_indices(n_samples);

    timer.reset();

    ForestPartition forest(indices.size());

    forest.print();

    {
        const auto components_sizes = forest.component_sizes();

        for (SampleIndexType component_index = 0; component_index < forest.n_components(); ++component_index) {
            // get the iterator to the first element of the current component and the last
            const auto [component_index_sequence_first, component_index_sequence_last] =
                forest.component_indices_range(component_index);

            std::cout << "Component " << component_index << "\n";
            print_data(std::vector<SampleIndexType>(component_index_sequence_first, component_index_sequence_last), 1);

            auto nn_buffer_with_memory =
                NearestNeighborsBufferWithMemory<typename std::vector<SampleValueType>::iterator>(
                    component_index_sequence_first, component_index_sequence_last, 1);

            for (auto in_component_sample_index_it = component_index_sequence_first;
                 in_component_sample_index_it != component_index_sequence_last;
                 ++in_component_sample_index_it) {
                math::heuristics::k_nearest_neighbors_range(data.begin(),
                                                            data.end(),
                                                            data.begin(),
                                                            data.end(),
                                                            n_features,
                                                            *in_component_sample_index_it,
                                                            nn_buffer_with_memory);

                const auto nearest_neighbor_index    = nn_buffer_with_memory.furthest_k_nearest_neighbor_index();
                const auto nearest_neighbor_distance = nn_buffer_with_memory.furthest_k_nearest_neighbor_distance();

                std::cout << "query: " << *in_component_sample_index_it << ", nn_index: " << nearest_neighbor_index
                          << ", nn_distance: " << nearest_neighbor_distance << "\n";
            }
        }
        // merge components
    }
    timer.print_elapsed_seconds(9);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
