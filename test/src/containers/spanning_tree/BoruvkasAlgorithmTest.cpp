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
using SampleValueType = float;
using EdgeType        = std::tuple<SampleIndexType, SampleIndexType, SampleValueType>;

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

    auto get_sample_index_component(const SampleIndexType& sample_index) const {
        return component_labels_[sample_index];
    }

    void merge_components(const EdgeType& edge) {
        const auto sample_index_1 = std::get<0>(edge);
        const auto sample_index_2 = std::get<1>(edge);
        const auto component_1    = component_labels_[sample_index_1];
        const auto component_2    = component_labels_[sample_index_2];

        if (component_1 == component_2) {
            return;
        }
        // range of labels from the first component
        const auto component_range_1_first = component_labels_.begin() + component_offsets_[component_1];
        const auto component_range_1_last =
            component_labels_.begin() + component_offsets_[component_1] + component_sizes_[component_1];
        // range of labels from the second component
        const auto component_range_2_first = component_labels_.begin() + component_offsets_[component_2];
        const auto component_range_2_last =
            component_labels_.begin() + component_offsets_[component_2] + component_sizes_[component_2];

        const auto component_1_size = std::distance(component_range_1_first, component_range_1_last);
        const auto component_2_size = std::distance(component_range_2_first, component_range_2_last);

        // the final component label that will unite the parent components
        const auto new_component = (component_1_size > component_2_size) ? component_1 : component_2;
        // update the labels of the shortest component with the label of the longest one
        // then decrement the size of the component that havent been chosen
        // and increment the size of the component that has been chosent accordingly
        if (new_component == component_1) {
            std::fill(component_range_2_first, component_range_2_last, new_component);
            component_sizes_[component_2] -= component_2_size;
            component_sizes_[component_1] += component_1_size;
        }
        if (new_component == component_2) {
            std::fill(component_range_1_first, component_range_1_last, new_component);
            component_sizes_[component_1] -= component_1_size;
            component_sizes_[component_2] += component_2_size;
        }

        // update();
    }
    /*
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
    */

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
        // prune_component_sizes();
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

    auto data = std::vector<SampleValueType>({1, 2, 1.5, 2.2, 2.5, 2.9, 2, 3, 4, 2, 3, 3, 3.5, 2.2, 2.3, 2});
    const std::size_t n_features      = 2;
    const std::size_t n_samples       = common::utils::get_n_samples(data.begin(), data.end(), n_features);
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

        // keep track of the shortest edge from a component's sample index to a sample index thats not within the
        // same component
        auto closest_edges = std::vector<EdgeType>(forest.n_components());

        for (SampleIndexType component_index = 0; component_index < forest.n_components(); ++component_index) {
            // get the iterator to the first element of the current component and the last
            const auto [component_range_first, component_range_last] = forest.component_indices_range(component_index);

            std::cout << "Component " << component_index << "\nIndices: ";
            print_data(std::vector<SampleIndexType>(component_range_first, component_range_last), 1);

            auto nn_buffer_with_memory =
                NearestNeighborsBufferWithMemory<typename std::vector<SampleValueType>::iterator>(
                    component_range_first, component_range_last, 1);

            // initialize the closest edge from the current comonent to infinity
            closest_edges[component_index] = EdgeType{0, 0, common::utils::infinity<SampleValueType>()};

            for (auto component_range_it = component_range_first; component_range_it != component_range_last;
                 ++component_range_it) {
                math::heuristics::k_nearest_neighbors_range(data.begin(),
                                                            data.end(),
                                                            data.begin(),
                                                            data.end(),
                                                            n_features,
                                                            *component_range_it,
                                                            nn_buffer_with_memory);

                const auto nearest_neighbor_index    = nn_buffer_with_memory.furthest_k_nearest_neighbor_index();
                const auto nearest_neighbor_distance = nn_buffer_with_memory.furthest_k_nearest_neighbor_distance();

                if (nearest_neighbor_distance < std::get<2>(closest_edges[component_index])) {
                    closest_edges[component_index] =
                        EdgeType{*component_range_it, nearest_neighbor_index, nearest_neighbor_distance};
                }

                std::cout << "query: " << *component_range_it << ", nn_index: " << nearest_neighbor_index
                          << ", nn_distance: " << nearest_neighbor_distance << "\n";
                std::cout << "from component: " << forest.get_sample_index_component(*component_range_it)
                          << " to component: " << forest.get_sample_index_component(nearest_neighbor_index) << "\n";
            }
            std::cout << "\n";
        }
        // merge components
        std::cout << "---\n\n";

        for (const auto& edge : closest_edges) {
            forest.merge_components(edge);
        }
    }
    forest.print();

    timer.print_elapsed_seconds(9);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
