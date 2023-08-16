#pragma once

#include "ffcl/common/Utils.hpp"

#include <sys/types.h>  // std::ssize_t
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace fs = std::filesystem;

namespace bench::io {

using DataType = float;

const fs::path folder_root        = fs::path("../bin/clustering");
const fs::path inputs_folder      = folder_root / fs::path("inputs");
const fs::path targets_folder     = folder_root / fs::path("targets");
const fs::path predictions_folder = folder_root / fs::path("predictions");
const fs::path centroids_folder   = folder_root / fs::path("centroids");
const fs::path conversions_folder = folder_root / fs::path("conversions");

typedef std::unique_ptr<std::FILE, int (*)(std::FILE*)> unique_fp;

unique_fp make_unique_fp(const char* filename, const char* flags) {
    return unique_fp(std::fopen(filename, flags), std::fclose);
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

auto get_files_names_at_path(const fs::path& directory_path) {
    std::vector<fs::path> file_names;

    try {
        for (const auto& entry : fs::directory_iterator(directory_path)) {
            if (fs::is_regular_file(entry)) {
                file_names.emplace_back(entry.path().filename());
            }
        }
    } catch (const fs::filesystem_error& ex) {
        std::cerr << "Error accessing directory: " << ex.what() << std::endl;
        return std::vector<fs::path>{};
    }
    return file_names;
}

namespace txt {

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

template <typename ElementsType = float>
std::vector<ElementsType> load_data(const fs::path& filename, char delimiter = ' ') {
    std::ifstream             filestream(filename);
    std::vector<ElementsType> data;

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

template <typename ElementsType = float>
void write_data(const std::vector<ElementsType>& data, std::size_t n_features, const fs::path& filename) {
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

}  // namespace txt

namespace bin {

template <typename ElementsType = float>
std::uintmax_t n_element(const fs::path& filepath) {
    return fs::file_size(filepath) / sizeof(ElementsType);
}

template <typename ElementsType = float>
std::tuple<std::vector<ElementsType>, std::size_t, std::size_t> decode(std::size_t     n_features,
                                                                       const fs::path& filepath) {
    // count the number of points in the pointcloud file
    const auto n_elements = static_cast<std::size_t>(n_element<ElementsType>(filepath));
    // initiate read
    auto fp = make_unique_fp(filepath.c_str(), "rb");
    if (!fp.get()) {
        std::cout << "Could not decode file at: " << filepath << std::endl;
        abort();
    }

    auto xyzr = std::vector<ElementsType>(n_elements);

    if (!fread(xyzr.data(), sizeof(ElementsType), n_elements, fp.get())) {
        std::cout << "Could not load data from file at: " << filepath << std::endl;
        abort();
    }
    const std::size_t n_samples = common::utils::get_n_samples(xyzr.begin(), xyzr.end(), n_features);
    return {xyzr, n_samples, n_features};
}

template <typename ElementsType>
void encode(const std::vector<ElementsType>& data, const fs::path& filepath) {
    static_assert(std::is_trivial<ElementsType>::value, "ElementType must be a trivially copyable type");

    auto fp = make_unique_fp(filepath.c_str(), "wb");
    if (!fwrite(data.data(), sizeof(ElementsType), data.size(), fp.get())) {
        std::cout << "Could not write data to file at: " << filepath << std::endl;
        abort();
    }
}

}  // namespace bin

}  // namespace bench::io