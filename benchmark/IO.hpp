#pragma once

#include <sys/types.h>  // std::ssize_t
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace fs = std::filesystem;

using dType = float;

const fs::path folder_root        = fs::path("../bin/clustering");
const fs::path inputs_folder      = folder_root / fs::path("inputs");
const fs::path targets_folder     = folder_root / fs::path("targets");
const fs::path predictions_folder = folder_root / fs::path("predictions");
const fs::path centroids_folder   = folder_root / fs::path("centroids");
const fs::path conversions_folder = folder_root / fs::path("conversions");

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