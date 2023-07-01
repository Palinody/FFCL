#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDTreeIndexed.hpp"

#include <cstddef>

namespace ffcl {

template <typename T>
class DBSCAN {
    static_assert(std::is_floating_point<T>::value, "DBSCAN only allows floating point types.");

  public:
    using LabelType = ssize_t;

    enum class PointStatus : LabelType {
        noise              = -1,
        unknown            = 0,
        reachable          = 1,
        directly_reachable = 2,
        core_point         = 3
    };

    struct Options {
        Options& min_samples(std::size_t min_samples) {
            min_samples_ = min_samples;
            return *this;
        }

        Options& epsilon(const T& epsilon) {
            epsilon_ = epsilon;
            return *this;
        }

        Options& operator=(const Options& options) {
            min_samples_ = options.min_samples_;
            epsilon_     = options.epsilon_;
            return *this;
        }

        std::size_t min_samples_ = 5;
        T           epsilon_     = 0.1;
    };

  public:
    DBSCAN(std::size_t n_features);

    DBSCAN(std::size_t n_features, const Options& options);

    DBSCAN(const DBSCAN&) = delete;

    DBSCAN<T>& set_options(const Options& options);

    template <typename Indexer>
    auto predict(const Indexer& indexer) const;

  private:
    // number of features (dimensions) that a DBSCAN instance should handle
    std::size_t n_features_;

    Options options_;
};

template <typename T>
DBSCAN<T>::DBSCAN(std::size_t n_features)
  : n_features_{n_features} {}

template <typename T>
DBSCAN<T>::DBSCAN(std::size_t n_features, const Options& options)
  : n_features_{n_features}
  , options_{options} {}

template <typename T>
DBSCAN<T>& DBSCAN<T>::set_options(const Options& options) {
    options_ = options;
    return *this;
}

template <typename T>
template <typename Indexer>
auto DBSCAN<T>::predict(const Indexer& indexer) const {
    common::utils::ignore_parameters(indexer);

    auto predictions = std::vector<LabelType>();
    return predictions;
}

}  // namespace ffcl