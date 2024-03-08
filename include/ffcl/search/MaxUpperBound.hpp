#pragma once

#include "ffcl/common/Utils.hpp"

namespace ffcl::search::algorithms {

template <typename IndicesIterator,
          typename SamplesIterator,
          typename QueriesToBuffersMap,
          typename Distance,
          typename... BufferArgs>
auto find_or_make_max_upper_bound(const IndicesIterator& indices_range_first,
                                  const IndicesIterator& indices_range_last,
                                  const SamplesIterator& samples_range_first,
                                  const SamplesIterator& samples_range_last,
                                  std::size_t            n_features,
                                  QueriesToBuffersMap&   queries_to_buffers_map,
                                  const Distance&        initial_max_upper_bound,
                                  BufferArgs&&... buffer_args) {
    common::ignore_parameters(samples_range_last);

    auto max_upper_bound = initial_max_upper_bound;

    for (auto index_it = indices_range_first; index_it != indices_range_last; ++index_it) {
        // Attempt to find the buffer associated with the current index in the buffer map.
        const auto index_to_buffer_it =
            queries_to_buffers_map.find_or_make_buffer_at(*index_it,
                                                          samples_range_first + *index_it * n_features,
                                                          samples_range_first + *index_it * n_features + n_features,
                                                          std::forward<BufferArgs>(buffer_args)...);

        const auto& buffer_at_index = index_to_buffer_it->second;

        if (buffer_at_index.furthest_distance() > max_upper_bound) {
            max_upper_bound = buffer_at_index.furthest_distance();
        }
    }
    return max_upper_bound;
}

template <typename IndicesIterator, typename QueriesToBuffersMap, typename Distance>
auto find_max_upper_bound(const IndicesIterator&     indices_range_first,
                          const IndicesIterator&     indices_range_last,
                          const QueriesToBuffersMap& queries_to_buffers_map,
                          const Distance&            initial_max_upper_bound = Distance{0}) {
    auto max_upper_bound = initial_max_upper_bound;

    for (auto index_it = indices_range_first; index_it != indices_range_last; ++index_it) {
        // Attempt to find the buffer associated with the current index in the buffer map.
        const auto index_to_buffer_it = queries_to_buffers_map.find(*index_it);

        const auto& buffer_at_index = index_to_buffer_it->second;

        if (buffer_at_index.furthest_distance() > max_upper_bound) {
            max_upper_bound = buffer_at_index.furthest_distance();
        }
    }
    return max_upper_bound;
}

}  // namespace ffcl::search::algorithms