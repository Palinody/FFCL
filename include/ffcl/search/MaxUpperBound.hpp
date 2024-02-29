#pragma once

#include "ffcl/common/Utils.hpp"

namespace ffcl::search::algorithms {

template <typename IndicesIterator, typename QueriesToBuffersMap, typename Distance>
auto find_max_upper_bound(const IndicesIterator&     indices_range_first,
                          const IndicesIterator&     indices_range_last,
                          const QueriesToBuffersMap& queries_to_buffers_map,
                          const Distance&            init_max_upper_bound = common::infinity<Distance>()) {
    auto max_upper_bound = init_max_upper_bound;

    for (auto index_it = indices_range_first; index_it != indices_range_last; ++index_it) {
        // Attempt to find the buffer associated with the current index in the buffer map.
        const auto index_to_buffer_it = queries_to_buffers_map.find(*index_it);
        // The max upper bound is infinite if at least one buffer associated with the query is missing.
        // In this case we can return infinity early.
        if (index_to_buffer_it == queries_to_buffers_map.end()) {
            return common::infinity<Distance>();

        } else {
            const auto& buffer_at_index = index_to_buffer_it->second;
            // Similarly to when the buffer corresponding to the current index hasn't been found, we return infinity
            // early if one of the buffer is empty.
            if (buffer_at_index.n_free_slots()) {
                return common::infinity<Distance>();

            } else {
                if (buffer_at_index.upper_bound() > max_upper_bound) {
                    max_upper_bound = buffer_at_index.upper_bound();
                }
            }
        }
    }
    return max_upper_bound;
}

}  // namespace ffcl::search::algorithms