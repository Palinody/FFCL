#pragma once

#include "ffcl/common/Utils.hpp"

namespace ffcl::search {

template <typename IndicesIterator, typename SamplesIterator, typename QueriesToBuffersMap>
void find_max_upper_bound(const IndicesIterator&                            indices_range_first,
                          const IndicesIterator&                            indices_range_last,
                          const SamplesIterator&                            samples_range_first,
                          const SamplesIterator&                            samples_range_last,
                          std::size_t                                       n_features,
                          QueriesToBuffersMap&                              queries_to_buffers_map,
                          const typename QueriesToBuffersMap::DistanceType& initial_max_upper_bound = 0) {
    auto max_upper_bound = initial_max_upper_bound;

    for (auto index_it = indices_range_first; index_it != indices_range_last; ++index_it) {
        const auto& buffer_at_index = queries_to_buffers_map
                                          .find_or_make_buffer_at(/**/ *index_it,
                                                                  /**/ samples_range_first,
                                                                  /**/ samples_range_last,
                                                                  /**/ n_features)
                                          ->second;

        if (buffer_at_index.upper_bound() > max_upper_bound) {
            max_upper_bound = buffer_at_index.upper_bound();
        }
    }
    return max_upper_bound;
}

}  // namespace ffcl::search
