// MLPACK comparison placeholder.
//
// MLPACK's DualTreeBoruvka is the closest C++ reference for the MST. It is not installed on this
// machine (it needs armadillo + ensmallen), so the MST is compared against the Python hdbscan library
// instead, whose Boruvka implementation is a direct port of MLPACK's.

#include <ostream>

namespace ffcl::benchmark {

void benchmark_mlpack(std::ostream& results) {
    (void)results;  // reported as unavailable in benchmark/README.md
}

}  // namespace ffcl::benchmark
