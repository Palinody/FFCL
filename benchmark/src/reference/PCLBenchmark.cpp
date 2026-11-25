// PCL comparison placeholder.
//
// The PCL and FLANN KD-trees are compared through FLANN only (PCL's own KD-tree is a FLANN wrapper), so
// enabling PCL would duplicate the FLANN table. This file exists so the CMake target stays stable and
// documents why no PCL table is produced.

#include <ostream>

namespace ffcl::benchmark {

void benchmark_pcl(std::ostream& results) {
    (void)results;  // reported as unavailable in benchmark/README.md
}

}  // namespace ffcl::benchmark
