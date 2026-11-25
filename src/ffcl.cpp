// Single translation unit of the shared library target.
//
// FFCL is header-only: the templates are instantiated in the client translation units. This file exists so
// that the `SHARED` target has a linker language and produces a real library file, and it gives the library a
// version symbol to link against.
#include "ffcl/common/Utils.hpp"

namespace ffcl::common {

auto version() -> const char* {
    return "ffcl";
}

}  // namespace ffcl::common
