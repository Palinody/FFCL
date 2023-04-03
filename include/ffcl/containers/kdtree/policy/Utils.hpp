#pragma once

namespace kdtree::policy {

template <typename... Args>
constexpr void ignore_parameters(Args&&...) noexcept {}

}  // namespace kdtree::policy