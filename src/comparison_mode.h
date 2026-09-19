#pragma once

#include <cstdint>

namespace zcci {

enum class ComparisonMode : uint8_t {
  kGated,
  kAlwaysOn,
};

#if (defined(ZCCI_COMPARISON_MODE_GATED_BUILD) + defined(ZCCI_COMPARISON_MODE_ALWAYS_ON_BUILD)) != 1
#error "Define exactly one comparison mode: ZCCI_COMPARISON_MODE_GATED_BUILD or ZCCI_COMPARISON_MODE_ALWAYS_ON_BUILD"
#endif

#if defined(ZCCI_COMPARISON_MODE_GATED_BUILD)
constexpr ComparisonMode kComparisonMode = ComparisonMode::kGated;
constexpr const char *kComparisonModeBuildFlag = "ZCCI_COMPARISON_MODE_GATED_BUILD";
#elif defined(ZCCI_COMPARISON_MODE_ALWAYS_ON_BUILD)
constexpr ComparisonMode kComparisonMode = ComparisonMode::kAlwaysOn;
constexpr const char *kComparisonModeBuildFlag = "ZCCI_COMPARISON_MODE_ALWAYS_ON_BUILD";
#endif

constexpr const char *comparison_mode_to_string(ComparisonMode mode) {
  return mode == ComparisonMode::kAlwaysOn ? "always_on" : "gated";
}

constexpr bool comparison_mode_forces_inference(ComparisonMode mode) {
  return mode == ComparisonMode::kAlwaysOn;
}

}  // namespace zcci
