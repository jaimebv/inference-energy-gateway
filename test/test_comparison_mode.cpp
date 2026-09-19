#include "comparison_mode.h"

#include <cstdio>
#include <cstdlib>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::printf("TEST FAILED: %s\n", message);
    std::exit(1);
  }
}

void pass(const char *name) {
  std::printf("TEST PASSED: %s\n", name);
}

}  // namespace

int main() {
#if defined(ZCCI_COMPARISON_MODE_GATED_BUILD)
  require(zcci::kComparisonMode == zcci::ComparisonMode::kGated,
          "gated build flag should select gated mode");
  require(!zcci::comparison_mode_forces_inference(zcci::kComparisonMode),
          "gated mode should leave inference controlled by the gate");
  require(zcci::comparison_mode_to_string(zcci::kComparisonMode)[0] == 'g',
          "gated mode should log as gated");
  pass("comparison_mode_gated_build");
#elif defined(ZCCI_COMPARISON_MODE_ALWAYS_ON_BUILD)
  require(zcci::kComparisonMode == zcci::ComparisonMode::kAlwaysOn,
          "always-on build flag should select always-on mode");
  require(zcci::comparison_mode_forces_inference(zcci::kComparisonMode),
          "always-on mode should force inference after the reference exists");
  require(zcci::comparison_mode_to_string(zcci::kComparisonMode)[0] == 'a',
          "always-on mode should log as always_on");
  pass("comparison_mode_always_on_build");
#endif
  return 0;
}
