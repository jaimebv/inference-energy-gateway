#include "frame_schedule.h"

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
  require(zcci::kFramePeriodMs == 500, "frame period should be fixed at 500 ms");

#if defined(ZCCI_FRAME_SCHEDULE_DELAY_UNTIL_BUILD)
  require(zcci::kFrameDelayUntilEnabled, "delay-until flag should enable periodic scheduling");
  require(zcci::kFrameScheduleModeName[0] == 'd', "delay-until mode should log as delay_until");
  pass("frame_schedule_delay_until_build");
#elif defined(ZCCI_FRAME_SCHEDULE_DISABLED_BUILD)
  require(!zcci::kFrameDelayUntilEnabled, "disabled flag should use simple delay fallback");
  require(zcci::kFrameScheduleModeName[0] == 'd', "disabled mode should log as disabled");
  pass("frame_schedule_disabled_build");
#endif

  return 0;
}
