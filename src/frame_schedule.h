#pragma once

#include <cstdint>

namespace zcci {

#if (defined(ZCCI_FRAME_SCHEDULE_DELAY_UNTIL_BUILD) + defined(ZCCI_FRAME_SCHEDULE_DISABLED_BUILD)) != 1
#error "Define exactly one frame schedule mode: ZCCI_FRAME_SCHEDULE_DELAY_UNTIL_BUILD or ZCCI_FRAME_SCHEDULE_DISABLED_BUILD"
#endif

constexpr uint32_t kFramePeriodMs = 500;

#if defined(ZCCI_FRAME_SCHEDULE_DELAY_UNTIL_BUILD)
constexpr bool kFrameDelayUntilEnabled = true;
constexpr const char *kFrameScheduleBuildFlag = "ZCCI_FRAME_SCHEDULE_DELAY_UNTIL_BUILD";
constexpr const char *kFrameScheduleModeName = "delay_until";
#elif defined(ZCCI_FRAME_SCHEDULE_DISABLED_BUILD)
constexpr bool kFrameDelayUntilEnabled = false;
constexpr const char *kFrameScheduleBuildFlag = "ZCCI_FRAME_SCHEDULE_DISABLED_BUILD";
constexpr const char *kFrameScheduleModeName = "disabled";
#endif

}  // namespace zcci
