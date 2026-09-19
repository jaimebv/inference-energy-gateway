#pragma once

#include <cstddef>
#include <cstdint>

namespace zcci {

constexpr size_t kGateGridWidth = 16;
constexpr size_t kGateGridHeight = 12;
constexpr size_t kGateCellCount = kGateGridWidth * kGateGridHeight;

struct GrayscaleFrameView {
  const uint8_t *pixels = nullptr;
  size_t width = 0;
  size_t height = 0;
};

struct GateConfig {
  uint8_t trigger_threshold = 12;
  uint8_t clear_threshold = 8;
  uint8_t trigger_consecutive_frames = 2;
  uint8_t clear_consecutive_frames = 3;
  uint16_t stable_reference_update_frames = 30;
  uint16_t changed_reference_reset_frames = 6;
};

struct GateState {
  bool has_reference = false;
  bool changed = false;
  uint8_t trigger_streak = 0;
  uint8_t clear_streak = 0;
  uint16_t stable_reference_streak = 0;
  uint16_t changed_reference_streak = 0;
  uint16_t frames_since_reference_update = 0;
  uint16_t reference_update_count = 0;
  uint8_t reference[kGateCellCount] = {};
};

enum class ReferenceUpdateReason {
  kNone,
  kFirstFrame,
  kStableSceneRefresh,
  kChangedSceneRebase,
};

enum class GateDecision {
  kInvalidFrame,
  kReferenceInitialized,
  kSkipExpensivePath,
  kRunExpensivePath,
};

struct GateResult {
  GateDecision decision = GateDecision::kInvalidFrame;
  uint8_t score = 0;
  bool reference_ready = false;
  bool changed = false;
  bool reference_updated = false;
  ReferenceUpdateReason reference_update_reason = ReferenceUpdateReason::kNone;
};

bool is_valid_frame(GrayscaleFrameView frame);
void reset_gate_state(GateState *state);
void reduce_frame_to_grid(GrayscaleFrameView frame, uint8_t *grid);
uint8_t compute_change_score(const uint8_t *current_grid, const uint8_t *reference_grid);
bool update_gate_state(GateState *state, uint8_t score, const GateConfig &config);
bool maybe_update_reference(GateState *state,
                            const uint8_t *current_grid,
                            uint8_t score,
                            const GateConfig &config,
                            ReferenceUpdateReason *reason);
GateResult evaluate_gate_frame(GateState *state, GrayscaleFrameView frame, const GateConfig &config);

}  // namespace zcci
