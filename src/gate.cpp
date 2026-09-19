#include "gate.h"

#include <cstring>

namespace zcci {
namespace {

uint8_t saturating_increment(uint8_t value) {
  return value == UINT8_MAX ? value : static_cast<uint8_t>(value + 1);
}

uint16_t saturating_increment(uint16_t value) {
  return value == UINT16_MAX ? value : static_cast<uint16_t>(value + 1);
}

void copy_reference(GateState *state, const uint8_t *grid) {
  std::memcpy(state->reference, grid, sizeof(state->reference));
  state->frames_since_reference_update = 0;
  state->stable_reference_streak = 0;
  state->changed_reference_streak = 0;
  state->trigger_streak = 0;
  state->clear_streak = 0;
  state->changed = false;
  state->has_reference = true;
  state->reference_update_count = saturating_increment(state->reference_update_count);
}

}  // namespace

bool is_valid_frame(GrayscaleFrameView frame) {
  return frame.pixels != nullptr && frame.width > 0 && frame.height > 0;
}

void reset_gate_state(GateState *state) {
  if (state == nullptr) {
    return;
  }

  *state = GateState{};
}

void reduce_frame_to_grid(GrayscaleFrameView frame, uint8_t *grid) {
  if (!is_valid_frame(frame) || grid == nullptr) {
    return;
  }

  for (size_t cell_y = 0; cell_y < kGateGridHeight; ++cell_y) {
    for (size_t cell_x = 0; cell_x < kGateGridWidth; ++cell_x) {
      uint32_t sum = 0;
      uint32_t samples = 0;
      const size_t row_start = (cell_y * frame.height) / kGateGridHeight;
      const size_t row_end = ((cell_y + 1) * frame.height) / kGateGridHeight;
      const size_t column_start = (cell_x * frame.width) / kGateGridWidth;
      const size_t column_end = ((cell_x + 1) * frame.width) / kGateGridWidth;

      for (size_t y = row_start; y < row_end; ++y) {
        const uint8_t *row = frame.pixels + (y * frame.width);
        for (size_t x = column_start; x < column_end; ++x) {
          sum += row[x];
          ++samples;
        }
      }

      grid[(cell_y * kGateGridWidth) + cell_x] =
          static_cast<uint8_t>(samples > 0 ? (sum / samples) : 0);
    }
  }
}

uint8_t compute_change_score(const uint8_t *current_grid, const uint8_t *reference_grid) {
  if (current_grid == nullptr || reference_grid == nullptr) {
    return 0;
  }

  uint32_t total_difference = 0;
  for (size_t i = 0; i < kGateCellCount; ++i) {
    const int diff = static_cast<int>(current_grid[i]) - static_cast<int>(reference_grid[i]);
    total_difference += static_cast<uint32_t>(diff < 0 ? -diff : diff);
  }

  return static_cast<uint8_t>(total_difference / kGateCellCount);
}

bool update_gate_state(GateState *state, uint8_t score, const GateConfig &config) {
  if (state == nullptr) {
    return false;
  }

  if (score >= config.trigger_threshold) {
    state->trigger_streak = saturating_increment(state->trigger_streak);
    state->clear_streak = 0;
  } else if (score <= config.clear_threshold) {
    state->clear_streak = saturating_increment(state->clear_streak);
    state->trigger_streak = 0;
  } else {
    state->trigger_streak = 0;
    state->clear_streak = 0;
  }

  if (!state->changed && state->trigger_streak >= config.trigger_consecutive_frames) {
    state->changed = true;
  } else if (state->changed && state->clear_streak >= config.clear_consecutive_frames) {
    state->changed = false;
  }

  return state->changed;
}

bool maybe_update_reference(GateState *state,
                            const uint8_t *current_grid,
                            uint8_t score,
                            const GateConfig &config,
                            ReferenceUpdateReason *reason) {
  if (reason != nullptr) {
    *reason = ReferenceUpdateReason::kNone;
  }
  if (state == nullptr || current_grid == nullptr || !state->has_reference) {
    return false;
  }

  state->frames_since_reference_update = saturating_increment(state->frames_since_reference_update);

  if (state->changed) {
    state->stable_reference_streak = 0;
    if (score >= config.trigger_threshold) {
      state->changed_reference_streak = saturating_increment(state->changed_reference_streak);
    } else {
      state->changed_reference_streak = 0;
    }

    if (config.changed_reference_reset_frames > 0 &&
        state->changed_reference_streak >= config.changed_reference_reset_frames) {
      copy_reference(state, current_grid);
      if (reason != nullptr) {
        *reason = ReferenceUpdateReason::kChangedSceneRebase;
      }
      return true;
    }
    return false;
  }

  state->changed_reference_streak = 0;
  if (score <= config.clear_threshold) {
    state->stable_reference_streak = saturating_increment(state->stable_reference_streak);
  } else {
    state->stable_reference_streak = 0;
  }

  if (config.stable_reference_update_frames > 0 &&
      state->stable_reference_streak >= config.stable_reference_update_frames) {
    copy_reference(state, current_grid);
    if (reason != nullptr) {
      *reason = ReferenceUpdateReason::kStableSceneRefresh;
    }
    return true;
  }

  return false;
}

GateResult evaluate_gate_frame(GateState *state, GrayscaleFrameView frame, const GateConfig &config) {
  GateResult result;
  if (state == nullptr || !is_valid_frame(frame)) {
    return result;
  }

  uint8_t current_grid[kGateCellCount] = {};
  reduce_frame_to_grid(frame, current_grid);

  if (!state->has_reference) {
    copy_reference(state, current_grid);

    result.decision = GateDecision::kReferenceInitialized;
    result.reference_ready = true;
    result.reference_updated = true;
    result.reference_update_reason = ReferenceUpdateReason::kFirstFrame;
    return result;
  }

  result.score = compute_change_score(current_grid, state->reference);
  result.changed = update_gate_state(state, result.score, config);
  result.reference_updated = maybe_update_reference(
      state, current_grid, result.score, config, &result.reference_update_reason);
  result.changed = state->changed;
  result.reference_ready = true;
  result.decision = result.changed ? GateDecision::kRunExpensivePath : GateDecision::kSkipExpensivePath;
  return result;
}

}  // namespace zcci
