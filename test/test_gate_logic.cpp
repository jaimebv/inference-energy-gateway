#include "gate.h"

#include <array>
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

std::array<uint8_t, zcci::kGateCellCount> filled_frame(uint8_t value) {
  std::array<uint8_t, zcci::kGateCellCount> pixels = {};
  pixels.fill(value);
  return pixels;
}

zcci::GrayscaleFrameView view(const std::array<uint8_t, zcci::kGateCellCount> &pixels) {
  return {
      .pixels = pixels.data(),
      .width = zcci::kGateGridWidth,
      .height = zcci::kGateGridHeight,
  };
}

void test_reduction_preserves_one_sample_cells() {
  std::array<uint8_t, zcci::kGateCellCount> pixels = {};
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = static_cast<uint8_t>(i & 0xff);
  }

  uint8_t grid[zcci::kGateCellCount] = {};
  zcci::reduce_frame_to_grid(view(pixels), grid);

  for (size_t i = 0; i < pixels.size(); ++i) {
    require(grid[i] == pixels[i], "16x12 frame should map one source pixel to one gate cell");
  }
  pass("reduction_preserves_one_sample_cells");
}

void test_first_frame_initializes_reference_without_triggering() {
  zcci::GateState state;
  const zcci::GateConfig config;
  const auto pixels = filled_frame(23);

  const zcci::GateResult result = zcci::evaluate_gate_frame(&state, view(pixels), config);

  require(result.decision == zcci::GateDecision::kReferenceInitialized,
          "first frame should initialize reference");
  require(result.reference_ready, "reference should be ready after first frame");
  require(!result.changed, "first frame should leave gate stable");
  require(state.has_reference, "state should own the reduced reference grid");
  require(state.reference[0] == 23, "reference should store reduced pixel values");
  pass("first_frame_initializes_reference_without_triggering");
}

void test_trigger_hysteresis_requires_consecutive_changed_frames() {
  zcci::GateState state;
  const zcci::GateConfig config;
  const auto reference = filled_frame(0);
  const auto changed = filled_frame(20);

  zcci::evaluate_gate_frame(&state, view(reference), config);
  zcci::GateResult result = zcci::evaluate_gate_frame(&state, view(changed), config);
  require(result.score == 20, "score should be mean absolute grid difference");
  require(result.decision == zcci::GateDecision::kSkipExpensivePath,
          "first changed frame should not pass two-frame trigger hysteresis");
  require(!state.changed, "gate should remain stable after one changed frame");

  result = zcci::evaluate_gate_frame(&state, view(changed), config);
  require(result.decision == zcci::GateDecision::kRunExpensivePath,
          "second consecutive changed frame should open gate");
  require(state.changed, "gate state should latch changed");
  pass("trigger_hysteresis_requires_consecutive_changed_frames");
}

void test_clear_hysteresis_requires_consecutive_stable_frames() {
  zcci::GateState state;
  const zcci::GateConfig config;
  const auto reference = filled_frame(0);
  const auto changed = filled_frame(20);
  const auto stable = filled_frame(0);

  zcci::evaluate_gate_frame(&state, view(reference), config);
  zcci::evaluate_gate_frame(&state, view(changed), config);
  zcci::evaluate_gate_frame(&state, view(changed), config);
  require(state.changed, "gate should be open before clear-hysteresis test");

  zcci::GateResult result = zcci::evaluate_gate_frame(&state, view(stable), config);
  require(result.decision == zcci::GateDecision::kRunExpensivePath,
          "first stable frame should not immediately close an open gate");
  result = zcci::evaluate_gate_frame(&state, view(stable), config);
  require(result.decision == zcci::GateDecision::kRunExpensivePath,
          "second stable frame should not immediately close an open gate");
  result = zcci::evaluate_gate_frame(&state, view(stable), config);
  require(result.decision == zcci::GateDecision::kSkipExpensivePath,
          "third stable frame should close the gate");
  require(!state.changed, "gate state should return to stable");
  pass("clear_hysteresis_requires_consecutive_stable_frames");
}

void test_invalid_frame_is_rejected_without_state_mutation() {
  zcci::GateState state;
  const zcci::GateConfig config;

  const zcci::GateResult result = zcci::evaluate_gate_frame(&state, {}, config);

  require(result.decision == zcci::GateDecision::kInvalidFrame,
          "invalid frame should return invalid decision");
  require(!result.reference_ready, "invalid frame should not initialize reference");
  require(!state.has_reference, "invalid frame should not mutate reference state");
  pass("invalid_frame_is_rejected_without_state_mutation");
}

void test_stable_scene_refreshes_reference_after_policy_window() {
  zcci::GateState state;
  zcci::GateConfig config;
  config.stable_reference_update_frames = 2;
  const auto reference = filled_frame(10);
  const auto drifted = filled_frame(12);

  zcci::evaluate_gate_frame(&state, view(reference), config);
  zcci::GateResult result = zcci::evaluate_gate_frame(&state, view(drifted), config);
  require(result.decision == zcci::GateDecision::kSkipExpensivePath,
          "stable drift should keep the expensive path skipped");
  require(!result.reference_updated, "first stable drift frame should not refresh reference");
  require(state.stable_reference_streak == 1, "stable policy should count low-score frames");
  require(state.reference[0] == 10, "reference should remain unchanged before policy window");

  result = zcci::evaluate_gate_frame(&state, view(drifted), config);
  require(result.decision == zcci::GateDecision::kSkipExpensivePath,
          "stable refresh frame should still skip expensive path");
  require(result.reference_updated, "stable policy window should refresh reference");
  require(result.reference_update_reason == zcci::ReferenceUpdateReason::kStableSceneRefresh,
          "stable refresh should report its reference update reason");
  require(state.reference[0] == 12, "stable refresh should rebase to the current grid");
  require(state.reference_update_count == 2, "first frame plus stable refresh should be counted");
  require(state.frames_since_reference_update == 0, "reference age should reset after refresh");
  pass("stable_scene_refreshes_reference_after_policy_window");
}

void test_changed_scene_rebases_reference_after_policy_window() {
  zcci::GateState state;
  zcci::GateConfig config;
  config.trigger_consecutive_frames = 1;
  config.changed_reference_reset_frames = 3;
  const auto reference = filled_frame(0);
  const auto changed = filled_frame(40);

  zcci::evaluate_gate_frame(&state, view(reference), config);
  zcci::GateResult result = zcci::evaluate_gate_frame(&state, view(changed), config);
  require(result.decision == zcci::GateDecision::kRunExpensivePath,
          "changed scene should open the gate before rebase window is reached");
  require(!result.reference_updated, "first changed frame should not immediately rebase");
  require(state.changed_reference_streak == 1, "changed policy should count open-gate frames");

  zcci::evaluate_gate_frame(&state, view(changed), config);
  result = zcci::evaluate_gate_frame(&state, view(changed), config);
  require(result.decision == zcci::GateDecision::kSkipExpensivePath,
          "persistent changed scene should become the new stable reference");
  require(result.reference_updated, "changed policy window should rebase reference");
  require(result.reference_update_reason == zcci::ReferenceUpdateReason::kChangedSceneRebase,
          "changed rebase should report its reference update reason");
  require(!state.changed, "gate should close after accepting the new scene as reference");
  require(state.reference[0] == 40, "changed rebase should store the new scene");
  require(state.changed_reference_streak == 0, "changed streak should reset after rebase");
  require(state.reference_update_count == 2, "first frame plus changed rebase should be counted");
  pass("changed_scene_rebases_reference_after_policy_window");
}

}  // namespace

int main() {
  test_reduction_preserves_one_sample_cells();
  test_first_frame_initializes_reference_without_triggering();
  test_trigger_hysteresis_requires_consecutive_changed_frames();
  test_clear_hysteresis_requires_consecutive_stable_frames();
  test_invalid_frame_is_rejected_without_state_mutation();
  test_stable_scene_refreshes_reference_after_policy_window();
  test_changed_scene_rebases_reference_after_policy_window();
  return 0;
}
