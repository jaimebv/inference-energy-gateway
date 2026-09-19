#include "energy_estimator.h"

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

void test_window_accumulates_busy_idle_and_skip_metrics() {
  zcci::EnergyEstimatorState state;
  const zcci::EnergyEstimatorConfig config = {
      .window_us = 10000,
  };
  const zcci::BoardEnergyProfile profile = {
      .supply_mv = 1000,
      .cpu_active_ma = 10,
      .cpu_idle_ma = 1,
      .camera_active_ma = 2,
      .psram_active_ma = 3,
      .board_baseline_ma = 4,
      .reference_inference_us = 1000,
  };
  zcci::EnergyWindowReport report;

  bool ready = zcci::record_energy_frame(
      &state,
      config,
      profile,
      {
          .timestamp_us = 0,
          .capture_us = 100,
          .gate_us = 100,
          .inference_us = 0,
          .output_us = 100,
          .reference_ready = false,
          .inference_ran = false,
          .comparison_forced = false,
      },
      &report);
  require(!ready, "first sample should start the window without reporting");

  ready = zcci::record_energy_frame(
      &state,
      config,
      profile,
      {
          .timestamp_us = 5000,
          .capture_us = 100,
          .gate_us = 100,
          .inference_us = 1000,
          .output_us = 100,
          .reference_ready = true,
          .inference_ran = true,
          .comparison_forced = true,
      },
      &report);
  require(!ready, "middle sample should accumulate without reporting");

  ready = zcci::record_energy_frame(
      &state,
      config,
      profile,
      {
          .timestamp_us = 10000,
          .capture_us = 100,
          .gate_us = 100,
          .inference_us = 0,
          .output_us = 100,
          .reference_ready = true,
          .inference_ran = false,
          .comparison_forced = false,
      },
      &report);

  require(ready, "sample at the window boundary should produce a report");
  require(report.window_index == 1, "first report should use window index one");
  require(report.elapsed_us == 10000, "elapsed time should match the configured window");
  require(report.frames == 3, "all samples should be counted");
  require(report.reference_ready_frames == 2, "reference-ready samples should be counted");
  require(report.skipped_frames == 1, "one reference-ready skipped frame should be counted");
  require(report.inference_frames == 1, "one inference frame should be counted");
  require(report.forced_frames == 1, "one forced comparison frame should be counted");
  require(report.capture_us == 300, "capture time should accumulate");
  require(report.gate_us == 300, "gate time should accumulate");
  require(report.inference_us == 1000, "inference time should accumulate");
  require(report.output_us == 300, "output time should accumulate");
  require(report.busy_us == 1900, "busy time should include capture, gate, inference, and output");
  require(report.idle_us == 8100, "idle estimate should be elapsed minus busy");
  require(report.avoided_inference_us == 1000, "skipped frame should estimate avoided inference");
  require(report.net_compute_saved_us == 700, "net saved compute should subtract gate cost");
  require(report.estimated_energy_uj == 90, "energy estimate should use board profile constants");
  pass("window_accumulates_busy_idle_and_skip_metrics");
}

void test_reset_clears_estimator_state() {
  zcci::EnergyEstimatorState state;
  zcci::EnergyWindowReport report;
  const zcci::EnergyEstimatorConfig config = {
      .window_us = 1,
  };
  const zcci::BoardEnergyProfile profile;

  zcci::record_energy_frame(&state, config, profile, {.timestamp_us = 5}, &report);
  require(state.active, "recording should activate the estimator");

  zcci::reset_energy_estimator(&state);
  require(!state.active, "reset should clear the active flag");
  require(state.next_window_index == 1, "reset should restore the first window index");
  require(state.accum.frames == 0, "reset should clear accumulated frames");
  pass("reset_clears_estimator_state");
}

}  // namespace

int main() {
  test_window_accumulates_busy_idle_and_skip_metrics();
  test_reset_clears_estimator_state();
  return 0;
}
