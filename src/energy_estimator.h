#pragma once

#include <cstdint>

namespace zcci {

constexpr uint64_t kDefaultEnergyWindowUs = 60ULL * 1000ULL * 1000ULL;

struct BoardEnergyProfile {
  uint32_t supply_mv = 3300;
  uint32_t cpu_active_ma = 80;
  uint32_t cpu_idle_ma = 20;
  uint32_t camera_active_ma = 20;
  uint32_t psram_active_ma = 5;
  uint32_t board_baseline_ma = 10;
  uint32_t reference_inference_us = 379260;
};

struct EnergyEstimatorConfig {
  uint64_t window_us = kDefaultEnergyWindowUs;
};

struct EnergyFrameSample {
  uint64_t timestamp_us = 0;
  uint64_t capture_us = 0;
  uint64_t gate_us = 0;
  uint64_t inference_us = 0;
  uint64_t output_us = 0;
  bool reference_ready = false;
  bool inference_ran = false;
  bool comparison_forced = false;
};

struct EnergyWindowReport {
  uint32_t window_index = 0;
  uint64_t elapsed_us = 0;
  uint32_t frames = 0;
  uint32_t reference_ready_frames = 0;
  uint32_t skipped_frames = 0;
  uint32_t inference_frames = 0;
  uint32_t forced_frames = 0;
  uint64_t capture_us = 0;
  uint64_t gate_us = 0;
  uint64_t inference_us = 0;
  uint64_t output_us = 0;
  uint64_t busy_us = 0;
  uint64_t idle_us = 0;
  uint64_t avoided_inference_us = 0;
  uint64_t net_compute_saved_us = 0;
  uint64_t estimated_energy_uj = 0;
};

struct EnergyEstimatorState {
  bool active = false;
  uint32_t next_window_index = 1;
  uint64_t window_start_us = 0;
  EnergyWindowReport accum;
};

void reset_energy_estimator(EnergyEstimatorState *state);
uint64_t estimate_energy_uj(const EnergyWindowReport &report, const BoardEnergyProfile &profile);
bool record_energy_frame(EnergyEstimatorState *state,
                         const EnergyEstimatorConfig &config,
                         const BoardEnergyProfile &profile,
                         const EnergyFrameSample &sample,
                         EnergyWindowReport *report);

}  // namespace zcci
