#include "energy_estimator.h"

namespace zcci {

namespace {

uint64_t saturating_add(uint64_t lhs, uint64_t rhs) {
  const uint64_t result = lhs + rhs;
  return result < lhs ? UINT64_MAX : result;
}

uint32_t saturating_increment(uint32_t value) {
  return value == UINT32_MAX ? UINT32_MAX : value + 1;
}

uint64_t energy_uj(uint32_t supply_mv, uint32_t current_ma, uint64_t duration_us) {
  return (static_cast<uint64_t>(supply_mv) * static_cast<uint64_t>(current_ma) * duration_us) /
         1000000ULL;
}

uint64_t sample_busy_us(const EnergyFrameSample &sample) {
  uint64_t busy = sample.capture_us;
  busy = saturating_add(busy, sample.gate_us);
  busy = saturating_add(busy, sample.inference_us);
  busy = saturating_add(busy, sample.output_us);
  return busy;
}

void start_window(EnergyEstimatorState *state, uint64_t timestamp_us) {
  state->active = true;
  state->window_start_us = timestamp_us;
  state->accum = {};
  state->accum.window_index = state->next_window_index;
}

void accumulate_sample(EnergyEstimatorState *state, const EnergyFrameSample &sample) {
  EnergyWindowReport *accum = &state->accum;
  accum->frames = saturating_increment(accum->frames);
  if (sample.reference_ready) {
    accum->reference_ready_frames = saturating_increment(accum->reference_ready_frames);
  }
  if (sample.inference_ran) {
    accum->inference_frames = saturating_increment(accum->inference_frames);
  } else if (sample.reference_ready) {
    accum->skipped_frames = saturating_increment(accum->skipped_frames);
  }
  if (sample.comparison_forced) {
    accum->forced_frames = saturating_increment(accum->forced_frames);
  }

  accum->capture_us = saturating_add(accum->capture_us, sample.capture_us);
  accum->gate_us = saturating_add(accum->gate_us, sample.gate_us);
  accum->inference_us = saturating_add(accum->inference_us, sample.inference_us);
  accum->output_us = saturating_add(accum->output_us, sample.output_us);
  accum->busy_us = saturating_add(accum->busy_us, sample_busy_us(sample));
}

void finish_report(EnergyWindowReport *report,
                   uint64_t elapsed_us,
                   const BoardEnergyProfile &profile) {
  report->elapsed_us = elapsed_us;
  report->idle_us = report->busy_us < elapsed_us ? elapsed_us - report->busy_us : 0;
  report->avoided_inference_us =
      static_cast<uint64_t>(report->skipped_frames) * profile.reference_inference_us;
  report->net_compute_saved_us =
      report->avoided_inference_us > report->gate_us ? report->avoided_inference_us - report->gate_us
                                                     : 0;
  report->estimated_energy_uj = estimate_energy_uj(*report, profile);
}

}  // namespace

void reset_energy_estimator(EnergyEstimatorState *state) {
  if (state == nullptr) {
    return;
  }
  *state = {};
}

uint64_t estimate_energy_uj(const EnergyWindowReport &report, const BoardEnergyProfile &profile) {
  uint64_t total = 0;
  total = saturating_add(total, energy_uj(profile.supply_mv, profile.cpu_active_ma, report.busy_us));
  total = saturating_add(total, energy_uj(profile.supply_mv, profile.cpu_idle_ma, report.idle_us));
  total =
      saturating_add(total, energy_uj(profile.supply_mv, profile.camera_active_ma, report.elapsed_us));
  total = saturating_add(
      total, energy_uj(profile.supply_mv, profile.psram_active_ma, report.inference_us));
  total = saturating_add(
      total, energy_uj(profile.supply_mv, profile.board_baseline_ma, report.elapsed_us));
  return total;
}

bool record_energy_frame(EnergyEstimatorState *state,
                         const EnergyEstimatorConfig &config,
                         const BoardEnergyProfile &profile,
                         const EnergyFrameSample &sample,
                         EnergyWindowReport *report) {
  if (state == nullptr || config.window_us == 0) {
    return false;
  }

  if (!state->active) {
    start_window(state, sample.timestamp_us);
  }

  accumulate_sample(state, sample);

  const uint64_t elapsed_us =
      sample.timestamp_us >= state->window_start_us ? sample.timestamp_us - state->window_start_us : 0;
  if (elapsed_us < config.window_us) {
    return false;
  }

  finish_report(&state->accum, elapsed_us, profile);
  if (report != nullptr) {
    *report = state->accum;
  }

  state->next_window_index = saturating_increment(state->next_window_index);
  start_window(state, sample.timestamp_us);
  return true;
}

}  // namespace zcci
