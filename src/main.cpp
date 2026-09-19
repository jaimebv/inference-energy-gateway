#include <cstdio>
#include <new>

#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "comparison_mode.h"
#include "energy_estimator.h"
#include "frame_schedule.h"
#include "gate.h"
#include "person_detect_model_data.h"
#include "sensor.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

// XIAO ESP32S3 Sense-style OV2640 wiring.
constexpr int kCameraPinPwdn = -1;
constexpr int kCameraPinReset = -1;
constexpr int kCameraPinXclk = 10;
constexpr int kCameraPinSccbSda = 40;
constexpr int kCameraPinSccbScl = 39;
constexpr int kCameraPinY9 = 48;
constexpr int kCameraPinY8 = 11;
constexpr int kCameraPinY7 = 12;
constexpr int kCameraPinY6 = 14;
constexpr int kCameraPinY5 = 16;
constexpr int kCameraPinY4 = 18;
constexpr int kCameraPinY3 = 17;
constexpr int kCameraPinY2 = 15;
constexpr int kCameraPinVsync = 38;
constexpr int kCameraPinHref = 47;
constexpr int kCameraPinPclk = 13;

// The gate uses the approved design note thresholds and hysteresis.
constexpr zcci::GateConfig kGateConfig = {
    .trigger_threshold = 12,
    .clear_threshold = 8,
    .trigger_consecutive_frames = 2,
    .clear_consecutive_frames = 3,
    .stable_reference_update_frames = 30,
    .changed_reference_reset_frames = 6,
};

constexpr framesize_t kCaptureFramesize = FRAMESIZE_QQVGA;
constexpr pixformat_t kCapturePixelFormat = PIXFORMAT_GRAYSCALE;
constexpr camera_fb_location_t kCaptureFramebufferLocation = CAMERA_FB_IN_DRAM;
constexpr camera_grab_mode_t kCaptureGrabMode = CAMERA_GRAB_WHEN_EMPTY;
constexpr size_t kCaptureFramebufferCount = 1;
constexpr uint32_t kFirstSnapshotDelayMs = 15000;
constexpr size_t kModelInputWidth = 96;
constexpr size_t kModelInputHeight = 96;
constexpr size_t kModelInputChannels = 1;
constexpr size_t kModelInputCellCount = kModelInputWidth * kModelInputHeight * kModelInputChannels;
constexpr int kPersonIndex = 1;
constexpr int kNotPersonIndex = 0;
constexpr size_t kTensorArenaSize = 160 * 1024;
constexpr zcci::EnergyEstimatorConfig kEnergyEstimatorConfig = {
    .window_us = zcci::kDefaultEnergyWindowUs,
};
constexpr zcci::BoardEnergyProfile kBoardEnergyProfile = {
    .supply_mv = 3300,
    .cpu_active_ma = 80,
    .cpu_idle_ma = 20,
    .camera_active_ma = 20,
    .psram_active_ma = 5,
    .board_baseline_ma = 10,
    .reference_inference_us = 379260,
};

struct MemorySnapshot {
  size_t free_heap = 0;
  size_t largest_heap_block = 0;
  size_t free_psram = 0;
  size_t largest_psram_block = 0;
};

struct FrameTelemetry {
  int64_t capture_us = 0;
  int64_t gate_us = 0;
  int64_t inference_us = 0;
  int64_t output_us = 0;
  int64_t total_us = 0;
  uint8_t score = 0;
  bool reference_ready = false;
  bool inference_ran = false;
  bool gate_open = false;
  bool comparison_forced = false;
  bool reference_updated = false;
  zcci::ReferenceUpdateReason reference_update_reason = zcci::ReferenceUpdateReason::kNone;
  size_t peak_heap_used = 0;
  size_t peak_psram_used = 0;
  size_t current_free_heap = 0;
  size_t current_free_psram = 0;
  size_t current_largest_heap_block = 0;
  size_t current_largest_psram_block = 0;
};

zcci::GateState g_gate_state;
zcci::EnergyEstimatorState g_energy_estimator;
const tflite::Model *g_model = nullptr;
tflite::MicroMutableOpResolver<5> g_op_resolver;
alignas(tflite::MicroInterpreter) uint8_t g_interpreter_storage[sizeof(tflite::MicroInterpreter)];
tflite::MicroInterpreter *g_interpreter = nullptr;
uint8_t *g_tensor_arena = nullptr;
TfLiteTensor *g_input_tensor = nullptr;
bool g_model_initialized = false;
bool g_ops_initialized = false;
size_t g_boot_free_heap = 0;
size_t g_boot_free_psram = 0;
size_t g_min_free_heap = 0;
size_t g_min_free_psram = 0;
uint32_t g_skipped_inference_runs = 0;
uint32_t g_forced_inference_runs = 0;
bool g_snapshot_emitted = false;

MemorySnapshot read_memory_snapshot() {
  MemorySnapshot snapshot;
  snapshot.free_heap = heap_caps_get_free_size(MALLOC_CAP_8BIT);
  snapshot.largest_heap_block = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
  snapshot.free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  snapshot.largest_psram_block = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
  return snapshot;
}

void update_memory_peaks(const MemorySnapshot &snapshot) {
  if (g_min_free_heap == 0 || snapshot.free_heap < g_min_free_heap) {
    g_min_free_heap = snapshot.free_heap;
  }
  if (g_min_free_psram == 0 || snapshot.free_psram < g_min_free_psram) {
    g_min_free_psram = snapshot.free_psram;
  }
}

size_t peak_used_bytes(size_t boot_free_bytes, size_t min_free_bytes) {
  if (boot_free_bytes < min_free_bytes) {
    return 0;
  }
  return boot_free_bytes - min_free_bytes;
}

uint64_t nonnegative_us(int64_t value) {
  return value > 0 ? static_cast<uint64_t>(value) : 0;
}

uint32_t permille(uint64_t part, uint64_t whole) {
  if (whole == 0) {
    return 0;
  }
  return static_cast<uint32_t>((part * 1000ULL) / whole);
}

uint32_t percent(uint32_t part, uint32_t whole) {
  if (whole == 0) {
    return 0;
  }
  return static_cast<uint32_t>((static_cast<uint64_t>(part) * 100ULL) / whole);
}

void log_memory_headroom(const char *stage) {
  const MemorySnapshot snapshot = read_memory_snapshot();
  update_memory_peaks(snapshot);
  const size_t peak_heap_used = peak_used_bytes(g_boot_free_heap, g_min_free_heap);
  const size_t peak_psram_used = peak_used_bytes(g_boot_free_psram, g_min_free_psram);

  std::printf(
      "[mem] stage=%s free_heap=%u largest_heap_block=%u free_psram=%u largest_psram_block=%u peak_heap_used=%u peak_psram_used=%u\n",
      stage,
      static_cast<unsigned>(snapshot.free_heap),
      static_cast<unsigned>(snapshot.largest_heap_block),
      static_cast<unsigned>(snapshot.free_psram),
      static_cast<unsigned>(snapshot.largest_psram_block),
      static_cast<unsigned>(peak_heap_used),
      static_cast<unsigned>(peak_psram_used));
}

const char *pixformat_to_string(pixformat_t format) {
  switch (format) {
    case PIXFORMAT_RGB565:
      return "RGB565";
    case PIXFORMAT_YUV422:
      return "YUV422";
    case PIXFORMAT_YUV420:
      return "YUV420";
    case PIXFORMAT_GRAYSCALE:
      return "GRAYSCALE";
    case PIXFORMAT_JPEG:
      return "JPEG";
    case PIXFORMAT_RGB888:
      return "RGB888";
    case PIXFORMAT_RAW:
      return "RAW";
    case PIXFORMAT_RGB444:
      return "RGB444";
    default:
      return "UNKNOWN";
  }
}

const char *comparison_mode_to_string() {
  return zcci::comparison_mode_to_string(zcci::kComparisonMode);
}

const char *framebuffer_location_to_string(camera_fb_location_t location) {
  return location == CAMERA_FB_IN_PSRAM ? "PSRAM" : "DRAM";
}

const char *reference_update_reason_to_string(zcci::ReferenceUpdateReason reason) {
  switch (reason) {
    case zcci::ReferenceUpdateReason::kFirstFrame:
      return "first_frame";
    case zcci::ReferenceUpdateReason::kStableSceneRefresh:
      return "stable_scene_refresh";
    case zcci::ReferenceUpdateReason::kChangedSceneRebase:
      return "changed_scene_rebase";
    case zcci::ReferenceUpdateReason::kNone:
    default:
      return "none";
  }
}

void configure_camera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = kCameraPinY2;
  config.pin_d1 = kCameraPinY3;
  config.pin_d2 = kCameraPinY4;
  config.pin_d3 = kCameraPinY5;
  config.pin_d4 = kCameraPinY6;
  config.pin_d5 = kCameraPinY7;
  config.pin_d6 = kCameraPinY8;
  config.pin_d7 = kCameraPinY9;
  config.pin_xclk = kCameraPinXclk;
  config.pin_pclk = kCameraPinPclk;
  config.pin_vsync = kCameraPinVsync;
  config.pin_href = kCameraPinHref;
  config.pin_sccb_sda = kCameraPinSccbSda;
  config.pin_sccb_scl = kCameraPinSccbScl;
  config.pin_pwdn = kCameraPinPwdn;
  config.pin_reset = kCameraPinReset;
  config.xclk_freq_hz = 15000000;
  config.frame_size = kCaptureFramesize;
  config.pixel_format = kCapturePixelFormat;
  config.grab_mode = kCaptureGrabMode;
  config.fb_location = kCaptureFramebufferLocation;
  config.fb_count = kCaptureFramebufferCount;

  log_memory_headroom("before_camera_init");

  const esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    std::printf("[camera] init_failed err=%d\n", static_cast<int>(err));
    while (true) {
      vTaskDelay(pdMS_TO_TICKS(1000));
    }
  }

  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor != nullptr) {
    sensor->set_vflip(sensor, 1);
    if (sensor->id.PID == OV3660_PID) {
      sensor->set_brightness(sensor, 1);
      sensor->set_saturation(sensor, -2);
    }
    std::printf(
        "[camera] sensor pid=0x%04x framesize=%d pixformat=%s fb_location=%s fb_count=%u quality=%d vflip=1\n",
        sensor->id.PID,
        static_cast<int>(sensor->status.framesize),
        pixformat_to_string(kCapturePixelFormat),
        framebuffer_location_to_string(kCaptureFramebufferLocation),
        static_cast<unsigned>(kCaptureFramebufferCount),
        sensor->status.quality);
  }

  log_memory_headroom("after_camera_init");
}

void log_gate_result(uint8_t score, const char *decision, zcci::ReferenceUpdateReason reason) {
  std::printf(
      "[gate] score=%u trigger_streak=%u clear_streak=%u stable_ref_streak=%u "
      "changed_ref_streak=%u frames_since_ref=%u ref_updates=%u state=%s "
      "decision=%s reference_update=%s\n",
      static_cast<unsigned>(score),
      static_cast<unsigned>(g_gate_state.trigger_streak),
      static_cast<unsigned>(g_gate_state.clear_streak),
      static_cast<unsigned>(g_gate_state.stable_reference_streak),
      static_cast<unsigned>(g_gate_state.changed_reference_streak),
      static_cast<unsigned>(g_gate_state.frames_since_reference_update),
      static_cast<unsigned>(g_gate_state.reference_update_count),
      g_gate_state.changed ? "changed" : "stable",
      decision,
      reference_update_reason_to_string(reason));
}

zcci::GrayscaleFrameView frame_view_from_camera_fb(const camera_fb_t *fb) {
  zcci::GrayscaleFrameView frame;
  if (fb != nullptr) {
    frame.pixels = fb->buf;
    frame.width = fb->width;
    frame.height = fb->height;
  }
  return frame;
}

void emit_first_frame_snapshot(const camera_fb_t *fb) {
  if (g_snapshot_emitted) {
    return;
  }

  g_snapshot_emitted = true;
  std::printf(
      "[snapshot] begin encoding=hex format=PGM_P5 width=%u height=%u row_bytes=%u length=%u\n",
      static_cast<unsigned>(fb->width),
      static_cast<unsigned>(fb->height),
      static_cast<unsigned>(fb->width),
      static_cast<unsigned>(fb->len));

  for (size_t y = 0; y < fb->height; ++y) {
    const uint8_t *row = fb->buf + (y * fb->width);
    std::printf("[snapshot] row=%u data=", static_cast<unsigned>(y));
    for (size_t x = 0; x < fb->width; ++x) {
      std::printf("%02x", row[x]);
    }
    std::printf("\n");
  }

  std::printf("[snapshot] end rows=%u\n", static_cast<unsigned>(fb->height));
}

bool allocate_tensor_arena() {
  if (g_tensor_arena != nullptr) {
    return true;
  }

  g_tensor_arena = static_cast<uint8_t *>(
      heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (g_tensor_arena == nullptr) {
    g_tensor_arena = static_cast<uint8_t *>(heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT));
  }

  if (g_tensor_arena == nullptr) {
    std::printf("[inference] tensor_arena_alloc_failed size=%u\n",
                  static_cast<unsigned>(kTensorArenaSize));
    return false;
  }

  std::printf("[inference] tensor_arena_allocated size=%u ptr=%p\n",
                static_cast<unsigned>(kTensorArenaSize),
                g_tensor_arena);
  return true;
}

bool initialize_model_runtime() {
  if (g_model_initialized) {
    return true;
  }

  if (!allocate_tensor_arena()) {
    return false;
  }

  g_model = tflite::GetModel(g_person_detect_model_data);
  if (g_model == nullptr) {
    std::printf("[inference] model_parse_failed\n");
    return false;
  }

  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    std::printf("[inference] schema_mismatch model=%u expected=%d\n",
                  static_cast<unsigned>(g_model->version()),
                  TFLITE_SCHEMA_VERSION);
    return false;
  }

  if (!g_ops_initialized) {
    g_op_resolver.AddConv2D();
    g_op_resolver.AddDepthwiseConv2D();
    g_op_resolver.AddAveragePool2D();
    g_op_resolver.AddReshape();
    g_op_resolver.AddSoftmax();
    g_ops_initialized = true;
  }

  g_interpreter = new (g_interpreter_storage)
      tflite::MicroInterpreter(g_model, g_op_resolver, g_tensor_arena, kTensorArenaSize);
  const TfLiteStatus allocate_status = g_interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    std::printf("[inference] allocate_tensors_failed\n");
    g_interpreter = nullptr;
    return false;
  }

  g_input_tensor = g_interpreter->input(0);
  if (g_input_tensor == nullptr) {
    std::printf("[inference] input_tensor_missing\n");
    g_interpreter = nullptr;
    return false;
  }

  if (g_input_tensor->type != kTfLiteInt8) {
    std::printf("[inference] unexpected_input_type=%d expected=%d\n",
                  static_cast<int>(g_input_tensor->type),
                  static_cast<int>(kTfLiteInt8));
    g_interpreter = nullptr;
    return false;
  }

  std::printf("[inference] model_ready input=%dx%dx%d arena=%u\n",
                static_cast<int>(g_input_tensor->dims->data[2]),
                static_cast<int>(g_input_tensor->dims->data[1]),
                static_cast<int>(g_input_tensor->dims->data[3]),
                static_cast<unsigned>(kTensorArenaSize));
  g_model_initialized = true;
  return true;
}

void populate_model_input_from_frame(const camera_fb_t *fb) {
  const int source_width = static_cast<int>(fb->width);
  const int source_height = static_cast<int>(fb->height);
  int start_x = (source_width - static_cast<int>(kModelInputWidth)) / 2;
  int start_y = (source_height - static_cast<int>(kModelInputHeight)) / 2;
  if (start_x < 0) {
    start_x = 0;
  }
  if (start_y < 0) {
    start_y = 0;
  }

  for (size_t y = 0; y < kModelInputHeight; ++y) {
    int source_y = start_y + static_cast<int>(y);
    if (source_y >= source_height) {
      source_y = source_height - 1;
    }
    const size_t source_row = static_cast<size_t>(source_y) * static_cast<size_t>(source_width);
    const size_t target_row = y * kModelInputWidth;

    for (size_t x = 0; x < kModelInputWidth; ++x) {
      int source_x = start_x + static_cast<int>(x);
      if (source_x >= source_width) {
        source_x = source_width - 1;
      }
      const uint8_t pixel = fb->buf[source_row + static_cast<size_t>(source_x)];
      g_input_tensor->data.int8[target_row + x] = static_cast<int8_t>(pixel ^ 0x80);
    }
  }
}

int read_quantized_value(const TfLiteTensor *tensor, int index) {
  if (tensor->type == kTfLiteInt8) {
    return tensor->data.int8[index];
  }
  if (tensor->type == kTfLiteUInt8) {
    return tensor->data.uint8[index];
  }
  return 0;
}

float dequantize_score(const TfLiteTensor *tensor, int index) {
  return (read_quantized_value(tensor, index) - tensor->params.zero_point) * tensor->params.scale;
}

int probability_percent(float probability) {
  if (probability < 0.0f) {
    probability = 0.0f;
  } else if (probability > 1.0f) {
    probability = 1.0f;
  }
  return static_cast<int>((probability * 100.0f) + 0.5f);
}

void run_expensive_path_model(const camera_fb_t *fb, uint8_t score) {
  if (!initialize_model_runtime()) {
    std::printf("[inference] model_unavailable score=%u\n", static_cast<unsigned>(score));
    return;
  }

  if (g_input_tensor == nullptr || g_interpreter == nullptr) {
    std::printf("[inference] runtime_not_ready score=%u\n", static_cast<unsigned>(score));
    return;
  }

  const int64_t preprocess_start_us = esp_timer_get_time();
  populate_model_input_from_frame(fb);
  const int64_t preprocess_end_us = esp_timer_get_time();

  const int64_t invoke_start_us = esp_timer_get_time();
  const TfLiteStatus invoke_status = g_interpreter->Invoke();
  const int64_t invoke_end_us = esp_timer_get_time();

  if (invoke_status != kTfLiteOk) {
    std::printf("[inference] invoke_failed score=%u preprocess_us=%lld\n",
                  static_cast<unsigned>(score),
                  static_cast<long long>(preprocess_end_us - preprocess_start_us));
    return;
  }

  const TfLiteTensor *output = g_interpreter->output(0);
  if (output == nullptr) {
    std::printf("[inference] output_tensor_missing score=%u\n", static_cast<unsigned>(score));
    return;
  }

  const int person_raw = read_quantized_value(output, kPersonIndex);
  const int no_person_raw = read_quantized_value(output, kNotPersonIndex);
  const float person_score = dequantize_score(output, kPersonIndex);
  const float no_person_score = dequantize_score(output, kNotPersonIndex);

  std::printf(
      "[inference] model_run score=%u preprocess_us=%lld invoke_us=%lld "
      "person_raw=%d no_person_raw=%d person_pct=%d no_person_pct=%d\n",
      static_cast<unsigned>(score),
      static_cast<long long>(preprocess_end_us - preprocess_start_us),
      static_cast<long long>(invoke_end_us - invoke_start_us),
      person_raw,
      no_person_raw,
      probability_percent(person_score),
      probability_percent(no_person_score));
}

void emit_frame_report(FrameTelemetry *telemetry) {
  const int64_t output_start_us = esp_timer_get_time();
  std::printf(
      "[frame] mode=%s capture_us=%lld gate_us=%lld inference_us=%lld total_us=%lld "
      "score=%u reference=%s reference_update=%s gate=%s inference=%s forced=%s "
      "skipped_runs=%u forced_runs=%u "
      "peak_heap_used=%u peak_psram_used=%u\n",
      comparison_mode_to_string(),
      static_cast<long long>(telemetry->capture_us),
      static_cast<long long>(telemetry->gate_us),
      static_cast<long long>(telemetry->inference_us),
      static_cast<long long>(telemetry->total_us),
      static_cast<unsigned>(telemetry->score),
      telemetry->reference_ready ? "yes" : "no",
      reference_update_reason_to_string(telemetry->reference_update_reason),
      telemetry->gate_open ? "open" : "closed",
      telemetry->inference_ran ? "ran" : "skipped",
      telemetry->comparison_forced ? "yes" : "no",
      static_cast<unsigned>(g_skipped_inference_runs),
      static_cast<unsigned>(g_forced_inference_runs),
      static_cast<unsigned>(telemetry->peak_heap_used),
      static_cast<unsigned>(telemetry->peak_psram_used));

  const int64_t output_end_us = esp_timer_get_time();
  telemetry->output_us = output_end_us - output_start_us;
  std::printf(
      "[output] report_us=%lld free_heap=%u largest_heap_block=%u free_psram=%u "
      "largest_psram_block=%u current_heap=%u current_psram=%u\n",
      static_cast<long long>(telemetry->output_us),
      static_cast<unsigned>(telemetry->current_free_heap),
      static_cast<unsigned>(telemetry->current_largest_heap_block),
      static_cast<unsigned>(telemetry->current_free_psram),
      static_cast<unsigned>(telemetry->current_largest_psram_block),
      static_cast<unsigned>(telemetry->current_free_heap),
      static_cast<unsigned>(telemetry->current_free_psram));
}

void emit_energy_window_report(const zcci::EnergyWindowReport &report) {
  std::printf(
      "[energy] window=%u mode=%s schedule=%s elapsed_ms=%llu frames=%u reference_ready_frames=%u "
      "skipped=%u inference=%u forced=%u skip_rate_pct=%u "
      "capture_ms=%llu gate_ms=%llu inference_ms=%llu output_ms=%llu busy_ms=%llu idle_ms=%llu "
      "busy_permille=%u idle_permille=%u avoided_inference_ms=%llu net_compute_saved_ms=%llu "
      "estimated_energy_mj=%llu profile_mv=%u cpu_active_ma=%u cpu_idle_ma=%u camera_ma=%u "
      "psram_ma=%u board_baseline_ma=%u reference_inference_us=%u\n",
      static_cast<unsigned>(report.window_index),
      comparison_mode_to_string(),
      zcci::kFrameScheduleModeName,
      static_cast<unsigned long long>(report.elapsed_us / 1000ULL),
      static_cast<unsigned>(report.frames),
      static_cast<unsigned>(report.reference_ready_frames),
      static_cast<unsigned>(report.skipped_frames),
      static_cast<unsigned>(report.inference_frames),
      static_cast<unsigned>(report.forced_frames),
      static_cast<unsigned>(percent(report.skipped_frames, report.reference_ready_frames)),
      static_cast<unsigned long long>(report.capture_us / 1000ULL),
      static_cast<unsigned long long>(report.gate_us / 1000ULL),
      static_cast<unsigned long long>(report.inference_us / 1000ULL),
      static_cast<unsigned long long>(report.output_us / 1000ULL),
      static_cast<unsigned long long>(report.busy_us / 1000ULL),
      static_cast<unsigned long long>(report.idle_us / 1000ULL),
      static_cast<unsigned>(permille(report.busy_us, report.elapsed_us)),
      static_cast<unsigned>(permille(report.idle_us, report.elapsed_us)),
      static_cast<unsigned long long>(report.avoided_inference_us / 1000ULL),
      static_cast<unsigned long long>(report.net_compute_saved_us / 1000ULL),
      static_cast<unsigned long long>(report.estimated_energy_uj / 1000ULL),
      static_cast<unsigned>(kBoardEnergyProfile.supply_mv),
      static_cast<unsigned>(kBoardEnergyProfile.cpu_active_ma),
      static_cast<unsigned>(kBoardEnergyProfile.cpu_idle_ma),
      static_cast<unsigned>(kBoardEnergyProfile.camera_active_ma),
      static_cast<unsigned>(kBoardEnergyProfile.psram_active_ma),
      static_cast<unsigned>(kBoardEnergyProfile.board_baseline_ma),
      static_cast<unsigned>(kBoardEnergyProfile.reference_inference_us));
}

void update_energy_estimator(const FrameTelemetry &telemetry) {
  zcci::EnergyWindowReport report;
  const zcci::EnergyFrameSample sample = {
      .timestamp_us = static_cast<uint64_t>(esp_timer_get_time()),
      .capture_us = nonnegative_us(telemetry.capture_us),
      .gate_us = nonnegative_us(telemetry.gate_us),
      .inference_us = nonnegative_us(telemetry.inference_us),
      .output_us = nonnegative_us(telemetry.output_us),
      .reference_ready = telemetry.reference_ready,
      .inference_ran = telemetry.inference_ran,
      .comparison_forced = telemetry.comparison_forced,
  };

  if (zcci::record_energy_frame(
          &g_energy_estimator, kEnergyEstimatorConfig, kBoardEnergyProfile, sample, &report)) {
    emit_energy_window_report(report);
  }
}

void process_frame_cycle() {
  FrameTelemetry telemetry;

  const int64_t capture_start_us = esp_timer_get_time();
  camera_fb_t *fb = esp_camera_fb_get();
  const int64_t capture_end_us = esp_timer_get_time();
  telemetry.capture_us = capture_end_us - capture_start_us;

  if (fb == nullptr) {
    std::printf("[capture] frame_capture_failed\n");
    return;
  }

  std::printf(
      "[capture] latency_us=%lld format=%s width=%u height=%u length=%u buf=%p\n",
      static_cast<long long>(capture_end_us - capture_start_us),
      pixformat_to_string(fb->format),
      static_cast<unsigned>(fb->width),
      static_cast<unsigned>(fb->height),
      static_cast<unsigned>(fb->len),
      fb->buf);

  if (fb->format != PIXFORMAT_GRAYSCALE) {
    std::printf("[gate] unsupported_format=%s expected=GRAYSCALE\n",
                  pixformat_to_string(fb->format));
    esp_camera_fb_return(fb);
    log_memory_headroom("after_capture");
    return;
  }

  emit_first_frame_snapshot(fb);

  const int64_t gate_start_us = esp_timer_get_time();
  const zcci::GateResult gate_result =
      zcci::evaluate_gate_frame(&g_gate_state, frame_view_from_camera_fb(fb), kGateConfig);
  const int64_t gate_end_us = esp_timer_get_time();
  telemetry.gate_us = gate_end_us - gate_start_us;
  telemetry.score = gate_result.score;
  telemetry.reference_ready = gate_result.reference_ready;
  telemetry.reference_updated = gate_result.reference_updated;
  telemetry.reference_update_reason = gate_result.reference_update_reason;

  if (gate_result.decision == zcci::GateDecision::kReferenceInitialized) {
    std::printf("[gate] reference_initialized cells=%u grid=%ux%u\n",
                  static_cast<unsigned>(zcci::kGateCellCount),
                  static_cast<unsigned>(zcci::kGateGridWidth),
                  static_cast<unsigned>(zcci::kGateGridHeight));
    log_gate_result(0, "skip_reference_only", gate_result.reference_update_reason);
    telemetry.gate_open = false;
    telemetry.inference_ran = false;
  } else {
    const bool changed = gate_result.changed;
    const bool force_model_path = zcci::comparison_mode_forces_inference(zcci::kComparisonMode);
    const bool model_should_run = changed || force_model_path;
    telemetry.gate_open = model_should_run;
    telemetry.inference_ran = model_should_run;
    telemetry.comparison_forced = force_model_path && !changed;

    if (model_should_run) {
      const int64_t inference_start_us = esp_timer_get_time();
      run_expensive_path_model(fb, telemetry.score);
      const int64_t inference_end_us = esp_timer_get_time();
      telemetry.inference_us = inference_end_us - inference_start_us;
      if (telemetry.comparison_forced) {
        ++g_forced_inference_runs;
        log_gate_result(telemetry.score, "force_expensive_path", gate_result.reference_update_reason);
      } else {
        log_gate_result(telemetry.score, "run_expensive_path", gate_result.reference_update_reason);
      }
    } else {
      ++g_skipped_inference_runs;
      std::printf("[inference] skipped score=%u\n", static_cast<unsigned>(telemetry.score));
      log_gate_result(telemetry.score, "skip_expensive_path", gate_result.reference_update_reason);
    }
    if (!model_should_run) {
      telemetry.inference_us = 0;
    }
  }

  esp_camera_fb_return(fb);
  log_memory_headroom("after_capture");

  telemetry.total_us = telemetry.capture_us + telemetry.gate_us + telemetry.inference_us;
  const MemorySnapshot snapshot = read_memory_snapshot();
  update_memory_peaks(snapshot);
  telemetry.peak_heap_used = peak_used_bytes(g_boot_free_heap, g_min_free_heap);
  telemetry.peak_psram_used = peak_used_bytes(g_boot_free_psram, g_min_free_psram);
  telemetry.current_free_heap = snapshot.free_heap;
  telemetry.current_largest_heap_block = snapshot.largest_heap_block;
  telemetry.current_free_psram = snapshot.free_psram;
  telemetry.current_largest_psram_block = snapshot.largest_psram_block;
  emit_frame_report(&telemetry);
  update_energy_estimator(telemetry);
}

}  // namespace

extern "C" void app_main(void) {
  vTaskDelay(pdMS_TO_TICKS(200));
  std::printf("\n");
  std::printf("[boot] zero-copy-camera-inference ESP-IDF gate scaffold\n");
  std::printf("[boot] comparison_mode=%s build_flag=%s\n",
                comparison_mode_to_string(),
                zcci::kComparisonModeBuildFlag);
  std::printf("[boot] frame_schedule=%s build_flag=%s frame_period_ms=%u delay_until=%s\n",
                zcci::kFrameScheduleModeName,
                zcci::kFrameScheduleBuildFlag,
                static_cast<unsigned>(zcci::kFramePeriodMs),
                zcci::kFrameDelayUntilEnabled ? "enabled" : "disabled");
  std::printf("[energy] window_ms=%llu profile_mv=%u cpu_active_ma=%u cpu_idle_ma=%u camera_ma=%u "
              "psram_ma=%u board_baseline_ma=%u reference_inference_us=%u\n",
                static_cast<unsigned long long>(kEnergyEstimatorConfig.window_us / 1000ULL),
                static_cast<unsigned>(kBoardEnergyProfile.supply_mv),
                static_cast<unsigned>(kBoardEnergyProfile.cpu_active_ma),
                static_cast<unsigned>(kBoardEnergyProfile.cpu_idle_ma),
                static_cast<unsigned>(kBoardEnergyProfile.camera_active_ma),
                static_cast<unsigned>(kBoardEnergyProfile.psram_active_ma),
                static_cast<unsigned>(kBoardEnergyProfile.board_baseline_ma),
                static_cast<unsigned>(kBoardEnergyProfile.reference_inference_us));
  std::printf("[snapshot] first_frame_delay_ms=%u\n",
                static_cast<unsigned>(kFirstSnapshotDelayMs));
  std::printf("[gate] grid=%ux%u trigger=%u clear=%u trigger_frames=%u clear_frames=%u "
              "stable_reference_update_frames=%u changed_reference_reset_frames=%u\n",
                static_cast<unsigned>(zcci::kGateGridWidth),
                static_cast<unsigned>(zcci::kGateGridHeight),
                static_cast<unsigned>(kGateConfig.trigger_threshold),
                static_cast<unsigned>(kGateConfig.clear_threshold),
                static_cast<unsigned>(kGateConfig.trigger_consecutive_frames),
                static_cast<unsigned>(kGateConfig.clear_consecutive_frames),
                static_cast<unsigned>(kGateConfig.stable_reference_update_frames),
                static_cast<unsigned>(kGateConfig.changed_reference_reset_frames));

  log_memory_headroom("boot");
  configure_camera();
  const MemorySnapshot baseline = read_memory_snapshot();
  g_boot_free_heap = baseline.free_heap;
  g_boot_free_psram = baseline.free_psram;
  g_min_free_heap = baseline.free_heap;
  g_min_free_psram = baseline.free_psram;
  vTaskDelay(pdMS_TO_TICKS(kFirstSnapshotDelayMs));
  process_frame_cycle();

  TickType_t next_frame_wake = xTaskGetTickCount();
  while (true) {
    if (zcci::kFrameDelayUntilEnabled) {
      vTaskDelayUntil(&next_frame_wake, pdMS_TO_TICKS(zcci::kFramePeriodMs));
    } else {
      vTaskDelay(pdMS_TO_TICKS(zcci::kFramePeriodMs));
    }
    process_frame_cycle();
  }
}
