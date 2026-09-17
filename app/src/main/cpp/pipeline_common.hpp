#ifndef PIPELINE_COMMON_HPP
#define PIPELINE_COMMON_HPP
#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#define TAG "AI_PIPE"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
struct PipelineConfig {
int genWidth = 512;
int genHeight = 512;
int genSteps = 4;
int targetWidth = 3840;
int targetHeight = 2160;
int tileSize = 256;
int tileOverlap = 16;
float maxBudgetSec = 30.0f;
};
struct ExecutionMetrics {
float diffusionDurationMs = 0.0f;
float upscaleDurationMs = 0.0f;
float totalDurationMs = 0.0f;
bool fallbackTriggered = false;
bool openClBuffersFreed = false;
};
class IProgressCallback {
public:
virtual ~IProgressCallback() = default;
virtual void onStageProgress(int stage, float progress, const char* statusMsg) = 0;
};
#endif
