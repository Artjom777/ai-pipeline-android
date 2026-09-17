package com.aipipe.app
import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
enum class PipelineStep {
IDLE,
STAGE_1_DIFFUSION,
STAGE_1_COMPLETED,
STAGE_2_UPSCALE,
COMPLETED,
ERROR
}
data class PipelineUiState(
val step: PipelineStep = PipelineStep.IDLE,
val stage: Int = 0,
val progress: Float = 0f,
val statusMessage: String = "Готов к запуску",
val prompt: String = "A futuristic cyberpunk city at night with vibrant neon reflections in rain",
val bitmap512: Bitmap? = null,
val bitmap4K: Bitmap? = null,
val metrics: PipelineResult? = null,
val isRunning: Boolean = false,
val errorMessage: String? = null,
val steps: Int = 4,
val targetResolution: String = "4K",
val enableLanczosFallback: Boolean = true,
val remainingSeconds: Int = 30
)
class PipelineViewModel(
private val bridge: NativePipelineBridge = NativePipelineBridge()
) : ViewModel() {
private val _uiState = MutableStateFlow(PipelineUiState())
val uiState: StateFlow<PipelineUiState> = _uiState.asStateFlow()
private var timerJob: Job? = null
init {
viewModelScope.launch(Dispatchers.Default) {
bridge.nativeInit("/data/local/tmp/models")
}
}
fun onPromptChanged(newPrompt: String) {
_uiState.update { it.copy(prompt = newPrompt) }
}
fun setSteps(newSteps: Int) {
_uiState.update { it.copy(steps = newSteps.coerceIn(1, 4)) }
}
fun setTargetResolution(resolution: String) {
_uiState.update { it.copy(targetResolution = resolution) }
}
fun setLanczosFallback(enabled: Boolean) {
_uiState.update { it.copy(enableLanczosFallback = enabled) }
}
fun startPipeline() {
if (_uiState.value.isRunning) return
val currentState = _uiState.value
val prompt = currentState.prompt
val is4K = currentState.targetResolution == "4K"
val targetW = if (is4K) 3840 else 1920
val targetH = if (is4K) 2160 else 1080
val timeoutSec = 30
timerJob?.cancel()
_uiState.update {
it.copy(
isRunning = true,
step = PipelineStep.STAGE_1_DIFFUSION,
stage = 1,
progress = 0.0f,
remainingSeconds = timeoutSec,
statusMessage = "Запуск MNN OpenCL (${currentState.steps} шагов LCM)...",
errorMessage = null,
metrics = null
)
}
timerJob = viewModelScope.launch(Dispatchers.Default) {
var left = timeoutSec
while (left > 0 && _uiState.value.isRunning) {
delay(1000)
left--
_uiState.update { it.copy(remainingSeconds = left) }
}
}
viewModelScope.launch(Dispatchers.Default) {
try {
val bmp512 = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
val bmpFinal = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
val callback = object : PipelineCallback {
override fun onProgress(stage: Int, progress: Float, message: String) {
_uiState.update { current ->
val nextStep = when (stage) {
1 -> if (progress >= 1.0f) PipelineStep.STAGE_1_COMPLETED else PipelineStep.STAGE_1_DIFFUSION
2 -> if (progress >= 1.0f) PipelineStep.COMPLETED else PipelineStep.STAGE_2_UPSCALE
else -> current.step
}
current.copy(
stage = stage,
step = nextStep,
progress = progress,
statusMessage = message,
bitmap512 = if (stage >= 2 || (stage == 1 && progress >= 0.95f)) bmp512 else current.bitmap512
)
}
}
}
val result = withContext(Dispatchers.Default) {
bridge.nativeExecutePipeline(
prompt = prompt,
timeoutSec = timeoutSec,
targetW = targetW,
targetH = targetH,
callback = callback,
outBitmap512 = bmp512,
outBitmap4K = bmpFinal
)
}
timerJob?.cancel()
if (result != null) {
_uiState.update {
it.copy(
isRunning = false,
step = PipelineStep.COMPLETED,
progress = 1.0f,
statusMessage = "Завершено за ${String.format("%.1f", result.totalDurationMs / 1000f)} с",
bitmap512 = bmp512,
bitmap4K = bmpFinal,
metrics = result
)
}
} else {
_uiState.update {
it.copy(
isRunning = false,
step = PipelineStep.ERROR,
errorMessage = "Ошибка инференса MNN/NCNN"
)
}
}
} catch (e: Exception) {
timerJob?.cancel()
_uiState.update {
it.copy(
isRunning = false,
step = PipelineStep.ERROR,
errorMessage = e.localizedMessage ?: "Сбой выполнения"
)
}
}
}
}
override fun onCleared() {
super.onCleared()
timerJob?.cancel()
bridge.nativeRelease()
}
}
