package com.aipipe.app
import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.AndroidViewModel
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
import java.io.File
import java.io.FileOutputStream
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
class PipelineViewModel(application: Application) : AndroidViewModel(application) {
private val bridge = NativePipelineBridge()
private val _uiState = MutableStateFlow(PipelineUiState())
val uiState: StateFlow<PipelineUiState> = _uiState.asStateFlow()
private var timerJob: Job? = null
init {
viewModelScope.launch(Dispatchers.IO) {
try {
val context = getApplication<Application>().applicationContext
val unet = extractModelIfMissing(context, "unet.mnn")
val vae = extractModelIfMissing(context, "vae_decoder.mnn")
val text = extractModelIfMissing(context, "text_encoder.mnn")
val unetPath = resolvePath(unet, "unet.mnn")
val vaePath = resolvePath(vae, "vae_decoder.mnn")
val textPath = resolvePath(text, "text_encoder.mnn")
val ok = bridge.nativeInit(context.filesDir.absolutePath, unetPath, vaePath, textPath)
if (ok) {
Log.i("AI_PIPE", "Native pipeline initialized successfully on startup")
_uiState.update { it.copy(statusMessage = "Модели MNN загружены") }
} else {
Log.w("AI_PIPE", "Native pipeline awaiting models: unet=$unetPath, vae=$vaePath, text=$textPath")
_uiState.update { it.copy(statusMessage = "Готов к запуску") }
}
} catch (t: Throwable) {
Log.e("AI_PIPE", "Failed to initialize native pipeline", t)
_uiState.update { it.copy(errorMessage = "Ошибка инициализации: ${t.message}") }
}
}
}
private fun extractModelIfMissing(context: Context, fileName: String): File {
val targetFile = File(context.filesDir, fileName)
if (!targetFile.exists() || targetFile.length() == 0L) {
try {
context.assets.open(fileName).use { input ->
FileOutputStream(targetFile).use { output ->
val buffer = ByteArray(65536)
var read: Int
while (input.read(buffer).also { read = it } != -1) {
output.write(buffer, 0, read)
}
output.flush()
}
}
Log.i("AI_PIPE", "Extracted asset $fileName to ${targetFile.absolutePath}")
} catch (e: Exception) {
Log.w("AI_PIPE", "Asset $fileName not found in assets: ${e.message}")
}
}
return targetFile
}
private fun resolvePath(internalFile: File, fileName: String): String {
if (internalFile.exists() && internalFile.length() > 0L) {
return internalFile.absolutePath
}
val tmpFile = File("/data/local/tmp/models", fileName)
if (tmpFile.exists() && tmpFile.length() > 0L) {
return tmpFile.absolutePath
}
val sdFile = File("/sdcard/models", fileName)
if (sdFile.exists() && sdFile.length() > 0L) {
return sdFile.absolutePath
}
return internalFile.absolutePath
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
viewModelScope.launch(Dispatchers.IO) {
try {
val context = getApplication<Application>().applicationContext
val unet = extractModelIfMissing(context, "unet.mnn")
val vae = extractModelIfMissing(context, "vae_decoder.mnn")
val text = extractModelIfMissing(context, "text_encoder.mnn")
val unetPath = resolvePath(unet, "unet.mnn")
val vaePath = resolvePath(vae, "vae_decoder.mnn")
val textPath = resolvePath(text, "text_encoder.mnn")
val initOk = bridge.nativeInit(context.filesDir.absolutePath, unetPath, vaePath, textPath)
if (!initOk) {
timerJob?.cancel()
val msg = "Файлы моделей не найдены: unet.mnn, vae_decoder.mnn, text_encoder.mnn"
Log.e("AI_PIPE", msg)
_uiState.update {
it.copy(
isRunning = false,
step = PipelineStep.ERROR,
errorMessage = msg
)
}
return@launch
}
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
errorMessage = "Ошибка инференса MNN/NCNN: проверьте файлы моделей"
)
}
}
} catch (t: Throwable) {
timerJob?.cancel()
Log.e("AI_PIPE", "Pipeline execution error", t)
_uiState.update {
it.copy(
isRunning = false,
step = PipelineStep.ERROR,
errorMessage = t.localizedMessage ?: "Сбой выполнения"
)
}
}
}
}
override fun onCleared() {
super.onCleared()
timerJob?.cancel()
try {
bridge.nativeRelease()
} catch (t: Throwable) {
Log.e("AI_PIPE", "Error during nativeRelease", t)
}
}
}
