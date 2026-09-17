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
import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.concurrent.TimeUnit
import java.util.Locale
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
val remainingSeconds: Int = 30,
val modelsMissing: Boolean = false,
val missingModelsList: List<String> = emptyList(),
val isDownloading: Boolean = false,
val downloadProgress: Float = 0f,
val downloadSpeed: String = "0.0 МБ/с",
val downloadStatus: String = ""
)
class PipelineViewModel(application: Application) : AndroidViewModel(application) {
private val bridge = NativePipelineBridge()
private val okHttpClient = OkHttpClient.Builder().followRedirects(true).followSslRedirects(true).connectTimeout(30, TimeUnit.SECONDS).readTimeout(120, TimeUnit.SECONDS).build()
private val _uiState = MutableStateFlow(PipelineUiState())
val uiState: StateFlow<PipelineUiState> = _uiState.asStateFlow()
private var timerJob: Job? = null
init {
viewModelScope.launch(Dispatchers.IO) {
try {
val context = getApplication<Application>().applicationContext
extractModelIfMissing(context, "unet.mnn")
extractModelIfMissing(context, "text_encoder.mnn")
extractModelIfMissing(context, "vae_decoder.mnn")
val missing = getMissingModels(context)
if (missing.isNotEmpty()) {
Log.w("AI_PIPE", "Missing models on startup: ${missing.joinToString(", ")}")
_uiState.update {
it.copy(
statusMessage = "Требуется загрузка моделей",
modelsMissing = true,
missingModelsList = missing,
errorMessage = null
)
}
return@launch
}
val unetPath = File(context.filesDir, "unet.mnn").absolutePath
val textPath = File(context.filesDir, "text_encoder.mnn").absolutePath
val vaePath = File(context.filesDir, "vae_decoder.mnn").absolutePath
val initCode = bridge.nativeInit(context.filesDir.absolutePath, unetPath, vaePath, textPath)
if (initCode == 0) {
Log.i("AI_PIPE", "Native pipeline initialized successfully on startup")
_uiState.update { it.copy(statusMessage = "Модели MNN загружены", errorMessage = null) }
} else {
val err = "Ошибка инициализации MNN (код $initCode)"
Log.w("AI_PIPE", err)
_uiState.update { it.copy(statusMessage = "Готов к запуску", errorMessage = err) }
}
} catch (t: Throwable) {
Log.e("AI_PIPE", "Failed to initialize native pipeline", t)
_uiState.update { it.copy(errorMessage = "Ошибка инициализации: ${t.message}") }
}
}
}
private fun getMissingModels(context: Context): List<String> {
val required = listOf("unet.mnn", "text_encoder.mnn", "vae_decoder.mnn")
return required.filter { fileName ->
val f = File(context.filesDir, fileName)
!f.exists() || f.length() == 0L
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
if (targetFile.exists() && targetFile.length() == 0L) {
targetFile.delete()
}
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
fun downloadModels() {
if (_uiState.value.isDownloading) return
_uiState.update {
it.copy(
isDownloading = true,
downloadProgress = 0f,
downloadSpeed = "0.0 МБ/с",
downloadStatus = "Подключение к Hugging Face LFS...",
errorMessage = null
)
}
viewModelScope.launch(Dispatchers.IO) {
try {
val context = getApplication<Application>().applicationContext
val models = listOf(
Pair("text_encoder.mnn", "https://huggingface.co/taobao-mnn/stable-diffusion-v1-5-mnn-opencl/resolve/main/text_encoder.mnn"),
Pair("vae_decoder.mnn", "https://huggingface.co/taobao-mnn/stable-diffusion-v1-5-mnn-opencl/resolve/main/vae_decoder.mnn"),
Pair("unet.mnn", "https://huggingface.co/taobao-mnn/stable-diffusion-v1-5-mnn-opencl/resolve/main/unet.mnn")
)
val totalModels = models.size
var completedModels = 0
for ((fileName, urlStr) in models) {
val targetFile = File(context.filesDir, fileName)
if (targetFile.exists() && targetFile.length() > 10000L) {
completedModels++
_uiState.update {
it.copy(
downloadProgress = completedModels.toFloat() / totalModels.toFloat(),
downloadStatus = "$fileName готов ($completedModels/$totalModels)"
)
}
continue
}
val tmpFile = File(context.filesDir, "$fileName.tmp")
if (tmpFile.exists()) tmpFile.delete()
_uiState.update {
it.copy(
downloadStatus = "Скачивание $fileName (${completedModels + 1}/$totalModels)..."
)
}
val request = Request.Builder()
.url(urlStr)
.header("User-Agent", "Mozilla/5.0")
.build()
okHttpClient.newCall(request).execute().use { response ->
if (!response.isSuccessful) {
throw java.io.IOException("HTTP error ${response.code} for $fileName: ${response.message}")
}
val body = response.body ?: throw java.io.IOException("Empty response body for $fileName")
val contentLength = body.contentLength()
body.byteStream().use { input ->
FileOutputStream(tmpFile).use { output ->
val buffer = ByteArray(65536)
var bytesRead = 0L
var lastTime = System.currentTimeMillis()
var lastBytes = 0L
var speedMBs = 0f
var read: Int
while (input.read(buffer).also { read = it } != -1) {
output.write(buffer, 0, read)
bytesRead += read
val now = System.currentTimeMillis()
val elapsed = now - lastTime
if (elapsed >= 500) {
val diffBytes = bytesRead - lastBytes
speedMBs = (diffBytes.toFloat() / (elapsed.toFloat() / 1000f)) / (1024f * 1024f)
lastTime = now
lastBytes = bytesRead
val fileProgress = if (contentLength > 0) bytesRead.toFloat() / contentLength.toFloat() else 0f
val overallProgress = (completedModels.toFloat() + fileProgress) / totalModels.toFloat()
val speedText = String.format(Locale.US, "%.1f МБ/с", speedMBs)
_uiState.update {
it.copy(
downloadProgress = overallProgress.coerceIn(0f, 1f),
downloadSpeed = speedText,
downloadStatus = "Скачивание $fileName: ${(fileProgress * 100).toInt()}%"
)
}
}
}
output.flush()
}
}
}
if (!tmpFile.exists() || tmpFile.length() == 0L) {
throw java.io.IOException("Failed to write $fileName: 0 bytes received")
}
if (targetFile.exists()) targetFile.delete()
val renamed = tmpFile.renameTo(targetFile)
if (!renamed) {
tmpFile.copyTo(targetFile, overwrite = true)
tmpFile.delete()
}
completedModels++
_uiState.update {
it.copy(
downloadProgress = completedModels.toFloat() / totalModels.toFloat(),
downloadStatus = "$fileName сохранён ($completedModels/$totalModels)"
)
}
}
val missing = getMissingModels(context)
if (missing.isEmpty()) {
val unetPath = File(context.filesDir, "unet.mnn").absolutePath
val textPath = File(context.filesDir, "text_encoder.mnn").absolutePath
val vaePath = File(context.filesDir, "vae_decoder.mnn").absolutePath
val initCode = bridge.nativeInit(context.filesDir.absolutePath, unetPath, vaePath, textPath)
_uiState.update {
it.copy(
isDownloading = false,
modelsMissing = false,
downloadProgress = 1f,
statusMessage = if (initCode == 0) "Модели MNN загружены" else "Готов к запуску",
errorMessage = null
)
}
} else {
_uiState.update {
it.copy(
isDownloading = false,
modelsMissing = true,
errorMessage = "Не все файлы сохранены: ${missing.joinToString(", ")}"
)
}
}
} catch (e: Exception) {
val err = e.localizedMessage ?: e.message ?: "Сбой соединения при загрузке весов"
Log.e("AI_PIPE", "Model download error: $err", e)
_uiState.update {
it.copy(
isDownloading = false,
errorMessage = "Ошибка скачивания: $err"
)
}
}
}
}
fun startPipeline() {
if (_uiState.value.isRunning || _uiState.value.isDownloading) return
val context = getApplication<Application>().applicationContext
val missing = getMissingModels(context)
if (missing.isNotEmpty()) {
_uiState.update {
it.copy(
isRunning = false,
modelsMissing = true,
missingModelsList = missing,
statusMessage = "Требуется загрузка моделей"
)
}
return
}
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
val unetPath = File(context.filesDir, "unet.mnn").absolutePath
val textPath = File(context.filesDir, "text_encoder.mnn").absolutePath
val vaePath = File(context.filesDir, "vae_decoder.mnn").absolutePath
val initCode = bridge.nativeInit(context.filesDir.absolutePath, unetPath, vaePath, textPath)
if (initCode != 0) {
timerJob?.cancel()
val msg = "Отсутствуют файлы моделей: ${missing.joinToString(", ")}. Поместите их в /data/data/com.aipipe.app/files/"
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
errorMessage = "Файлы моделей (.mnn) не найдены по пути: ${context.filesDir.absolutePath}"
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
