package com.aipipe.app
import android.graphics.Bitmap
import android.util.Log
class NativePipelineBridge {
companion object {
private const val TAG = "AI_PIPE"
private var isLoaded = false
init {
try {
System.loadLibrary("c++_shared")
Log.i(TAG, "Loaded c++_shared successfully")
} catch (t: Throwable) {
Log.e(TAG, "Failed loading c++_shared", t)
}
try {
System.loadLibrary("MNN")
Log.i(TAG, "Loaded MNN successfully")
} catch (t: Throwable) {
Log.e(TAG, "Failed loading MNN", t)
}
try {
System.loadLibrary("MNN_CL")
Log.i(TAG, "Loaded MNN_CL successfully")
} catch (t: Throwable) {
Log.w(TAG, "Failed loading MNN_CL: " + t.message)
}
try {
System.loadLibrary("native_pipeline")
Log.i(TAG, "Loaded native_pipeline successfully")
isLoaded = true
} catch (t: Throwable) {
Log.e(TAG, "Failed loading native_pipeline", t)
}
}
}
fun isNativeLoaded(): Boolean = isLoaded
fun nativeInit(modelDir: String, unetPath: String, vaePath: String, textEncoderPath: String): Int {
if (!isLoaded) {
Log.e(TAG, "nativeInit: native_pipeline library is not loaded")
return -2
}
return try {
nativeInitInternal(modelDir, unetPath, vaePath, textEncoderPath)
} catch (t: Throwable) {
Log.e(TAG, "nativeInitInternal error", t)
-2
}
}
fun nativeExecutePipeline(
prompt: String,
timeoutSec: Int,
targetW: Int,
targetH: Int,
callback: PipelineCallback,
outBitmap512: Bitmap,
outBitmap4K: Bitmap
): PipelineResult? {
if (!isLoaded) {
Log.e(TAG, "nativeExecutePipeline: native_pipeline library is not loaded")
return null
}
return try {
nativeExecutePipelineInternal(prompt, timeoutSec, targetW, targetH, callback, outBitmap512, outBitmap4K)
} catch (t: Throwable) {
Log.e(TAG, "nativeExecutePipelineInternal error", t)
null
}
}
fun nativeRelease() {
if (!isLoaded) return
try {
nativeReleaseInternal()
} catch (t: Throwable) {
Log.e(TAG, "nativeReleaseInternal error", t)
}
}
private external fun nativeInitInternal(modelDir: String, unetPath: String, vaePath: String, textEncoderPath: String): Int
private external fun nativeExecutePipelineInternal(
prompt: String,
timeoutSec: Int,
targetW: Int,
targetH: Int,
callback: PipelineCallback,
outBitmap512: Bitmap,
outBitmap4K: Bitmap
): PipelineResult?
private external fun nativeReleaseInternal()
}
