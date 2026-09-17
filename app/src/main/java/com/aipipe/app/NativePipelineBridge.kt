package com.aipipe.app
import android.graphics.Bitmap
import android.util.Log
class NativePipelineBridge {
companion object {
private const val TAG = "AI_PIPE"
init {
try {
System.loadLibrary("c++_shared")
Log.i(TAG, "Loaded c++_shared successfully")
} catch (e: Throwable) {
Log.e(TAG, "Failed loading c++_shared", e)
}
try {
System.loadLibrary("MNN")
Log.i(TAG, "Loaded MNN successfully")
} catch (e: Throwable) {
Log.e(TAG, "Failed loading MNN", e)
}
try {
System.loadLibrary("MNN_CL")
Log.i(TAG, "Loaded MNN_CL successfully")
} catch (e: Throwable) {
Log.w(TAG, "Failed loading MNN_CL: " + e.message)
}
try {
System.loadLibrary("native_pipeline")
Log.i(TAG, "Loaded native_pipeline successfully")
} catch (e: Throwable) {
Log.e(TAG, "Failed loading native_pipeline", e)
throw e
}
}
}
external fun nativeInit(modelDir: String, unetPath: String, vaePath: String, textEncoderPath: String): Boolean
external fun nativeExecutePipeline(
prompt: String,
timeoutSec: Int,
targetW: Int,
targetH: Int,
callback: PipelineCallback,
outBitmap512: Bitmap,
outBitmap4K: Bitmap
): PipelineResult?
external fun nativeRelease()
}
