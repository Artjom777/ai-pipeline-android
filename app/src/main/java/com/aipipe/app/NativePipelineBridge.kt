package com.aipipe.app
import android.graphics.Bitmap
class NativePipelineBridge {
companion object {
init {
try {
System.loadLibrary("c++_shared")
} catch (_: Throwable) {}
try {
System.loadLibrary("OpenCL")
} catch (_: Throwable) {}
try {
System.loadLibrary("MNN")
} catch (_: Throwable) {}
try {
System.loadLibrary("native_pipeline")
} catch (_: Throwable) {}
}
}
external fun nativeInit(modelDir: String): Boolean
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
