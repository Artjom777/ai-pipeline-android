package com.aipipe.app
import android.graphics.Bitmap
class NativePipelineBridge {
companion object {
init {
System.loadLibrary("native_pipeline")
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
