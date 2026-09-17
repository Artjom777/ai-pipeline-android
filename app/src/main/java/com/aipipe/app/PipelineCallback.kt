package com.aipipe.app
interface PipelineCallback {
fun onProgress(stage: Int, progress: Float, message: String)
}
