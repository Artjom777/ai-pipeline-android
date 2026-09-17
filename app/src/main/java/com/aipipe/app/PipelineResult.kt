package com.aipipe.app
data class PipelineResult(
val diffusionDurationMs: Float,
val upscaleDurationMs: Float,
val totalDurationMs: Float,
val fallbackTriggered: Boolean,
val withinBudget: Boolean
)
