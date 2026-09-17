package com.aipipe.app.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.aipipe.app.PipelineStep
import com.aipipe.app.PipelineUiState
import com.aipipe.app.PipelineViewModel
import com.aipipe.app.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PipelineScreen(viewModel: PipelineViewModel) {
    val state by viewModel.uiState.collectAsState()
    var selectedPreviewTab by remember { mutableIntStateOf(0) }
    val scrollState = rememberScrollState()

    val samplePrompts = listOf(
        "A cyberpunk neon city in rain",
        "Macro photograph of a mechanical bee",
        "Sci-fi spacecraft orbiting gas giant"
    )

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Bolt,
                            contentDescription = null,
                            tint = NeonCyan,
                            modifier = Modifier.size(28.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Column {
                            Text(
                                "Mali AI Pipeline (<=30s)",
                                fontWeight = FontWeight.Bold,
                                fontSize = 18.sp,
                                color = TextPrimary
                            )
                            Text(
                                "MNN OpenCL + NCNN Vulkan",
                                fontSize = 12.sp,
                                color = TextSecondary
                            )
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = CardBackground)
            )
        },
        containerColor = DarkBackground
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(scrollState)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            ChipsetSpecCard()

            Card(
                colors = CardDefaults.cardColors(containerColor = CardBackground),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(modifier = Modifier.padding(14.dp)) {
                    Text(
                        "Промпт для LCM/SD-Turbo:",
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary,
                        fontSize = 14.sp
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedTextField(
                        value = state.prompt,
                        onValueChange = { viewModel.onPromptChanged(it) },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !state.isRunning,
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = NeonCyan,
                            unfocusedBorderColor = BorderColor,
                            focusedTextColor = TextPrimary,
                            unfocusedTextColor = TextPrimary
                        ),
                        maxLines = 3
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        samplePrompts.forEach { p ->
                            SuggestionChip(
                                onClick = { viewModel.onPromptChanged(p) },
                                label = { Text(p.take(18) + "...", fontSize = 11.sp) },
                                enabled = !state.isRunning,
                                colors = SuggestionChipDefaults.suggestionChipColors(
                                    containerColor = DarkBackground,
                                    labelColor = TextSecondary
                                )
                            )
                        }
                    }
                }
            }

            Button(
                onClick = { viewModel.startPipeline() },
                enabled = !state.isRunning,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(50.dp),
                shape = RoundedCornerShape(10.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = NeonCyan,
                    contentColor = DarkBackground,
                    disabledContainerColor = BorderColor
                )
            ) {
                if (state.isRunning) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(22.dp),
                        color = DarkBackground,
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Text("Обработка (Stage ${state.stage}/2)...", fontWeight = FontWeight.Bold)
                } else {
                    Icon(imageVector = Icons.Default.Speed, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Запустить пайплайн (Лимит 30с)", fontWeight = FontWeight.Bold)
                }
            }

            ExecutionStatusCard(state)

            PreviewCard(
                state = state,
                selectedTab = selectedPreviewTab,
                onTabSelected = { selectedPreviewTab = it }
            )

            state.metrics?.let { metrics ->
                MetricsCard(metrics)
            }
        }
    }
}

@Composable
fun ChipsetSpecCard() {
    Card(
        colors = CardDefaults.cardColors(containerColor = CardBackground),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.Memory,
                    contentDescription = null,
                    tint = MaliOrange,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    "Оптимизации MediaTek / Mali",
                    fontWeight = FontWeight.Bold,
                    color = MaliOrange,
                    fontSize = 13.sp
                )
            }
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                "• Этап 1: MNN OpenCL (MNN_OPENCL=ON, MNN_LOW_MEMORY=ON), 4 шага LCM, 512x512.\n" +
                "• Без safety_checker/NSFW (0MB оверхеда).\n" +
                "• Принудительный сброс буферов OpenCL перед Этапом 2.\n" +
                "• Этап 2: NCNN Vulkan Real-ESRGAN Compact с тайлингом 256x256 (overlap 16px).\n" +
                "• 2-этапный фоллбэк: 2x нейросеть + GPU Lanczos при нехватке лимита 30с.",
                fontSize = 11.sp,
                color = TextSecondary,
                lineHeight = 16.sp
            )
        }
    }
}

@Composable
fun ExecutionStatusCard(state: PipelineUiState) {
    Card(
        colors = CardDefaults.cardColors(containerColor = CardBackground),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Статус выполнения:", fontWeight = FontWeight.SemiBold, color = TextPrimary, fontSize = 13.sp)
                val badgeColor = when (state.step) {
                    PipelineStep.IDLE -> TextSecondary
                    PipelineStep.STAGE_1_DIFFUSION -> NeonPurple
                    PipelineStep.STAGE_1_COMPLETED -> NeonGreen
                    PipelineStep.STAGE_2_UPSCALE -> NeonCyan
                    PipelineStep.COMPLETED -> NeonGreen
                    PipelineStep.ERROR -> Color.Red
                }
                Text(
                    state.step.name,
                    color = badgeColor,
                    fontWeight = FontWeight.Bold,
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace
                )
            }
            Spacer(modifier = Modifier.height(6.dp))
            LinearProgressIndicator(
                progress = { state.progress },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp)
                    .clip(RoundedCornerShape(3.dp)),
                color = if (state.stage == 1) NeonPurple else NeonCyan,
                trackColor = BorderColor
            )
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                state.statusMessage,
                color = TextSecondary,
                fontSize = 12.sp,
                fontFamily = FontFamily.Monospace
            )
            if (state.errorMessage != null) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(state.errorMessage, color = Color.Red, fontSize = 12.sp)
            }
        }
    }
}

@Composable
fun PreviewCard(
    state: PipelineUiState,
    selectedTab: Int,
    onTabSelected: (Int) -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = CardBackground),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            TabRow(
                selectedTabIndex = selectedTab,
                containerColor = DarkBackground,
                contentColor = NeonCyan
            ) {
                Tab(
                    selected = selectedTab == 0,
                    onClick = { onTabSelected(0) },
                    text = { Text("Stage 1 (512x512)", fontSize = 12.sp) }
                )
                Tab(
                    selected = selectedTab == 1,
                    onClick = { onTabSelected(1) },
                    text = { Text("Stage 2 (4K Upscale)", fontSize = 12.sp) }
                )
            }
            Spacer(modifier = Modifier.height(10.dp))

            val currentBitmap = if (selectedTab == 0) state.bitmap512 else (state.bitmap4K ?: state.bitmap512)
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(260.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(DarkBackground)
                    .border(1.dp, BorderColor, RoundedCornerShape(8.dp)),
                contentAlignment = Alignment.Center
            ) {
                if (currentBitmap != null) {
                    Image(
                        bitmap = currentBitmap.asImageBitmap(),
                        contentDescription = "Preview",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Fit
                    )
                } else {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Icon(
                            imageVector = Icons.Default.Bolt,
                            contentDescription = null,
                            tint = BorderColor,
                            modifier = Modifier.size(48.dp)
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            if (selectedTab == 0) "Ожидание Stage 1 (MNN LCM)..." else "Ожидание Stage 2 (NCNN 4K)...",
                            color = TextSecondary,
                            fontSize = 12.sp
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun MetricsCard(metrics: com.aipipe.app.PipelineResult) {
    Card(
        colors = CardDefaults.cardColors(containerColor = CardBackground),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Метрики тайминга (Mali/MNN/NCNN):", fontWeight = FontWeight.Bold, color = TextPrimary, fontSize = 13.sp)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = if (metrics.withinBudget) Icons.Default.CheckCircle else Icons.Default.Warning,
                        contentDescription = null,
                        tint = if (metrics.withinBudget) NeonGreen else Color.Red,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        if (metrics.withinBudget) "Лимит <=30c: PASS" else "Лимит превышен",
                        color = if (metrics.withinBudget) NeonGreen else Color.Red,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            MetricRow("Генерация MNN (512x512, 4 шага LCM):", "${metrics.diffusionDurationMs.toInt()} ms")
            MetricRow("Апскейл NCNN (Real-ESRGAN Vulkan):", "${metrics.upscaleDurationMs.toInt()} ms")
            MetricRow("Суммарное время пайплайна:", "${metrics.totalDurationMs.toInt()} ms (${String.format("%.2f", metrics.totalDurationMs / 1000f)} с)")
            MetricRow(
                "Режим 2-этапного масштабирования:",
                if (metrics.fallbackTriggered) "АКТИВИРОВАН (2x нейросеть + GPU Lanczos)" else "Стандартный тайлинг"
            )
            MetricRow("Очистка OpenCL буферов:", "ВЫПОЛНЕНА")
            MetricRow("Safety Checker / NSFW Filter:", "ИСКЛЮЧЕН ИЗ ГРАФА")
        }
    }
}

@Composable
fun MetricRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, color = TextSecondary, fontSize = 11.sp)
        Text(value, color = TextPrimary, fontSize = 11.sp, fontWeight = FontWeight.Medium)
    }
}
