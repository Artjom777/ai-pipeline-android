package com.aipipe.app.ui
import android.graphics.Bitmap
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Compare
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Timer
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.aipipe.app.PipelineUiState
import com.aipipe.app.PipelineViewModel
import com.aipipe.app.ui.theme.*
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PipelineScreen(viewModel: PipelineViewModel) {
val state by viewModel.uiState.collectAsState()
var showSettingsSheet by remember { mutableStateOf(false) }
var splitPosition by remember { mutableFloatStateOf(0.5f) }
val animatedProgress by animateFloatAsState(targetValue = state.progress, label = "progressAnim")
val scrollState = rememberScrollState()
val samplePrompts = listOf(
"Cyberpunk neon metropolis in heavy rain",
"Hyperrealistic mechanical dragonfly macro",
"Orbital station above Saturn rings 8k"
)
Scaffold(
topBar = {
TopAppBar(
title = {
Column {
Text(
text = "AI Pipeline",
style = MaterialTheme.typography.titleMedium,
fontWeight = FontWeight.Bold,
color = TextPrimary
)
Text(
text = "MNN OpenCL + NCNN Vulkan",
style = MaterialTheme.typography.bodySmall,
color = TextSecondary
)
}
},
actions = {
Surface(
shape = RoundedCornerShape(16.dp),
color = CardBackground,
border = androidx.compose.foundation.BorderStroke(1.dp, BorderColor),
modifier = Modifier.padding(end = 4.dp)
) {
Row(
verticalAlignment = Alignment.CenterVertically,
modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
) {
Icon(
imageVector = Icons.Default.Memory,
contentDescription = null,
tint = MaliOrange,
modifier = Modifier.size(14.dp)
)
Spacer(modifier = Modifier.width(4.dp))
Text(
text = "Mali-G720 / FP16",
fontSize = 11.sp,
fontWeight = FontWeight.Medium,
color = TextPrimary
)
}
}
IconButton(onClick = { showSettingsSheet = true }) {
Icon(
imageVector = Icons.Default.Tune,
contentDescription = null,
tint = NeonCyan
)
}
},
colors = TopAppBarDefaults.topAppBarColors(containerColor = DarkBackground)
)
},
bottomBar = {
Surface(
color = CardBackground,
tonalElevation = 8.dp,
border = androidx.compose.foundation.BorderStroke(1.dp, BorderColor)
) {
Column(
modifier = Modifier
.fillMaxWidth()
.padding(horizontal = 16.dp, vertical = 12.dp),
verticalArrangement = Arrangement.spacedBy(8.dp)
) {
Row(
horizontalArrangement = Arrangement.spacedBy(6.dp),
modifier = Modifier.fillMaxWidth()
) {
samplePrompts.forEach { p ->
SuggestionChip(
onClick = { viewModel.onPromptChanged(p) },
label = { Text(p.take(16) + "...", fontSize = 11.sp) },
enabled = !state.isRunning,
colors = SuggestionChipDefaults.suggestionChipColors(
containerColor = DarkBackground,
labelColor = TextSecondary
)
)
}
}
OutlinedTextField(
value = state.prompt,
onValueChange = { viewModel.onPromptChanged(it) },
modifier = Modifier.fillMaxWidth(),
enabled = !state.isRunning,
maxLines = 2,
placeholder = { Text("Опишите желаемое изображение...", color = TextSecondary) },
colors = OutlinedTextFieldDefaults.colors(
focusedBorderColor = NeonCyan,
unfocusedBorderColor = BorderColor,
focusedTextColor = TextPrimary,
unfocusedTextColor = TextPrimary
),
shape = RoundedCornerShape(12.dp)
)
if (state.errorMessage != null) {
Text(
text = state.errorMessage ?: "",
color = Color.Red,
fontSize = 12.sp,
fontWeight = FontWeight.Medium,
modifier = Modifier.fillMaxWidth()
)
}
Button(
onClick = { viewModel.startPipeline() },
enabled = !state.isRunning,
modifier = Modifier
.fillMaxWidth()
.height(52.dp),
shape = RoundedCornerShape(12.dp),
colors = ButtonDefaults.buttonColors(
containerColor = NeonCyan,
contentColor = DarkBackground,
disabledContainerColor = BorderColor
)
) {
if (state.isRunning) {
CircularProgressIndicator(
modifier = Modifier.size(20.dp),
color = DarkBackground,
strokeWidth = 2.dp
)
Spacer(modifier = Modifier.width(10.dp))
Text("Обработка на Mali GPU...", fontWeight = FontWeight.Bold)
} else {
Icon(imageVector = Icons.Default.AutoAwesome, contentDescription = null)
Spacer(modifier = Modifier.width(8.dp))
Text("Сгенерировать (${state.targetResolution})", fontWeight = FontWeight.Bold)
}
}
}
}
},
containerColor = DarkBackground
) { innerPadding ->
Column(
modifier = Modifier
.fillMaxSize()
.padding(innerPadding)
.verticalScroll(scrollState)
.padding(16.dp),
verticalArrangement = Arrangement.spacedBy(16.dp)
) {
ZoomableComparisonCard(
bitmap512 = state.bitmap512,
bitmap4K = state.bitmap4K,
splitPos = splitPosition,
onSplitChange = { splitPosition = it }
)
if (state.errorMessage != null) {
Surface(
shape = RoundedCornerShape(12.dp),
color = Color(0x33FF0000),
border = androidx.compose.foundation.BorderStroke(1.dp, Color.Red),
modifier = Modifier.fillMaxWidth()
) {
Text(
text = state.errorMessage ?: "",
color = Color.Red,
fontSize = 13.sp,
fontWeight = FontWeight.Medium,
modifier = Modifier.padding(12.dp)
)
}
}
Surface(
shape = RoundedCornerShape(12.dp),
color = CardBackground,
border = androidx.compose.foundation.BorderStroke(1.dp, BorderColor),
modifier = Modifier.fillMaxWidth()
) {
Column(modifier = Modifier.padding(12.dp)) {
Row(
modifier = Modifier.fillMaxWidth(),
horizontalArrangement = Arrangement.SpaceBetween,
verticalAlignment = Alignment.CenterVertically
) {
Text(
text = state.statusMessage,
fontSize = 12.sp,
fontWeight = FontWeight.Medium,
color = if (state.step == PipelineStep.ERROR || state.errorMessage != null) Color.Red else TextPrimary,
modifier = Modifier.weight(1f)
)
Row(
verticalAlignment = Alignment.CenterVertically,
modifier = Modifier.padding(start = 8.dp)
) {
Icon(
imageVector = Icons.Default.Timer,
contentDescription = null,
tint = if (state.remainingSeconds <= 5) Color.Red else NeonCyan,
modifier = Modifier.size(16.dp)
)
Spacer(modifier = Modifier.width(4.dp))
Text(
text = "${state.remainingSeconds}с / 30с",
fontSize = 12.sp,
fontWeight = FontWeight.Bold,
fontFamily = FontFamily.Monospace,
color = if (state.remainingSeconds <= 5) Color.Red else NeonCyan
)
}
}
Spacer(modifier = Modifier.height(8.dp))
LinearProgressIndicator(
progress = { animatedProgress },
modifier = Modifier
.fillMaxWidth()
.height(6.dp)
.clip(RoundedCornerShape(3.dp)),
color = if (state.stage == 1) NeonPurple else NeonCyan,
trackColor = BorderColor
)
}
}
}
}
if (showSettingsSheet) {
ModalBottomSheet(
onDismissRequest = { showSettingsSheet = false },
containerColor = CardBackground,
dragHandle = { BottomSheetDefaults.DragHandle(color = BorderColor) }
) {
Column(
modifier = Modifier
.fillMaxWidth()
.padding(horizontal = 24.dp)
.padding(bottom = 36.dp),
verticalArrangement = Arrangement.spacedBy(16.dp)
) {
Text(
text = "Настройки инференса",
style = MaterialTheme.typography.titleMedium,
fontWeight = FontWeight.Bold,
color = TextPrimary
)
Column {
Row(
modifier = Modifier.fillMaxWidth(),
horizontalArrangement = Arrangement.SpaceBetween
) {
Text(text = "Шаги генерации (LCM Scheduler):", fontSize = 13.sp, color = TextPrimary)
Text(text = "${state.steps}", fontSize = 13.sp, fontWeight = FontWeight.Bold, color = NeonCyan)
}
Slider(
value = state.steps.toFloat(),
onValueChange = { viewModel.setSteps(it.toInt()) },
valueRange = 1f..4f,
steps = 2,
colors = SliderDefaults.colors(
thumbColor = NeonCyan,
activeTrackColor = NeonCyan,
inactiveTrackColor = BorderColor
)
)
}
Column {
Text(text = "Итоговое разрешение апскейла:", fontSize = 13.sp, color = TextPrimary)
Spacer(modifier = Modifier.height(6.dp))
val options = listOf("1080p", "4K")
Row(
modifier = Modifier.fillMaxWidth(),
horizontalArrangement = Arrangement.spacedBy(12.dp)
) {
options.forEach { opt ->
Surface(
shape = RoundedCornerShape(8.dp),
color = if (state.targetResolution == opt) NeonCyan.copy(alpha = 0.2f) else DarkBackground,
border = androidx.compose.foundation.BorderStroke(
1.dp,
if (state.targetResolution == opt) NeonCyan else BorderColor
),
modifier = Modifier
.weight(1f)
.selectable(
selected = state.targetResolution == opt,
onClick = { viewModel.setTargetResolution(opt) },
role = Role.RadioButton
)
) {
Row(
modifier = Modifier.padding(12.dp),
verticalAlignment = Alignment.CenterVertically,
horizontalArrangement = Arrangement.Center
) {
RadioButton(
selected = state.targetResolution == opt,
onClick = null,
colors = RadioButtonDefaults.colors(selectedColor = NeonCyan)
)
Spacer(modifier = Modifier.width(6.dp))
Text(
text = opt,
fontSize = 13.sp,
fontWeight = FontWeight.SemiBold,
color = TextPrimary
)
}
}
}
}
}
Row(
modifier = Modifier.fillMaxWidth(),
horizontalArrangement = Arrangement.SpaceBetween,
verticalAlignment = Alignment.CenterVertically
) {
Column(modifier = Modifier.weight(1f)) {
Text(text = "Быстрый GPU Lanczos-фолбэк", fontSize = 13.sp, color = TextPrimary)
Text(
text = "Автопереключение при нехватке времени (лимит 30с)",
fontSize = 11.sp,
color = TextSecondary
)
}
Switch(
checked = state.enableLanczosFallback,
onCheckedChange = { viewModel.setLanczosFallback(it) },
colors = SwitchDefaults.colors(
checkedThumbColor = DarkBackground,
checkedTrackColor = NeonCyan,
uncheckedTrackColor = BorderColor
)
)
}
}
}
}
}
@Composable
fun ZoomableComparisonCard(
bitmap512: Bitmap?,
bitmap4K: Bitmap?,
splitPos: Float,
onSplitChange: (Float) -> Unit
) {
var scale by remember { mutableFloatStateOf(1f) }
var offset by remember { mutableStateOf(Offset.Zero) }
Surface(
shape = RoundedCornerShape(16.dp),
color = CardBackground,
border = androidx.compose.foundation.BorderStroke(1.dp, BorderColor),
modifier = Modifier
.fillMaxWidth()
.height(340.dp)
) {
Box(modifier = Modifier.fillMaxSize()) {
if (bitmap512 == null && bitmap4K == null) {
Box(
modifier = Modifier
.fillMaxSize()
.background(
Brush.radialGradient(
colors = listOf(Color(0xFF1E293B), DarkBackground),
radius = 400f
)
),
contentAlignment = Alignment.Center
) {
Column(
horizontalAlignment = Alignment.CenterHorizontally,
verticalArrangement = Arrangement.spacedBy(8.dp)
) {
Box(
modifier = Modifier
.size(56.dp)
.clip(CircleShape)
.background(BorderColor.copy(alpha = 0.5f)),
contentAlignment = Alignment.Center
) {
Icon(
imageVector = Icons.Default.AutoAwesome,
contentDescription = null,
tint = NeonCyan,
modifier = Modifier.size(28.dp)
)
}
Text(
text = "Готов к генерации",
style = MaterialTheme.typography.titleSmall,
color = TextPrimary,
fontWeight = FontWeight.SemiBold
)
Text(
text = "Введите промпт и запустите пайплайн",
fontSize = 12.sp,
color = TextSecondary
)
}
}
} else {
val finalBmp = bitmap4K ?: bitmap512
val origBmp = bitmap512 ?: bitmap4K
Box(
modifier = Modifier
.fillMaxSize()
.clipToBounds()
.pointerInput(Unit) {
detectTransformGestures { _, pan, zoom, _ ->
scale = (scale * zoom).coerceIn(1f, 4f)
val maxOffsetX = (size.width * (scale - 1)) / 2f
val maxOffsetY = (size.height * (scale - 1)) / 2f
offset = Offset(
x = (offset.x + pan.x).coerceIn(-maxOffsetX, maxOffsetX),
y = (offset.y + pan.y).coerceIn(-maxOffsetY, maxOffsetY)
)
}
}
.graphicsLayer {
scaleX = scale
scaleY = scale
translationX = offset.x
translationY = offset.y
}
) {
if (finalBmp != null) {
Image(
bitmap = finalBmp.asImageBitmap(),
contentDescription = null,
contentScale = ContentScale.Crop,
modifier = Modifier.fillMaxSize()
)
}
if (origBmp != null && bitmap4K != null) {
Image(
bitmap = origBmp.asImageBitmap(),
contentDescription = null,
contentScale = ContentScale.Crop,
modifier = Modifier
.fillMaxSize()
.clip(LeftSplitShape(splitPos))
)
}
}
if (bitmap512 != null && bitmap4K != null) {
Box(
modifier = Modifier
.fillMaxHeight()
.width(2.dp)
.align(Alignment.CenterStart)
.offset(x = (340.dp * splitPos))
.background(Color.White)
)
Surface(
shape = RoundedCornerShape(12.dp),
color = DarkBackground.copy(alpha = 0.8f),
border = androidx.compose.foundation.BorderStroke(1.dp, BorderColor),
modifier = Modifier
.align(Alignment.BottomCenter)
.padding(bottom = 12.dp)
) {
Row(
verticalAlignment = Alignment.CenterVertically,
modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
) {
Icon(
imageVector = Icons.Default.Compare,
contentDescription = null,
tint = NeonCyan,
modifier = Modifier.size(16.dp)
)
Spacer(modifier = Modifier.width(6.dp))
Text(
text = "512px",
fontSize = 11.sp,
fontWeight = FontWeight.Bold,
color = NeonPurple
)
Slider(
value = splitPos,
onValueChange = onSplitChange,
valueRange = 0.05f..0.95f,
modifier = Modifier
.width(140.dp)
.height(24.dp),
colors = SliderDefaults.colors(
thumbColor = Color.White,
activeTrackColor = NeonPurple,
inactiveTrackColor = NeonCyan
)
)
Text(
text = "4K",
fontSize = 11.sp,
fontWeight = FontWeight.Bold,
color = NeonCyan
)
}
}
}
}
}
}
}
class LeftSplitShape(private val fraction: Float) : Shape {
override fun createOutline(size: Size, layoutDirection: LayoutDirection, density: Density): Outline {
return Outline.Rectangle(Rect(0f, 0f, size.width * fraction, size.height))
}
}
