package com.aipipe.app.ui.theme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
private val DarkColorScheme = darkColorScheme(
primary = NeonCyan,
secondary = NeonPurple,
tertiary = MaliOrange,
background = DarkBackground,
surface = CardBackground,
onPrimary = DarkBackground,
onSecondary = TextPrimary,
onBackground = TextPrimary,
onSurface = TextPrimary
)
@Composable
fun AIPipelineTheme(content: @Composable () -> Unit) {
MaterialTheme(
colorScheme = DarkColorScheme,
content = content
)
}
