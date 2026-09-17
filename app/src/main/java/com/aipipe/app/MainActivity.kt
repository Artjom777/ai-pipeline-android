package com.aipipe.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import com.aipipe.app.ui.PipelineScreen
import com.aipipe.app.ui.theme.AIPipelineTheme

class MainActivity : ComponentActivity() {

    private val viewModel: PipelineViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AIPipelineTheme {
                PipelineScreen(viewModel = viewModel)
            }
        }
    }
}
