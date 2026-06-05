package com.example.pathguidance

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var instructionView: TextView
    private lateinit var reasonView: TextView
    private lateinit var statusView: TextView
    private lateinit var analyzeButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var guidanceClient: GuidanceClient
    private lateinit var speechManager: SpeechManager

    private var imageCapture: ImageCapture? = null

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                updateStatus("Camera permission is required.")
                speak("Camera permission is required.")
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        guidanceClient = GuidanceClient(getString(R.string.default_backend_url))
        speechManager = SpeechManager(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        setContentView(createContentView())
        analyzeButton.setOnClickListener { captureAndAnalyze() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun createContentView(): View {
        val root = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
        }

        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
        root.addView(
            previewView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            ),
        )

        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(32, 28, 32, 40)
            setBackgroundColor(Color.argb(230, 17, 24, 39))
        }

        statusView = TextView(this).apply {
            text = "Ready"
            setTextColor(Color.WHITE)
            textSize = 18f
            gravity = Gravity.CENTER
        }

        instructionView = TextView(this).apply {
            text = "Point camera toward the path"
            setTextColor(Color.WHITE)
            textSize = 26f
            gravity = Gravity.CENTER
        }

        reasonView = TextView(this).apply {
            text = ""
            setTextColor(Color.rgb(209, 213, 219))
            textSize = 16f
            gravity = Gravity.CENTER
            maxLines = 3
        }

        progressBar = ProgressBar(this).apply {
            visibility = View.GONE
        }

        analyzeButton = Button(this).apply {
            text = "Analyze"
            textSize = 22f
            minHeight = 128
        }

        panel.addView(statusView, panelParams())
        panel.addView(instructionView, panelParams(topMargin = 16))
        panel.addView(reasonView, panelParams(topMargin = 8))
        panel.addView(progressBar, panelParams(topMargin = 16))
        panel.addView(analyzeButton, panelParams(topMargin = 20))

        val panelLayoutParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM,
        )
        root.addView(panel, panelLayoutParams)

        return root
    }

    private fun panelParams(topMargin: Int = 0): LinearLayout.LayoutParams {
        return LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT,
        ).apply {
            this.topMargin = topMargin
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(
            {
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()

                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        this,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageCapture,
                    )
                    updateStatus("Ready")
                } catch (error: Exception) {
                    updateStatus("Could not start camera.")
                    speak("Could not start camera.")
                }
            },
            ContextCompat.getMainExecutor(this),
        )
    }

    private fun captureAndAnalyze() {
        val capture = imageCapture ?: return
        setAnalyzing(true)
        updateStatus("Analyzing")
        speak("Analyzing")

        val photoFile = File(cacheDir, "path-guidance-frame.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        capture.takePicture(
            outputOptions,
            cameraExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exception: ImageCaptureException) {
                    runOnUiThread {
                        setAnalyzing(false)
                        updateStatus("Could not capture image.")
                        speak("Could not capture image.")
                    }
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    analyzeImage(photoFile)
                }
            },
        )
    }

    private fun analyzeImage(photoFile: File) {
        cameraExecutor.execute {
            try {
                val result = guidanceClient.analyze(photoFile)
                runOnUiThread {
                    instructionView.text = result.spokenInstruction
                    reasonView.text = result.reason
                    updateStatus(result.direction)
                    setAnalyzing(false)
                    speak(result.spokenInstruction)
                }
            } catch (error: Exception) {
                runOnUiThread {
                    instructionView.text = "Unable to analyze image"
                    reasonView.text = error.message.orEmpty()
                    updateStatus("Error")
                    setAnalyzing(false)
                    speak("Unable to analyze image.")
                }
            }
        }
    }

    private fun setAnalyzing(analyzing: Boolean) {
        analyzeButton.isEnabled = !analyzing
        progressBar.visibility = if (analyzing) View.VISIBLE else View.GONE
    }

    private fun updateStatus(text: String) {
        statusView.text = text
    }

    private fun speak(text: String) {
        speechManager.speak(text)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        speechManager.shutdown()
    }
}
