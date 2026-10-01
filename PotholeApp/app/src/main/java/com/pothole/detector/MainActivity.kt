package com.pothole.detector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.Priority
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var statsText: TextView
    private lateinit var cameraExecutor: ExecutorService
    private var detector: Detector? = null
    private val analyzing = AtomicBoolean(false)

    private var frames = 0
    private var lastFpsTime = System.currentTimeMillis()
    private var lastInferMs = 0L
    private var lastLat: Double? = null
    private var lastLon: Double? = null

    // Temporal filter: shake gives 1-frame false boxes, real potholes persist.
    private var hitStreak = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        statsText = findViewById(R.id.statsText)
        cameraExecutor = Executors.newSingleThreadExecutor()

        val perms = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.ACCESS_FINE_LOCATION
        )
        if (perms.all {
                ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
            }
        ) {
            startApp()
        } else {
            ActivityCompat.requestPermissions(this, perms, 10)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 10 && grantResults.isNotEmpty()
            && grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            startApp() // GPS optional: works without it, coords just show "no fix"
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    private fun startApp() {
        startGps()
        // Load model off the UI thread (ORT init takes ~1-2 s)
        Thread {
            try {
                detector = Detector(this)
                runOnUiThread { startCamera() }
            } catch (e: Exception) {
                runOnUiThread {
                    Toast.makeText(this, "Model load failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setTargetResolution(android.util.Size(640, 480))
                .build()
            analysis.setAnalyzer(cameraExecutor) { image ->
                try {
                    if (analyzing.compareAndSet(false, true)) {
                        try {
                            val t0 = System.currentTimeMillis()
                            val bmp = rgbaToBitmap(image)
                            val raw = detector?.detect(bmp) ?: emptyList()
                            // Reject tiny / sky boxes (shake-blur noise), keep road boxes
                            val imgArea = bmp.width * bmp.height.toFloat()
                            val dets = raw.filter {
                                val area = it.box.width() * it.box.height()
                                area >= imgArea * 0.003f &&
                                        it.box.bottom / bmp.height >= 0.35f
                            }
                            // Require 2 consecutive frames before alerting
                            if (dets.isNotEmpty()) hitStreak++ else hitStreak = 0
                            val stable = hitStreak >= 2
                            lastInferMs = System.currentTimeMillis() - t0
                            frames++
                            val now = System.currentTimeMillis()
                            runOnUiThread {
                                overlayView.setResults(dets, bmp.width, bmp.height)
                                if (now - lastFpsTime >= 500) {
                                    val fps = frames * 1000.0 / (now - lastFpsTime)
                                    val nearest = dets.minByOrNull { it.distanceM }
                                    val alert = if (stable && nearest != null)
                                        "  POTHOLE! ${"%.1f".format(nearest.distanceM)}m" else ""
                                    val gps = if (lastLat != null)
                                        "  GPS: ${"%.5f".format(lastLat)},${"%.5f".format(lastLon)}"
                                    else "  GPS: no fix"
                                    statsText.text =
                                        "INF: ${lastInferMs}ms  FPS: ${"%.1f".format(fps)}$alert$gps"
                                    frames = 0
                                    lastFpsTime = now
                                }
                            }
                        } finally {
                            analyzing.set(false)
                        }
                    }
                } finally {
                    image.close()
                }
            }
            provider.unbindAll()
            provider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun rgbaToBitmap(image: androidx.camera.core.ImageProxy): Bitmap {
        val plane = image.planes[0]
        val buffer = plane.buffer
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val w = image.width
        val h = image.height
        // Repack rows (rowStride may exceed w*pixelStride)
        val rowBytes = w * 4
        val tmp = ByteArray(rowBytes * h)
        val row = ByteArray(plane.rowStride)
        buffer.rewind()
        for (y in 0 until h) {
            buffer.get(row, 0, rowStride.coerceAtMost(row.size))
            System.arraycopy(row, 0, tmp, y * rowBytes, rowBytes)
        }
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.copyPixelsFromBuffer(java.nio.ByteBuffer.wrap(tmp))
        return bmp
    }

    private fun startGps() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED
        ) return // GPS optional
        try {
            val client = LocationServices.getFusedLocationProviderClient(this)
            val req = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 2000L).build()
            val cb = object : LocationCallback() {
                override fun onLocationResult(r: LocationResult) {
                    r.lastLocation?.let {
                        lastLat = it.latitude
                        lastLon = it.longitude
                    }
                }
            }
            client.requestLocationUpdates(req, cb, mainLooper)
        } catch (_: Exception) { /* GPS unavailable: coords stay "no fix" */ }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        detector?.close()
    }
}
