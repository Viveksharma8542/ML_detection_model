package com.pothole.detector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/** One detection: box in ORIGINAL image pixels + confidence + distance. */
data class Detection(
    val box: RectF,
    val conf: Float,
    val label: String,
    val distanceM: Float
)

class Detector(context: Context) {

    companion object {
        const val INPUT_SIZE = 160
        const val CONF_THRESHOLD = 0.45f   // 0.6 missed real potholes on bike; 0.45 = balance recall vs false alarms
        const val IOU_THRESHOLD = 0.45f
        const val MAX_DET = 10
        private const val LABEL = "Pothole"

        // ---- Distance calibration (ground-plane linear model) ----
        // distance = DIST_SLOPE * yBottomNorm + DIST_INTERCEPT,
        // where yBottomNorm = box bottom (0 = top of image, 1 = bottom).
        // Defaults: y=0.9 -> ~3 m, y=0.6 -> ~10 m (phone ~1.2 m high, slight tilt).
        // Recalibrate for your mount: film markers at 3 m and 8 m, read the
        // normalized bottom-y of each box, solve the two equations, update below.
        const val DIST_SLOPE = -23.3f
        const val DIST_INTERCEPT = 24.0f
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    // Reused per-frame buffers: avoids GC pauses that cause 70->130ms fluctuation
    private val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
    private val chR = FloatArray(INPUT_SIZE * INPUT_SIZE)
    private val chG = FloatArray(INPUT_SIZE * INPUT_SIZE)
    private val chB = FloatArray(INPUT_SIZE * INPUT_SIZE)
    private val buf: FloatBuffer = ByteBuffer
        .allocateDirect(3 * INPUT_SIZE * INPUT_SIZE * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()

    init {
        val modelBytes = context.assets.open("best.onnx").readBytes()
        val opts = OrtSession.SessionOptions()
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
        opts.setIntraOpNumThreads(threads)
        opts.setInterOpNumThreads(1)
        // XNNPACK only. Do NOT add NNAPI for YOLO: unsupported ops cause
        // CPU<->NPU transfers per node -> 70ms becomes 1000ms+ on many phones.
        try {
            opts.addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
        } catch (_: Exception) {}
        session = env.createSession(modelBytes, opts)
    }

    /** Run detection on an ARGB bitmap. Returns boxes in that bitmap's pixel space. */
    fun detect(bitmap: Bitmap): List<Detection> {
        val srcW = bitmap.width
        val srcH = bitmap.height

        // ---- Letterbox to 160x160 (same as laptop pipeline) ----
        val scale = min(INPUT_SIZE / srcW.toFloat(), INPUT_SIZE / srcH.toFloat())
        val newW = (srcW * scale).toInt()
        val newH = (srcH * scale).toInt()
        val padX = (INPUT_SIZE - newW) / 2f
        val padY = (INPUT_SIZE - newH) / 2f
        val resized = Bitmap.createScaledBitmap(bitmap, newW, newH, false)
        val letterboxed = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(letterboxed)
        canvas.drawColor(android.graphics.Color.rgb(114, 114, 114))
        canvas.drawBitmap(resized, padX, padY, null)
        resized.recycle()

        // ---- ARGB -> float32 CHW RGB / 255 (reuses buffers, no per-frame alloc) ----
        letterboxed.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        letterboxed.recycle()
        for (i in pixels.indices) {
            val p = pixels[i]
            chR[i] = ((p shr 16) and 0xFF) / 255f
            chG[i] = ((p shr 8) and 0xFF) / 255f
            chB[i] = (p and 0xFF) / 255f
        }
        buf.clear()
        buf.put(chR); buf.put(chG); buf.put(chB)
        buf.rewind()

        // ---- Inference ----
        val inputTensor = OnnxTensor.createTensor(
            env, buf, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )
        inputTensor.use {
            val result = session.run(mapOf("images" to it))
            result.use { res ->
                @Suppress("UNCHECKED_CAST")
                val out = (res[0].value as Array<Array<FloatArray>>)[0] // [5][525]
                return parseOutput(out, scale, padX, padY, srcW, srcH)
            }
        }
    }

    /** [5][525] (cx, cy, w, h in 160-space + conf) -> NMS -> original-image boxes. */
    private fun parseOutput(
        out: Array<FloatArray>,
        scale: Float, padX: Float, padY: Float,
        srcW: Int, srcH: Int
    ): List<Detection> {
        val n = out[0].size
        val cands = ArrayList<Raw>(32)
        for (i in 0 until n) {
            val conf = out[4][i]
            if (conf < CONF_THRESHOLD) continue
            // Undo letterbox -> original image pixels
            val x1 = (out[0][i] - out[2][i] / 2f - padX) / scale
            val y1 = (out[1][i] - out[3][i] / 2f - padY) / scale
            val x2 = (out[0][i] + out[2][i] / 2f - padX) / scale
            val y2 = (out[1][i] + out[3][i] / 2f - padY) / scale
            cands.add(Raw(
                x1.coerceIn(0f, srcW.toFloat()),
                y1.coerceIn(0f, srcH.toFloat()),
                x2.coerceIn(0f, srcW.toFloat()),
                y2.coerceIn(0f, srcH.toFloat()),
                conf
            ))
        }
        // NMS (single class)
        cands.sortByDescending { it.conf }
        val kept = ArrayList<Raw>(MAX_DET)
        for (c in cands) {
            if (kept.size >= MAX_DET) break
            var keep = true
            for (k in kept) {
                if (iou(c, k) > IOU_THRESHOLD) { keep = false; break }
            }
            if (keep) kept.add(c)
        }
        return kept.map {
            val yBottomNorm = (it.y2 / srcH.toFloat()).coerceIn(0f, 1f)
            val dist = (DIST_SLOPE * yBottomNorm + DIST_INTERCEPT).coerceIn(0.5f, 60f)
            Detection(RectF(it.x1, it.y1, it.x2, it.y2), it.conf, LABEL, dist)
        }
    }

    private data class Raw(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val conf: Float)

    private fun iou(a: Raw, b: Raw): Float {
        val ix1 = max(a.x1, b.x1); val iy1 = max(a.y1, b.y1)
        val ix2 = min(a.x2, b.x2); val iy2 = min(a.y2, b.y2)
        val inter = max(0f, ix2 - ix1) * max(0f, iy2 - iy1)
        if (inter <= 0f) return 0f
        val union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return if (union <= 0f) 0f else inter / union
    }

    fun close() {
        try { session.close() } catch (_: Exception) {}
    }
}
