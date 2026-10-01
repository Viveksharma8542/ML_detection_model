package com.pothole.detector

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/** Draws detection boxes over the camera preview (mirrors PreviewView fitCenter). */
class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var boxes: List<Detection> = emptyList()
    private var imgW = 1f
    private var imgH = 1f

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val labelBgPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
    }
    private val textPaint = Paint().apply {
        color = Color.BLACK
        textSize = 42f
    }

    /** Boxes are in original-image pixels; view maps them like PreviewView fitCenter. */
    fun setResults(detections: List<Detection>, imageWidth: Int, imageHeight: Int) {
        boxes = detections
        imgW = imageWidth.toFloat()
        imgH = imageHeight.toFloat()
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (boxes.isEmpty()) return
        val scale = minOf(width / imgW, height / imgH)
        val dx = (width - imgW * scale) / 2f
        val dy = (height - imgH * scale) / 2f
        for (d in boxes) {
            val r = RectF(
                d.box.left * scale + dx,
                d.box.top * scale + dy,
                d.box.right * scale + dx,
                d.box.bottom * scale + dy
            )
            canvas.drawRect(r, boxPaint)
            val label = "${d.label} ${"%.2f".format(d.conf)}"
            val dist = "${"%.1f".format(d.distanceM)}m"
            val tw = maxOf(textPaint.measureText(label), textPaint.measureText(dist))
            canvas.drawRect(r.left, r.top - 100f, r.left + tw + 16f, r.top, labelBgPaint)
            canvas.drawText(label, r.left + 8f, r.top - 58f, textPaint)
            canvas.drawText(dist, r.left + 8f, r.top - 10f, textPaint)
        }
    }
}
