package com.mtjin.visionapp.customview

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Point
import android.view.MotionEvent
import android.view.View
import android.view.View.OnTouchListener


class DrawView(context: Context?) : View(context),
    OnTouchListener {
    var points: MutableList<Point> = ArrayList()
    var paint = Paint()
    override fun onDraw(canvas: Canvas) {
        for (point in points) {
            canvas.drawCircle(point.x.toFloat(), point.y.toFloat(), 2F, paint)
        }
    }

    override fun onTouch(view: View, event: MotionEvent): Boolean {
        val point = Point()
        point.x = event.x.toInt()
        point.y = event.y.toInt()
        points.add(point)
        invalidate()
        return true
    }

    init {
        isFocusable = true
        isFocusableInTouchMode = true
        setOnTouchListener(this)
        paint.color = Color.BLACK
    }
}

internal class Point {
    var x = 0f
    var y = 0f
}