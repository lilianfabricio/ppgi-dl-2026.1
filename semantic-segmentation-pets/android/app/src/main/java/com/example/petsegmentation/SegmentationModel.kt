package com.example.petsegmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream

class SegmentationModel(context: Context) {

    companion object {
        private const val MODEL_NAME = "model.pt"
        private const val INPUT_SIZE = 256
        private const val NUM_CLASSES = 2

        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val module: Module = LiteModuleLoader.load(assetFilePath(context, MODEL_NAME))

    fun segment(source: Bitmap): Bitmap {
        val resized = Bitmap.createScaledBitmap(source, INPUT_SIZE, INPUT_SIZE, true)
        val input = bitmapToFloatBuffer(resized)
        val inputTensor = Tensor.fromBlob(input, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))
        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
        val mask = argmaxMask(outputTensor)
        return overlayMask(source, mask)
    }

    private fun bitmapToFloatBuffer(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val buffer = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val pixel = pixels[y * INPUT_SIZE + x]
                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8) and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f

                val index = y * INPUT_SIZE + x
                buffer[index] = (r - MEAN[0]) / STD[0]
                buffer[INPUT_SIZE * INPUT_SIZE + index] = (g - MEAN[1]) / STD[1]
                buffer[2 * INPUT_SIZE * INPUT_SIZE + index] = (b - MEAN[2]) / STD[2]
            }
        }
        return buffer
    }

    private fun argmaxMask(output: Tensor): BooleanArray {
        val data = output.dataAsFloatArray
        val mask = BooleanArray(INPUT_SIZE * INPUT_SIZE)
        val planeSize = INPUT_SIZE * INPUT_SIZE

        for (index in 0 until planeSize) {
            val background = data[index]
            val pet = data[planeSize + index]
            mask[index] = pet > background
        }
        return mask
    }

    private fun overlayMask(source: Bitmap, mask: BooleanArray): Bitmap {
        val output = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        val paint = Paint().apply {
            color = Color.argb(115, 255, 64, 64)
            style = Paint.Style.FILL
        }

        val scaleX = source.width.toFloat() / INPUT_SIZE
        val scaleY = source.height.toFloat() / INPUT_SIZE

        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                if (mask[y * INPUT_SIZE + x]) {
                    canvas.drawRect(
                        x * scaleX,
                        y * scaleY,
                        (x + 1) * scaleX,
                        (y + 1) * scaleY,
                        paint
                    )
                }
            }
        }
        return output
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }
        return file.absolutePath
    }
}
