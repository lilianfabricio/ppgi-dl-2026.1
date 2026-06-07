package com.example.petsegmentation

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import com.example.petsegmentation.databinding.ActivityMainBinding
import java.io.File
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var segmentationModel: SegmentationModel
    private var selectedBitmap: Bitmap? = null
    private var cameraUri: Uri? = null
    private val executor = Executors.newSingleThreadExecutor()

    private val galleryLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { loadBitmapFromUri(it) }
    }

    private val cameraLauncher = registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) {
            cameraUri?.let { loadBitmapFromUri(it) }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.statusText.text = getString(R.string.status_loading_model)
        executor.execute {
            try {
                segmentationModel = SegmentationModel(this)
                runOnUiThread {
                    binding.statusText.text = getString(R.string.status_idle)
                }
            } catch (error: Exception) {
                runOnUiThread {
                    binding.statusText.text = getString(R.string.status_error, error.message ?: "modelo")
                    Toast.makeText(this, R.string.model_missing, Toast.LENGTH_LONG).show()
                }
            }
        }

        binding.galleryButton.setOnClickListener {
            galleryLauncher.launch("image/*")
        }

        binding.cameraButton.setOnClickListener {
            val photoFile = File(cacheDir, "capture.jpg")
            cameraUri = FileProvider.getUriForFile(
                this,
                "${applicationContext.packageName}.fileprovider",
                photoFile
            )
            cameraLauncher.launch(cameraUri)
        }

        binding.inferButton.setOnClickListener {
            val bitmap = selectedBitmap ?: return@setOnClickListener
            if (!::segmentationModel.isInitialized) {
                Toast.makeText(this, R.string.model_missing, Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            binding.inferButton.isEnabled = false
            binding.statusText.text = getString(R.string.status_running)

            executor.execute {
                try {
                    val overlay = segmentationModel.segment(bitmap)
                    runOnUiThread {
                        binding.segmentedImageView.setImageBitmap(overlay)
                        binding.statusText.text = getString(R.string.status_done)
                        binding.inferButton.isEnabled = true
                    }
                } catch (error: Exception) {
                    runOnUiThread {
                        binding.statusText.text = getString(R.string.status_error, error.message ?: "inferência")
                        binding.inferButton.isEnabled = true
                    }
                }
            }
        }
    }

    private fun loadBitmapFromUri(uri: Uri) {
        contentResolver.openInputStream(uri)?.use { stream ->
            val bitmap = BitmapFactory.decodeStream(stream) ?: return
            selectedBitmap = bitmap
            binding.originalImageView.setImageBitmap(bitmap)
            binding.segmentedImageView.setImageDrawable(null)
            binding.inferButton.isEnabled = ::segmentationModel.isInitialized
        }
    }

    override fun onDestroy() {
        executor.shutdownNow()
        super.onDestroy()
    }
}
