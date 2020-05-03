package com.mtjin.visionapp

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.view.animation.Animation
import android.view.animation.AnimationUtils
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.drawable.toBitmap
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.mtjin.library.DrawView
import com.shashank.sony.fancytoastlib.FancyToast
import com.yalantis.ucrop.UCrop
import java.io.File
import java.io.FileOutputStream


class GalleryActivity : AppCompatActivity() {
    private lateinit var imageView: ImageView
    private lateinit var drawImageView: DrawView
    private lateinit var loadImageButton: Button
    private lateinit var indicateButton: Button
    private lateinit var sendButton: Button
    private lateinit var fab: FloatingActionButton
    private lateinit var drawFab: FloatingActionButton
    private lateinit var undoFab: FloatingActionButton
    private lateinit var clearFab: FloatingActionButton
    private lateinit var saveFab: FloatingActionButton
    private lateinit var cameraFab : FloatingActionButton
    private var imageUri: Uri? = null

    private var fabOpenAnim: Animation? = null
    private var fabCloseAnim: Animation? = null
    private var isFabOpen = false


    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_gallery)
        initView()
        initListener()
    }

    private fun initListener() {

        fab.setOnClickListener {
            anim()
        }
    }

    private fun initView() {
        imageView = findViewById(R.id.gallery_iv_image)
        drawImageView = findViewById(R.id.gallery_iv_draw)
        drawImageView.setStrokeWidth(30f)
        loadImageButton = findViewById(R.id.gallery_btn_load)
        indicateButton = findViewById(R.id.gallery_btn_indicate)
        sendButton = findViewById(R.id.gallery_btn_send)
        fab = findViewById(R.id.fab)
        drawFab = findViewById(R.id.fab2_draw)
        undoFab = findViewById(R.id.fab3_undo)
        clearFab = findViewById(R.id.fab4_clear)
        saveFab = findViewById(R.id.fab5_save)
        cameraFab = findViewById(R.id.fab6_camera)

        fabOpenAnim = AnimationUtils.loadAnimation(
            applicationContext,
            R.anim.fab_open
        )
        fabCloseAnim = AnimationUtils.loadAnimation(
            applicationContext,
            R.anim.fab_close
        )
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            if (requestCode == PICK_IMAGE) {
                val sourceUri = data!!.data
                if (sourceUri != null) {
                    val destinationUri = Uri.fromFile(File(cacheDir, "cropped"))
                    openCropActivity(sourceUri, destinationUri)
                } else {
                    Toast.makeText(this, getString(R.string.get_img_error_msg), Toast.LENGTH_SHORT)
                        .show()
                }
            } else if (requestCode == UCrop.REQUEST_CROP) {
                val resultUri = UCrop.getOutput(data!!)
                if (resultUri != null) {
                    //초기화
                    imageView.alpha = 1f
                    imageView.visibility = View.VISIBLE
                    drawImageView.visibility = View.GONE
                    imageView.setImageDrawable(null)
                    drawImageView.clear()
                    //이미지뷰에 세팅
                    imageUri = resultUri
                    imageView.setImageURI(imageUri)
                    // 캔버스와 크기 맞춰줌 및 초기화
                    drawImageView.layoutParams.width = imageView.width
                    drawImageView.layoutParams.height = imageView.height
                    Log.d("AAAA " , "" +  imageView.width + "   " + imageView.height)
                } else {
                    Toast.makeText(this, getString(R.string.get_img_error_msg), Toast.LENGTH_SHORT)
                        .show()
                }
            }
        } else if (resultCode == UCrop.RESULT_ERROR) {
            Toast.makeText(this, getString(R.string.get_img_error_msg), Toast.LENGTH_SHORT).show()
        }
    }

    private fun openCropActivity(
        sourceUri: Uri,
        destinationUri: Uri
    ) {
        UCrop.of(sourceUri, destinationUri)
            .start(this)
    }

    private fun anim() {
        if (isFabOpen) {
            drawFab.startAnimation(fabCloseAnim)
            undoFab.startAnimation(fabCloseAnim)
            clearFab.startAnimation(fabCloseAnim)
            saveFab.startAnimation(fabCloseAnim)
            cameraFab.startAnimation(fabCloseAnim)
            drawFab.isClickable = false
            undoFab.isClickable = false
            clearFab.isClickable = false
            saveFab.isClickable = false
            cameraFab.isClickable = false
            isFabOpen = false
        } else {
            drawFab.startAnimation(fabOpenAnim)
            undoFab.startAnimation(fabOpenAnim)
            clearFab.startAnimation(fabOpenAnim)
            saveFab.startAnimation(fabOpenAnim)
            cameraFab.startAnimation(fabOpenAnim)
            drawFab.isClickable = true
            undoFab.isClickable = true
            clearFab.isClickable = true
            saveFab.isClickable = true
            cameraFab.isClickable = true
            isFabOpen = true
        }
    }

    companion object {
        const val PICK_IMAGE = 1
    }

    // + 메뉴 Fab 버튼 열기
    fun openMenu(view: View) {
        anim()
    }

    // 그린거 되돌리기
    fun undoDraw(view: View) {
        FancyToast.makeText(
            this,
            "그리기 이전으로 지우기",
            FancyToast.LENGTH_LONG,
            FancyToast.INFO,
            true
        ).show()
        drawImageView.undo()
    }

    // 마킹 그리기
    fun drawLine(view: View) {
        FancyToast.makeText(
            this,
            "강조할 부분에 점을 찍어주세요",
            FancyToast.LENGTH_LONG,
            FancyToast.INFO,
            true
        ).show()
        imageView.alpha = 0.25f
        drawImageView.visibility = View.VISIBLE
        // 캔버스와 크기 맞춰줌 및 초기화
        drawImageView.layoutParams.width = imageView.width
        drawImageView.layoutParams.height = imageView.height
        anim()
    }

    // 그린거 초기화
    fun clearDraw(view: View) {
        FancyToast.makeText(
            this,
            "그리기 초기화",
            FancyToast.LENGTH_LONG,
            FancyToast.INFO,
            true
        ).show()
        drawImageView.clear()
        anim()
    }

    // 세팅된 이미지 갤러리에 저장
    fun saveDrawFile(view: View) {
        val bitmap = imageView.drawable.toBitmap()
        var outStream: FileOutputStream? = null
        val sdCard: File = Environment.getExternalStorageDirectory()
        val dir = File(sdCard.absolutePath + "/VisinApp")
        dir.mkdirs()
        val fileName =
            String.format("%d.jpg", System.currentTimeMillis())
        val outFile = File(dir, fileName)
        outStream = FileOutputStream(outFile)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outStream)
        outStream.flush()
        outStream.close()
    }

    // 사진 서버에 전송
    fun requestDrawImage(view: View) {}

    // +Fab 메뉴 버튼(그리기관련) 나오게 하기
    fun showIndicateMenu(view: View) {
        if (imageUri == null) {
            FancyToast.makeText(
                this,
                "사진 먼저 세팅해주세요",
                FancyToast.LENGTH_LONG,
                FancyToast.WARNING,
                true
            ).show()
        } else {
            fab.visibility = View.VISIBLE
        }
    }

    // 갤러리에서 사진 불러오기
    fun getImageFromGallery(view: View) {
        val intent = Intent(Intent.ACTION_PICK)
        intent.type = MediaStore.Images.Media.CONTENT_TYPE
        intent.type = "image/*"
        startActivityForResult(
            intent,
            PICK_IMAGE
        )
    }
}
