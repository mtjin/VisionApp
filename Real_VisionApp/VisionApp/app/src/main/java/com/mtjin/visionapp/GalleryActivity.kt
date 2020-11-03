package com.mtjin.visionapp

import android.app.Activity
import android.content.Intent
import android.database.Cursor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
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
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.drawable.toBitmap
import androidx.core.view.isVisible
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.mtjin.library.DrawView
import com.mtjin.visionapp.api.ApiClient
import com.mtjin.visionapp.api.ApiInterface
import com.shashank.sony.fancytoastlib.FancyToast
import okhttp3.MediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream


class GalleryActivity : AppCompatActivity() {
    private lateinit var imageView: ImageView
    private lateinit var backDrawImageView: DrawView
    private lateinit var foreDrawImageView: DrawView
    private lateinit var loadImageButton: Button
    private lateinit var indicateButton: Button
    private lateinit var sendButton: Button
    private lateinit var fab: FloatingActionButton
    private lateinit var backDrawFab: FloatingActionButton
    private lateinit var foreDrawFab: FloatingActionButton
    private lateinit var undoFab: FloatingActionButton
    private lateinit var clearFab: FloatingActionButton
    private lateinit var saveFab: FloatingActionButton
    private lateinit var cameraFab: FloatingActionButton
    private var imageUri: Uri? = null
    private var file: File? = null

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
        imageView = findViewById(R.id.iv_image)
        backDrawImageView = findViewById(R.id.dv_draw_back)
        backDrawImageView.setStrokeWidth(30f)
        foreDrawImageView = findViewById(R.id.dv_draw_fore)
        foreDrawImageView.setStrokeWidth(30f)
        foreDrawImageView.setPenColor(Color.parseColor("#F70000"))
        loadImageButton = findViewById(R.id.gallery_btn_load)
        indicateButton = findViewById(R.id.gallery_btn_indicate)
        sendButton = findViewById(R.id.gallery_btn_send)
        fab = findViewById(R.id.fab)
        backDrawFab = findViewById(R.id.fab2_back_draw)
        undoFab = findViewById(R.id.fab3_undo)
        clearFab = findViewById(R.id.fab4_clear)
        saveFab = findViewById(R.id.fab5_save)
        cameraFab = findViewById(R.id.fab6_camera)
        foreDrawFab = findViewById(R.id.fab7_fore_draw)

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
        if (requestCode == PICK_IMAGE && resultCode == Activity.RESULT_OK && data != null && data.data != null) {
            var resultUri = data.data!!
            val inputStream: InputStream? = contentResolver.openInputStream(data.data!!)
            val img: Bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            //전송할 파일 등록
            var cursor: Cursor? = null
            try {
                val proj =
                    arrayOf(MediaStore.Images.Media.DATA)
                cursor = contentResolver.query(resultUri, proj, null, null, null)
                if (BuildConfig.DEBUG && cursor == null) {
                    error("Assertion failed")
                }
                val column_index: Int = cursor!!.getColumnIndexOrThrow(MediaStore.Images.Media.DATA)
                cursor.moveToFirst()
                file = File(cursor.getString(column_index))
            } finally {
                cursor?.close()
            }
            //초기화
            imageView.alpha = 1f
            imageView.setImageDrawable(null)
            backDrawImageView.clear()
            //이미지뷰에 세팅
            imageUri = resultUri
            imageView.setImageURI(imageUri)
            // 캔버스와 크기 맞춰줌 및 초기화
            backDrawImageView.layoutParams.width = imageView.width
            backDrawImageView.layoutParams.height = imageView.height
            foreDrawImageView.layoutParams.width = imageView.width
            foreDrawImageView.layoutParams.height = imageView.height
        }
    }

    private fun anim() {
        if (isFabOpen) {
            backDrawFab.startAnimation(fabCloseAnim)
            undoFab.startAnimation(fabCloseAnim)
            clearFab.startAnimation(fabCloseAnim)
            saveFab.startAnimation(fabCloseAnim)
            cameraFab.startAnimation(fabCloseAnim)
            foreDrawFab.startAnimation(fabCloseAnim)
            backDrawFab.isClickable = false
            undoFab.isClickable = false
            clearFab.isClickable = false
            saveFab.isClickable = false
            cameraFab.isClickable = false
            foreDrawFab.isClickable = false
            isFabOpen = false
        } else {
            backDrawFab.startAnimation(fabOpenAnim)
            undoFab.startAnimation(fabOpenAnim)
            clearFab.startAnimation(fabOpenAnim)
            saveFab.startAnimation(fabOpenAnim)
            cameraFab.startAnimation(fabOpenAnim)
            foreDrawFab.startAnimation(fabOpenAnim)
            backDrawFab.isClickable = true
            undoFab.isClickable = true
            clearFab.isClickable = true
            saveFab.isClickable = true
            cameraFab.isClickable = true
            cameraFab.isClickable = true
            foreDrawFab.isClickable = true
            isFabOpen = true
        }
    }

    // + 메뉴 Fab 버튼 열기
    fun openMenu(view: View) {
        anim()
    }

    // 그린거 되돌리기
    fun undoDraw(view: View) {
        FancyToast.makeText(
            this,
            "그리기 되돌리기",
            FancyToast.LENGTH_LONG,
            FancyToast.INFO,
            true
        ).show()
        backDrawImageView.undo()
    }

    // 마킹 그리기
    fun drawBackLine(view: View) {
        FancyToast.makeText(
            this,
            "강조할 부분에 점을 찍어주세요 :)",
            FancyToast.LENGTH_LONG,
            FancyToast.INFO,
            true
        ).show()
        foreDrawImageView.isClickable = false
        foreDrawImageView.isEnabled = false
        foreDrawImageView.isVisible = false
        backDrawImageView.isClickable = true
        backDrawImageView.isEnabled = true
        backDrawImageView.isVisible = true
        imageView.alpha = 0.25f
        // 캔버스와 크기 맞춰줌 및 초기화
        backDrawImageView.layoutParams.width = imageView.width
        backDrawImageView.layoutParams.height = imageView.height
        anim()
    }

    // 마킹 그리기
    fun drawForeLine(view: View) {
        FancyToast.makeText(
            this,
            "강조하지 않을 부분에 점을 찍어주세요 :)",
            FancyToast.LENGTH_LONG,
            FancyToast.INFO,
            true
        ).show()
        foreDrawImageView.isClickable = true
        foreDrawImageView.isVisible = true
        foreDrawImageView.isEnabled = true
        backDrawImageView.isClickable = false
        backDrawImageView.isVisible = false
        backDrawImageView.isEnabled = false
        imageView.alpha = 0.25f
        // 캔버스와 크기 맞춰줌 및 초기화
        foreDrawImageView.layoutParams.width = imageView.width
        foreDrawImageView.layoutParams.height = imageView.height
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
        backDrawImageView.clear()
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
        bitmap.compress(Bitmap.CompressFormat.JPEG, 70, outStream)
        outStream.flush()
        outStream.close()
        FancyToast.makeText(
            this,
            "이미지 저장",
            FancyToast.LENGTH_LONG,
            FancyToast.INFO,
            true
        ).show()
    }

    // 사진 서버에 전송
    fun requestDrawImage(view: View) {
        if (backDrawImageView.getPointList() == null || backDrawImageView.getPointList()!!
                .isEmpty()
        ) {
            FancyToast.makeText(
                this,
                "선을 먼저 그려주세요",
                FancyToast.LENGTH_LONG,
                FancyToast.WARNING,
                true
            ).show()
            return
        }
        val xList = ArrayList<Float>() //background point
        val yList = ArrayList<Float>()
        val nxList = ArrayList<Float>() //foreground
        val nyList = ArrayList<Float>()
        while (backDrawImageView.getPointList()!!.isNotEmpty()) {
            with(backDrawImageView.getPointList()) {
                val pointList = this?.pop()
                for (point in pointList!!) {
                    xList.add(point.x)
                    yList.add(point.y)
                }
            }
        }
        foreDrawImageView.isVisible = true
        backDrawImageView.isVisible = true
        while (foreDrawImageView.getPointList()!!.isNotEmpty()) {
            with(foreDrawImageView.getPointList()) {
                val pointList = this?.pop()
                for (point in pointList!!) {
                    nxList.add(point.x)
                    nyList.add(point.y)
                }
            }
        }
        val apiInteface = ApiClient.getApiClient().create(ApiInterface::class.java)
        //파일 세팅
        val requestBody = RequestBody.create(MediaType.parse("image/jpeg"), file)
        val body: MultipartBody.Part =
            MultipartBody.Part.createFormData("image", file?.name, requestBody)
        apiInteface.getTest(body, xList, yList, nxList, nyList)
            .enqueue(object : Callback<ResponseBody> {
                override fun onFailure(call: Call<ResponseBody>, t: Throwable) {
                    Log.d("AAA", "FAIL REQUEST ==> " + t.localizedMessage)
                    foreDrawImageView.isVisible = false
                    backDrawImageView.isVisible = false
                    backDrawImageView.clear()
                }

                override fun onResponse(
                    call: Call<ResponseBody>,
                    response: Response<ResponseBody>
                ) {
                    Log.d("AAA", "REQUEST SUCCESS ==> ")
                    val file = response.body()?.byteStream()
                    val bitmap = BitmapFactory.decodeStream(file)
                    imageView.setImageURI(null)
                    imageView.setImageBitmap(bitmap)
                    foreDrawImageView.isVisible = false
                    backDrawImageView.isVisible = false
                    backDrawImageView.clear()
                }
            })
    }

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
        intent.setDataAndType(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            "image/*"
        )
        startActivityForResult(intent, PICK_IMAGE)
    }

    companion object {
        const val PICK_IMAGE = 1
    }
}
