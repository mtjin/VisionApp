package com.mtjin.visionapp

import android.Manifest
import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.gun0912.tedpermission.PermissionListener
import com.gun0912.tedpermission.TedPermission
import com.ramotion.circlemenu.CircleMenuView
import com.shashank.sony.fancytoastlib.FancyToast
import java.util.*


class MainActivity : AppCompatActivity() {
    private lateinit var permissionListener: PermissionListener
    private lateinit var circleMenuView: CircleMenuView
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        initVIew()
        initListener()
        initPermission()
    }

    private fun initPermission() {
        permissionListener = object : PermissionListener {
            override fun onPermissionGranted() {
                FancyToast.makeText(
                    this@MainActivity,
                    getString(R.string.success_camera_auth_msg),
                    FancyToast.LENGTH_LONG,
                    FancyToast.SUCCESS,
                    true
                ).show()
            }

            override fun onPermissionDenied(deniedPermissions: ArrayList<String>?) {
                TedPermission.with(this@MainActivity)
                    .setPermissionListener(permissionListener)
                    .setRationaleMessage(getString(R.string.request_camera_auth_msg))
                    .setDeniedMessage(getString(R.string.deny_camera_auth_msg))
                    .setPermissions(
                        Manifest.permission.WRITE_EXTERNAL_STORAGE,
                        Manifest.permission.INTERNET,
                        Manifest.permission.READ_EXTERNAL_STORAGE
                    )
                    .check()
            }
        }
        TedPermission.with(this)
            .setPermissionListener(permissionListener)
            .setRationaleMessage(getString(R.string.request_permisson_auth_msg))
            .setDeniedMessage("거부하시면 사용에 지장이 있습니다.ㅠㅠ.\n[설정] > [권한] 에서 권한을 허용할 수 있어요.")
            .setPermissions( Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.INTERNET,
                Manifest.permission.READ_EXTERNAL_STORAGE)
            .check()

    }

    private fun initListener() {
        circleMenuView.eventListener = object : CircleMenuView.EventListener() {
            override fun onMenuOpenAnimationStart(view: CircleMenuView) {}
            override fun onMenuOpenAnimationEnd(view: CircleMenuView) {}
            override fun onMenuCloseAnimationStart(view: CircleMenuView) {}
            override fun onMenuCloseAnimationEnd(view: CircleMenuView) {}
            override fun onButtonClickAnimationStart(view: CircleMenuView, index: Int) {}
            override fun onButtonClickAnimationEnd(view: CircleMenuView, index: Int) {
                if (index == 0) {
                    val intent = Intent(this@MainActivity, GalleryActivity::class.java)
                    startActivity(intent)
                    overridePendingTransition(R.anim.fade_in_splash, R.anim.fade_out_splash)
                    FancyToast.makeText(
                        this@MainActivity,
                        getString(R.string.photo),
                        FancyToast.LENGTH_LONG,
                        FancyToast.SUCCESS,
                        true
                    ).show()
                } else if (index == 1) {
                    val intent = Intent(this@MainActivity, CameraActivity::class.java)
                    startActivity(intent)
                    overridePendingTransition(R.anim.fade_in_splash, R.anim.fade_out_splash)
                    FancyToast.makeText(
                        this@MainActivity,
                        getString(R.string.album),
                        FancyToast.LENGTH_LONG,
                        FancyToast.SUCCESS,
                        true
                    ).show()
                }
            }

            override fun onButtonLongClick(view: CircleMenuView, index: Int): Boolean {
                return true
            }

            override fun onButtonLongClickAnimationStart(
                view: CircleMenuView,
                index: Int
            ) {
            }

            override fun onButtonLongClickAnimationEnd(view: CircleMenuView, index: Int) {
                if (index == 0) {
                    val intent = Intent(this@MainActivity, CameraActivity::class.java)
                    startActivity(intent)
                    overridePendingTransition(R.anim.fade_in_splash, R.anim.fade_out_splash)
                    FancyToast.makeText(
                        this@MainActivity,
                        getString(R.string.photo),
                        FancyToast.LENGTH_LONG,
                        FancyToast.SUCCESS,
                        true
                    ).show()
                } else if (index == 1) {
                    val intent = Intent(this@MainActivity, GalleryActivity::class.java)
                    startActivity(intent)
                    overridePendingTransition(R.anim.fade_in_splash, R.anim.fade_out_splash)
                    FancyToast.makeText(
                        this@MainActivity,
                        getString(R.string.album),
                        FancyToast.LENGTH_LONG,
                        FancyToast.SUCCESS,
                        true
                    ).show()
                }
            }
        }

    }

    private fun initVIew() {
        circleMenuView = findViewById(R.id.main_circle_menu)
    }

}
