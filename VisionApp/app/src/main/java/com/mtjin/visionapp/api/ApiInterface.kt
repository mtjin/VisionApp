package com.mtjin.visionapp.api

import okhttp3.MultipartBody
import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface ApiInterface {
    @Multipart
    @POST("/predict")
    fun getTest(
        @Part file: MultipartBody.Part
    ): Call<ResponseBody>
}