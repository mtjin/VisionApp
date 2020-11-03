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
        @Part file: MultipartBody.Part,
        @Part("x") x: ArrayList<Float>,
        @Part("y") y: ArrayList<Float>,
        @Part("nx") nx: ArrayList<Float>,
        @Part("ny") ny: ArrayList<Float>
    ): Call<ResponseBody>

}