package com.mtjin.visionapp.api

import okhttp3.MultipartBody
import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.PartMap

interface ApiInterface {
    @Multipart
    @POST("/predict")
    fun getTest(
        @Part file: MultipartBody.Part,
        @Part("x") x: ArrayList<Float>,
        @Part("y") y: ArrayList<Float>
    ): Call<ResponseBody>

//    @FormUrlEncoded
//    @POST("/predict")
//    fun getTest2(
//        @Body point: Point
//    ): Call<ResponseBody>
//
//    @FormUrlEncoded
//    @POST("/predict")
//    fun getTest3(
//        @Field("TEST") test : String
//    ): Call<ResponseBody>
}