package com.mtjin.visionapp.api

import com.mtjin.visionapp.model.Point
import okhttp3.MultipartBody
import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.http.*

interface ApiInterface {
    @Multipart
    @POST("/predict")
    fun getTest(
        @Part file: MultipartBody.Part,
        @Part("y") x: Float,
        @Part("x") y:  Float
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