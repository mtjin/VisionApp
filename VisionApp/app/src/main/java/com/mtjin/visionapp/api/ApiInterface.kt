package com.mtjin.visionapp.api

import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.http.Field
import retrofit2.http.FormUrlEncoded
import retrofit2.http.POST

interface ApiInterface {
    @FormUrlEncoded
    @POST("/predict")
    fun getTest(
        @Field("image") image: String,
        @Field("x") x: String,
        @Field("y") y: String
    ): Call<ResponseBody>
}