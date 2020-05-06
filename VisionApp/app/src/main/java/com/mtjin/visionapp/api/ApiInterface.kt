package com.mtjin.visionapp.api

import retrofit2.Call
import retrofit2.http.GET
import retrofit2.http.Query

interface ApiInterface {
    @GET("/")
    fun getTest(
    ): Call<String>
}