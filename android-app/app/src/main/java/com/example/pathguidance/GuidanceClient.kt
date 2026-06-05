package com.example.pathguidance

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

class GuidanceClient(
    private val baseUrl: String,
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(180, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    fun analyze(imageFile: File): GuidanceResult {
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "image",
                imageFile.name,
                imageFile.asRequestBody("image/jpeg".toMediaType()),
            )
            .build()

        val request = Request.Builder()
            .url("${baseUrl.trimEnd('/')}/guide")
            .post(requestBody)
            .build()

        client.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                throw IOException("Backend error ${response.code}: $body")
            }

            val json = JSONObject(body)
            return GuidanceResult(
                direction = json.optString("direction", "stop"),
                reason = json.optString("reason", "No reason provided."),
                spokenInstruction = json.optString(
                    "spoken_instruction",
                    "Unable to determine path.",
                ),
            )
        }
    }
}
