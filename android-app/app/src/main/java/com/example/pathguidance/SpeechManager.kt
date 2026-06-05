package com.example.pathguidance

import android.content.Context
import android.speech.tts.TextToSpeech
import java.util.Locale

class SpeechManager(
    context: Context,
) : TextToSpeech.OnInitListener {
    private var ready = false
    private val tts = TextToSpeech(context.applicationContext, this)

    override fun onInit(status: Int) {
        ready = status == TextToSpeech.SUCCESS
        if (ready) {
            tts.language = Locale.US
            tts.setSpeechRate(1.0f)
        }
    }

    fun speak(text: String) {
        if (!ready) return
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "path-guidance")
    }

    fun shutdown() {
        tts.stop()
        tts.shutdown()
    }
}
