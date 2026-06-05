# Path Guidance Android App

Prototype Android client for the path guidance backend.

## What it does

- Shows a live camera preview.
- Captures one frame when you tap **Analyze**.
- Sends that JPEG to `POST /guide`.
- Displays the returned direction/reason.
- Speaks `spoken_instruction` using Android TextToSpeech.

## Backend URL

The default backend URL is in:

```text
app/src/main/res/values/strings.xml
```

For the Android emulator, keep:

```text
http://10.0.2.2:8000
```

For a physical Android phone, change it to your Mac's LAN IP address, for example:

```text
http://192.168.1.20:8000
```

Run the backend with:

```bash
cd ../backend
../.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

## Build

Open `android-app/` in Android Studio and run the `app` configuration.

This machine does not currently have Android Studio/SDK installed, and the global `gradle` command fails before loading a project, so local APK build verification was not possible from this workspace yet.
