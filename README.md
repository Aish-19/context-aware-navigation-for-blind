# Context-Aware Navigation for Visually Impaired Users

A real-time multimodal navigation system combining object detection, vision-language understanding, and text-to-speech audio guidance to support safe, independent outdoor navigation for visually impaired users.

---

## Overview

Visually impaired individuals rely heavily on mobility canes and auditory cues for navigation, making independent travel in unfamiliar environments highly challenging. This project builds an assistive navigation system that overlays computer vision capabilities — real-time obstacle detection and contextual scene understanding — to generate safety-first, context-aware audio instructions.

The system combines:
- **YOLOv26s** for real-time object detection and segmentation
- **Qwen VLM** for scene-level contextual reasoning
- **Text-to-Speech (TTS)** for audio instruction delivery

---

## Research Questions

**General:** How effectively can a real-time multimodal navigation system generate safe, actionable audio guidance for visually impaired users in dynamic outdoor environments, compared to a vision-only baseline?

**Specific questions addressed:**
1. How accurately can YOLOv26s detect pedestrians, vehicles, traffic lights, and obstacles in real time?
2. What is the minimum dataset size required for effective fine-tuning of new classes?
3. How can detected objects be translated into actionable navigation instructions?
4. How can depth estimation infer spatial proximity (e.g., "car 2 meters ahead — stop")?
5. What confidence threshold should be used before issuing a safety-critical warning?
6. How do false positives and false negatives impact user safety?
7. Do models perform consistently across different lighting and environmental conditions?

---

## System Pipeline

```
Video Input → YOLOv26s Detection → Depth Estimation → Qwen VLM Scene Understanding → TTS Audio Output
```

1. **Object Detection** — YOLOv26s detects and segments obstacles, vehicles, pedestrians, and navigation cues in real time
2. **Depth Estimation** — approximates distance to detected objects for proximity-aware alerts
3. **Scene Understanding** — Qwen VLM interprets the full scene and generates concise, safety-focused navigation instructions
4. **Audio Output** — TTS converts instructions to speech (e.g., "Obstacle ahead, 2 meters — stop", "Sidewalk clear, continue forward")

---

## Datasets

### Cityscapes (Primary — via AWS S3)
Download from: https://www.cityscapes-dataset.com/

Files needed:
- `leftImg8bit_trainvaltest.zip`
- `gtFine_trainvaltest.zip`

Upload to S3:
```bash
aws s3 cp leftImg8bit_trainvaltest.zip s3://your-bucket/raw/
aws s3 cp gtFine_trainvaltest.zip s3://your-bucket/raw/
```

### Roboflow Datasets
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")

# Stairs
rf.workspace("tom-lai-8bp7n").project("stairs-i2yia").version(3).download("yolo26")
```

### Local Datasets
- Walk / Don't Walk pedestrian signs (ZIP)
- Electric Scooter (ZIP)

---

## Unified Class Map

| ID | Class | Source |
|---|---|---|
| 0 | person | Cityscapes |
| 1 | rider | Cityscapes |
| 2 | car | Cityscapes |
| 3 | truck | Cityscapes |
| 4 | bus | Cityscapes |
| 5 | on_rails | Cityscapes |
| 6 | motorcycle | Cityscapes |
| 7 | bicycle | Cityscapes |
| 8 | caravan | Cityscapes |
| 9 | trailer | Cityscapes |
| 10 | wall | Cityscapes |
| 11 | fence | Cityscapes |
| 12 | guard_rail | Cityscapes |
| 13 | pole | Cityscapes |
| 14 | polegroup | Cityscapes |
| 15 | vegetation | Cityscapes |
| 16 | terrain | Cityscapes |
| 17 | stairs | Roboflow |
| 18 | pedestrian_walk | Local ZIP |
| 19 | pedestrian_stop | Local ZIP |
| 20 | electric_scooter | Local ZIP |

---

## Project Structure

```
context-aware-navigation/
├── Extraction-cityscapes.ipynb   # Cityscapes → YOLO format conversion
├── YOLO Object detection         # YOLOv26s fine-tuning notebook
├── backend/                      # FastAPI Qwen VLM guidance API
├── android-app/                  # Android camera + TTS prototype
├── path_guidance.py              # Standalone Qwen VLM image guidance script
├── tts_demo.py                   # Local TTS parsing/speech demo
├── extract.py                    # EC2 extraction script
├── merge.py                      # Dataset merge script (coming soon)
├── data.yaml                     # Unified dataset config
└── README.md
```

---

## Setup & Reproduction

### 1. Clone the repo
```bash
git clone https://github.com/your-username/context-aware-navigation.git
cd context-aware-navigation
```

### 2. Install dependencies
```bash
pip install opencv-python tqdm pillow ultralytics roboflow
```

### 3. Set up AWS (EC2 + S3)
- Launch EC2 `c5.4xlarge` instance (us-west-2)
- Attach IAM role with S3 access
- Upload Cityscapes ZIPs to S3

### 4. Run extraction on EC2
```bash
python3 extract.py
```

### 5. Merge all datasets
```bash
python3 merge.py  # coming soon
```

### 6. Train YOLOv26s
```bash
from ultralytics import YOLO
model = YOLO("yolo26s.pt")
model.train(data="data.yaml", epochs=50, imgsz=640)
```

---

## VLM Guidance Prototype

This repo includes a working Qwen2.5-VL prototype that turns a camera frame into a concise navigation instruction:

```json
{
  "direction": "move slightly right",
  "reason": "The most open sidewalk region is on the right side.",
  "spoken_instruction": "Move slightly to the right."
}
```

### Local Python Demo

Install the VLM dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-vlm.txt
```

Run the standalone image guidance script:

```bash
.venv/bin/python path_guidance.py
```

Run the text-to-speech demo:

```bash
.venv/bin/python tts_demo.py
```

### FastAPI Backend

Install backend dependencies:

```bash
.venv/bin/python -m pip install -r backend/requirements.txt
```

Start the API:

```bash
cd backend
../.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Analyze an image:

```bash
curl -F "image=@../frame.jpg" http://127.0.0.1:8000/guide
```

### Android App

Open `android-app/` in Android Studio.

- Emulator backend URL: `http://10.0.2.2:8000`
- Physical phone backend URL: change `android-app/app/src/main/res/values/strings.xml` to your computer's LAN IP, for example `http://192.168.1.20:8000`

The app captures a camera frame, sends it to the FastAPI backend, displays the returned direction and reason, and speaks `spoken_instruction` with Android TextToSpeech.

---

## Progress

- [x] Cityscapes dataset extraction to YOLO segmentation format
- [x] Train/val/test split
- [x] AWS S3 + EC2 pipeline set up
- [ ] Dataset merging (Cityscapes + Roboflow + local ZIPs)
- [ ] Unified class remapping
- [ ] YOLOv26s fine-tuning
- [ ] Depth estimation integration
- [x] Qwen VLM scene understanding prototype
- [x] TTS audio output prototype
- [x] Android camera + TTS prototype
- [ ] End-to-end system evaluation

---

## Evaluation Metrics

- Object detection accuracy (precision, recall, mAP per class)
- Safety effectiveness (obstacle avoidance rate)
- Robustness across lighting conditions (rain, night, bright sunlight)
- Audio instruction quality (clarity, conciseness, correctness)

---

## Authors

Aishwarya — Computer Vision Project, University Course
