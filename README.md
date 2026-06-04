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

## Progress

- [x] Cityscapes dataset extraction to YOLO segmentation format
- [x] Train/val/test split
- [x] AWS S3 + EC2 pipeline set up
- [ ] Dataset merging (Cityscapes + Roboflow + local ZIPs)
- [ ] Unified class remapping
- [ ] YOLOv26s fine-tuning
- [ ] Depth estimation integration
- [ ] Qwen VLM scene understanding
- [ ] TTS audio output
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
