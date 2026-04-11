# 🦺 PPE Compliance Monitor — Real-Time Safety Detection

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?style=flat-square&logo=streamlit)](https://ppe-compliance-monitor-7ssrkxc83lsifmxnfngote.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=flat-square&logo=opencv)
![Classes](https://img.shields.io/badge/Classes-10-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> Real-time PPE (Personal Protective Equipment) compliance monitoring using YOLOv8 and OpenCV — automated safety detection for construction sites and industrial operations.

🔗 **[Try the Live Demo →](https://ppe-compliance-monitor-7ssrkxc83lsifmxnfngote.streamlit.app/)**

---

## 📌 Overview

This project detects PPE compliance in real time using a YOLOv8 object detection model trained on 10 classes. It is designed for industrial and construction environments where worker safety monitoring is critical.

**Detects 10 Classes:**

| Class | Type |
|---|---|
| ✅ `Hardhat` | Helmet worn |
| ❌ `NO-Hardhat` | Helmet not worn |
| ✅ `Mask` | Mask worn |
| ❌ `NO-Mask` | Mask not worn |
| ✅ `Safety Vest` | Vest worn |
| ❌ `NO-Safety Vest` | Vest not worn |
| 🟡 `Person` | Worker detected |
| 🟡 `Safety Cone` | Cone detected |
| 🟡 `machinery` | Machinery detected |
| 🟡 `vehicle` | Vehicle detected |

**Supports:**
- 📷 Webcam (live feed)
- 🎞️ Video files (`.mp4`, `.avi`, etc.)
- 🖼️ Images (`.jpg`, `.png`, etc.)
- 📡 RTSP streams (IP cameras)

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Model | YOLOv8s |
| mAP@0.5 | 0.856 |
| Precision | 0.883 |
| Recall | 0.821 |
| Input Size | 640×640 |
| Classes | 10 (PPE categories) |
| Inference Speed | ~30 FPS |
| Framework | Ultralytics + PyTorch |

---

## 🗂️ Project Structure

```
PPE-Compliance-Monitor/
├── app.py                # Streamlit web application
├── detect.py             # Inference / detection script
├── train.py              # Model training script
├── requirements.txt      # Python dependencies
├── packages.txt          # System dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Arunsuresh677/PPE-Compliance-Monitor.git
cd PPE-Compliance-Monitor
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run Streamlit Web App
```bash
streamlit run app.py
```

### Run on Webcam
```bash
python detect.py --source 0
```

### Run on Video File
```bash
python detect.py --source path/to/video.mp4
```

### Run on Image
```bash
python detect.py --source path/to/image.jpg
```

### Run on RTSP Stream
```bash
python detect.py --source rtsp://username:password@ip_address:port/stream
```

---

## 🏋️ Training

```bash
python train.py --data data.yaml --epochs 50 --imgsz 640
```

Dataset should follow YOLO format:
```
datasets/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| YOLOv8 (Ultralytics) | Object Detection |
| OpenCV | Video/Image Processing |
| PyTorch | Deep Learning Backend |
| Streamlit | Web Application |
| Hugging Face | Model Hosting |
| Python 3.8+ | Core Language |

---

## 👤 Author

**Arun S**
AI/ML Engineer
[LinkedIn](https://linkedin.com/in/arun-s-481b2b3a2) · [GitHub](https://github.com/Arunsuresh677)

---

## 📄 License

This project is licensed under the MIT License.