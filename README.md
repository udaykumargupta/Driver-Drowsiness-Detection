# Driver Drowsiness Detection

A real-time driver drowsiness detection system using a hybrid deep learning and geometric approach. It combines a MobileNet-based CNN for eye state classification with a Mouth Aspect Ratio (MAR)-based method for yawning detection to ensure accurate and low-latency performance on resource-constrained hardware.

## 🚗 Problem Statement

Driver drowsiness is a leading cause of road accidents due to delayed response and impaired decision-making. Traditional detection methods like EEG or steering analysis are often intrusive or unreliable. This project aims to deliver a **non-intrusive, real-time, vision-based system** for detecting signs of fatigue like prolonged eye closure and yawning.

---

## 🎯 Objectives

- Detect driver drowsiness using visual cues (eye closure and yawning).
- Use **MobileNet CNN** for eye state classification (open/closed).
- Use **MAR (Mouth Aspect Ratio)** for yawn detection without model training.
- Display real-time alerts using OpenCV overlays.
- Run efficiently on low-power edge devices (Jetson Nano, Raspberry Pi, etc.).

---

## 🧠 Methodology

### 📸 Input
- Webcam-based real-time video stream

### 🔍 Detection Pipeline
1. **Face & Eye Detection:** Haar cascade classifier
2. **Eye State Detection:** MobileNet CNN model trained on 85k labeled images
3. **Yawn Detection:** MAR computed from MediaPipe face landmarks
4. **Frame Threshold Logic:**
   - `≥15` closed eye frames → **Drowsiness Alert**
   - `≥10` yawn frames → **Yawning Alert**
5. **Alert Display:** Real-time warnings overlaid using OpenCV

---

## 🧪 Results

### 👁 Eye Model (MobileNet CNN)
- **Accuracy:** 97.9%
- **Precision:** 97.71%
- **Recall:** 98.17%
- **F1-Score:** 97.94%

### 👄 Yawn Detection (MAR)
- **Accuracy:** 83.87%
- **False Positive Rate:** 9.68%
- **No training required**

### ⚙️ Performance
- **~45 FPS on GPU**
- **~15 FPS on CPU**
- **Low latency (~22ms on GPU)**

---

## 🛠 Tech Stack

- **Languages:** Python
- **Libraries:** TensorFlow, OpenCV, MediaPipe, NumPy, SciPy
- **Model:** MobileNet (Keras `.h5`)
- **Tools:** Jupyter Notebook, Git, GitHub

---
