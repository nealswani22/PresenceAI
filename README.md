# PresenceAI👨‍💻

An AI-powered computer vision application built with Python, Flask, OpenCV, and MediaPipe. The system processes live webcam feeds, detects facial landmarks, computes multi-metric confidence scores and displays real-time visual analytics through a web dashboard.

---

## Features

* **Real-Time Landmark Detection:** Captures facial landmarks using MediaPipe Face Mesh.

* **Multi-Feature Confidence Scoring:**
  * **Head Pitch:** Computes facial geometry to analyze head positioning.
  * **Mouth Curve:** Tracks mouth curvature to measure smile formation.
  * **Brow Height:** Evaluates eyebrow position relative to the eyes.
  * **Eye Openness:** Measures eyelid aperture for visual engagement.
  * **Attentiveness:** Tracks the retinal movement to evaluate attentiveness.

* **Temporal Smoothing:** Uses a moving window average (`N=20`) to reduce fluctuations in the confidence score.

* **Face Mesh Toggle:** Allows the Face Mesh visualization to be enabled or disabled while continuing the underlying confidence analysis.

* **Live Web Dashboard:** Displays the webcam feed, overall confidence score, individual metrics, and confidence classification in real time.

* **Docker Support:** The application is containerized using Docker and served using Gunicorn.

* **Cloud Deployment:** Deployed as a web service using Render.

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend** | Python, Flask, OpenCV, MediaPipe, NumPy |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Communication** | REST API |
| **Deployment** | Render |

---

## Scoring Heuristics

Confidence is calculated using a weighted composite score bounded between `0.00` and `1.00`:

* **Mouth Curve (25%):** Tracks smile formation.
* **Eye Openness (20%):** Measures visual engagement and eye openness.
* **Head Pitch (15%):** Evaluates head positioning.
* **Brow Height (15%):** Measures eyebrow position.
* **Attentiveness (25%):** Tracks the user’s eye 

**Classification Threshold:**

* Score `≥ 0.62` → **`CONFIDENT`**
* Score `< 0.62` → **`UNDER-CONFIDENT`**

---

## Project Structure

```text
Confidence_Tracker/
│
├── app.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
