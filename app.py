import os
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import base64

app = Flask(__name__)
CORS(app)


m_l, m_r, m_t, m_b = 61, 291, 13, 14
left_b_i, left_b_o = 107, 70
right_b_i, right_b_o = 336, 300
left_e_t, left_e_b = 159, 145
right_e_t, right_e_b = 386, 374
n, c, f = 1, 152, 10

left_iris = [474, 475, 476, 477]
right_iris = [469, 470, 471, 472]

left_eye_left = 33
left_eye_right = 133
right_eye_left = 362
right_eye_right = 263

HISTORY_LEN = 20
score_history = []

# ==========================================
# BENCHMARK VARIABLES
# ==========================================

benchmark_start = time.perf_counter()
benchmark_frames = 0
benchmark_latencies = []

# ==========================================
# MEDIAPIPE
# ==========================================

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def get_iris_center(landmarks, iris_indices, w, h):

    points = [
        get_point(landmarks, idx, w, h)
        for idx in iris_indices
    ]

    return np.mean(points, axis=0)


def compute_confidence_score(landmarks, w, h):

    scores = {}

    nose = get_point(landmarks, n, w, h)
    chin = get_point(landmarks, c, w, h)
    forehead = get_point(landmarks, f, w, h)

    left_iris_center = get_iris_center(
        landmarks,
        left_iris,
        w,
        h
    )

    right_iris_center = get_iris_center(
        landmarks,
        right_iris,
        w,
        h
    )

    l_eye_left = get_point(
        landmarks,
        left_eye_left,
        w,
        h
    )

    l_eye_right = get_point(
        landmarks,
        left_eye_right,
        w,
        h
    )

    r_eye_left = get_point(
        landmarks,
        right_eye_left,
        w,
        h
    )

    r_eye_right = get_point(
        landmarks,
        right_eye_right,
        w,
        h
    )

    left_eye_width = (
        np.linalg.norm(
            l_eye_right - l_eye_left
        ) + 1e-6
    )

    right_eye_width = (
        np.linalg.norm(
            r_eye_right - r_eye_left
        ) + 1e-6
    )

    left_eye_min_x = min(
        l_eye_left[0],
        l_eye_right[0]
    )

    left_eye_max_x = max(
        l_eye_left[0],
        l_eye_right[0]
    )

    right_eye_min_x = min(
        r_eye_left[0],
        r_eye_right[0]
    )

    right_eye_max_x = max(
        r_eye_left[0],
        r_eye_right[0]
    )

    left_gaze_x = (
        left_iris_center[0] - left_eye_min_x
    ) / (
        left_eye_max_x - left_eye_min_x + 1e-6
    )

    right_gaze_x = (
        right_iris_center[0] - right_eye_min_x
    ) / (
        right_eye_max_x - right_eye_min_x + 1e-6
    )

    avg_gaze_x = (
        left_gaze_x + right_gaze_x
    ) / 2

    upper = nose[1] - forehead[1]
    lower = chin[1] - nose[1]

    ratio = upper / (lower + 1e-6)

    scores["head_pitch"] = float(
        np.clip(
            (ratio - 0.4) / 0.7,
            0,
            1
        )
    )

    ml = get_point(
        landmarks,
        m_l,
        w,
        h
    )

    mr = get_point(
        landmarks,
        m_r,
        w,
        h
    )

    mt = get_point(
        landmarks,
        m_t,
        w,
        h
    )

    corner_avg_y = (
        ml[1] + mr[1]
    ) / 2

    center_y = mt[1]

    mouth_width = (
        np.linalg.norm(mr - ml)
        + 1e-6
    )

    curve = (
        center_y - corner_avg_y
    ) / mouth_width

    scores["mouth_curve"] = float(
        np.clip(
            0.5 + curve * 3,
            0,
            1
        )
    )

    l_brow_i = get_point(
        landmarks,
        left_b_i,
        w,
        h
    )

    l_brow_o = get_point(
        landmarks,
        left_b_o,
        w,
        h
    )

    r_brow_i = get_point(
        landmarks,
        right_b_i,
        w,
        h
    )

    r_brow_o = get_point(
        landmarks,
        right_b_o,
        w,
        h
    )

    l_eye_t = get_point(
        landmarks,
        left_e_t,
        w,
        h
    )

    r_eye_t = get_point(
        landmarks,
        right_e_t,
        w,
        h
    )

    left_brow_height = (
        l_eye_t[1]
        - (
            l_brow_i[1]
            + l_brow_o[1]
        ) / 2
    ) / (h + 1e-6)

    right_brow_height = (
        r_eye_t[1]
        - (
            r_brow_i[1]
            + r_brow_o[1]
        ) / 2
    ) / (h + 1e-6)

    avg_brow = (
        left_brow_height
        + right_brow_height
    ) / 2

    scores["brow_height"] = float(
        np.clip(
            avg_brow * 30,
            0,
            1
        )
    )

    le_t = get_point(
        landmarks,
        left_e_t,
        w,
        h
    )

    le_b = get_point(
        landmarks,
        left_e_b,
        w,
        h
    )

    re_t = get_point(
        landmarks,
        right_e_t,
        w,
        h
    )

    re_b = get_point(
        landmarks,
        right_e_b,
        w,
        h
    )

    left_open = (
        abs(le_b[1] - le_t[1])
        / (h + 1e-6)
    )

    right_open = (
        abs(re_b[1] - re_t[1])
        / (h + 1e-6)
    )

    avg_eye = (
        left_open + right_open
    ) / 2

    scores["eye_openness"] = float(
        np.clip(
            avg_eye * 50,
            0,
            1
        )
    )

    left_eye_height = (
        abs(le_b[1] - le_t[1])
        + 1e-6
    )

    right_eye_height = (
        abs(re_b[1] - re_t[1])
        + 1e-6
    )

    left_gaze_y = (
        left_iris_center[1] - le_t[1]
    ) / left_eye_height

    right_gaze_y = (
        right_iris_center[1] - re_t[1]
    ) / right_eye_height

    avg_gaze_y = (
        left_gaze_y + right_gaze_y
    ) / 2

    camera_gaze_x = 0.37
    right_limit = 0.24
    left_limit = 0.60

    if avg_gaze_x < camera_gaze_x:

        eye_look_score = (
            avg_gaze_x - right_limit
        ) / (
            camera_gaze_x - right_limit
        )

    else:

        eye_look_score = (
            left_limit - avg_gaze_x
        ) / (
            left_limit - camera_gaze_x
        )

    eye_look_score = float(
        np.clip(
            eye_look_score,
            0,
            1
        )
    )

    scores["eye_look"] = eye_look_score

    weights = {

        "head_pitch": 0.18,

        "mouth_curve": 0.10,

        "brow_height": 0.20,

        "eye_openness": 0.25,

        "eye_look": 0.27

    }

    final = sum(
        scores[k] * weights[k]
        for k in weights
    )

    return final, scores


def smooth_score(new_score):

    score_history.append(new_score)

    if len(score_history) > HISTORY_LEN:
        score_history.pop(0)

    return sum(score_history) / len(score_history)


@app.route('/')
def index():

    return render_template('index.html')


@app.route('/api/process_frame', methods=['POST'])
def process_frame():

    global benchmark_frames
    global benchmark_start
    global benchmark_latencies

    # ==========================================
    # START TOTAL REQUEST TIMER
    # ==========================================

    request_start = time.perf_counter()

    data = request.json

    if not data or 'image' not in data:

        return jsonify({
            'error': 'No image provided'
        }), 400

    img_bytes = base64.b64decode(
        data['image'].split(',')[1]
    )

    np_arr = np.frombuffer(
        img_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        return jsonify({
            'error': 'Invalid image'
        }), 400

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # ==========================================
    # MEDIAPIPE TIMER
    # ==========================================

    mediapipe_start = time.perf_counter()

    results = mp_face_mesh.process(rgb)

    mediapipe_latency = (
        time.perf_counter()
        - mediapipe_start
    ) * 1000

    # ==========================================
    # DEFAULT METRICS
    # ==========================================

    metrics = {

        "score": 0.0,

        "status": "NO FACE DETECTED",

        "sub_scores": {

            "head_pitch": 0.0,

            "mouth_curve": 0.0,

            "brow_height": 0.0,

            "eye_openness": 0.0,

            "eye_look": 0.0

        }

    }

    landmarks = []

    confidence_latency = 0

    # ==========================================
    # CONFIDENCE CALCULATION
    # ==========================================

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            landmarks = [

                {
                    "x": float(lm.x),
                    "y": float(lm.y)
                }

                for lm in face_landmarks.landmark

            ]

            confidence_start = time.perf_counter()

            raw_score, sub_scores = (
                compute_confidence_score(
                    face_landmarks.landmark,
                    w,
                    h
                )
            )

            smoothed = smooth_score(
                raw_score
            )

            confidence_latency = (
                time.perf_counter()
                - confidence_start
            ) * 1000

            status_text = (

                "CONFIDENT"

                if smoothed >= 0.62

                else "UNDER-CONFIDENT"

            )

            metrics = {

                "score": round(
                    smoothed,
                    2
                ),

                "status": status_text,

                "sub_scores": {

                    k: round(v, 2)

                    for k, v in sub_scores.items()

                }

            }

    # ==========================================
    # TOTAL LATENCY
    # ==========================================

    total_latency = (
        time.perf_counter()
        - request_start
    ) * 1000

    benchmark_frames += 1

    benchmark_latencies.append(
        total_latency
    )

    # Keep only last 100 measurements
    if len(benchmark_latencies) > 100:

        benchmark_latencies.pop(0)

    elapsed = (
        time.perf_counter()
        - benchmark_start
    )

    current_fps = (
        benchmark_frames / elapsed
        if elapsed > 0
        else 0
    )

    average_latency = (
        sum(benchmark_latencies)
        / len(benchmark_latencies)
    )

    # ==========================================
    # PRINT BENCHMARK
    # ==========================================

    if benchmark_frames % 20 == 0:

        print(
            "\n========== PRESENCEAI BENCHMARK =========="
        )

        print(
            f"Frames processed: {benchmark_frames}"
        )

        print(
            f"Backend FPS: {current_fps:.2f}"
        )

        print(
            f"Average API latency: "
            f"{average_latency:.2f} ms"
        )

        print(
            f"Latest API latency: "
            f"{total_latency:.2f} ms"
        )

        print(
            f"MediaPipe latency: "
            f"{mediapipe_latency:.2f} ms"
        )

        print(
            f"Confidence calculation: "
            f"{confidence_latency:.2f} ms"
        )

        print(
            "===========================================\n"
        )

    # ==========================================
    # RETURN DATA TO FRONTEND
    # ==========================================

    return jsonify({

        "metrics": metrics,

        "landmarks": landmarks,

        "benchmark": {

            "fps": round(
                current_fps,
                2
            ),

            "latency_ms": round(
                total_latency,
                2
            ),

            "average_latency_ms": round(
                average_latency,
                2
            ),

            "mediapipe_latency_ms": round(
                mediapipe_latency,
                2
            )

        }

    })


if __name__ == '__main__':

    port = int(
        os.environ.get(
            "PORT",
            5001
        )
    )

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )
