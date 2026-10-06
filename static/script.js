const video = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const confidenceScore =
    document.getElementById("confidenceScore");

const confidenceBar =
    document.getElementById("confidenceBar");

const confidenceStatus =
    document.getElementById("confidenceStatus");

const headPitch =
    document.getElementById("headPitch");

const headPitchBar =
    document.getElementById("headPitchBar");

const mouthCurve =
    document.getElementById("mouthCurve");

const mouthCurveBar =
    document.getElementById("mouthCurveBar");

const browHeight =
    document.getElementById("browHeight");

const browHeightBar =
    document.getElementById("browHeightBar");

const eyeOpenness =
    document.getElementById("eyeOpenness");

const eyeOpennessBar =
    document.getElementById("eyeOpennessBar");

const eyeAttention =
    document.getElementById("eyeAttention");

const eyeAttentionBar =
    document.getElementById("eyeAttentionBar");

const statusText =
    document.getElementById("statusText");

const analysisStatus =
    document.getElementById("analysisStatus");

const faceStatus =
    document.getElementById("faceStatus");

const startButton =
    document.getElementById("startButton");

const stopButton =
    document.getElementById("stopButton");

const resetButton =
    document.getElementById("resetButton");

const meshToggle =
    document.getElementById("meshToggle");

let stream;
let running = false;
let processing = false;
let meshVisible = true;


// ==========================================
// BENCHMARK VARIABLES
// ==========================================

let benchmarkFrames = 0;

let benchmarkStartTime = null;

let frontendLatencies = [];


// ==========================================
// CAMERA
// ==========================================

async function startCamera() {

    try {

        stream =
            await navigator.mediaDevices.getUserMedia({

                video: true,

                audio: false

            });

        video.srcObject = stream;

        video.onloadedmetadata = () => {

            canvas.width =
                video.videoWidth;

            canvas.height =
                video.videoHeight;

        };

    }

    catch (error) {

        console.error(
            "Camera error:",
            error
        );

        statusText.textContent =
            "CAMERA ERROR";

    }

}


// ==========================================
// START ANALYSIS
// ==========================================

async function startAnalysis() {

    if (!stream) {

        await startCamera();

    }

    running = true;

    benchmarkFrames = 0;

    benchmarkStartTime =
        performance.now();

    frontendLatencies = [];

    statusText.textContent =
        "ANALYZING";

    analysisStatus.textContent =
        "RUNNING";

    processFrame();

}


// ==========================================
// STOP
// ==========================================

function stopAnalysis() {

    running = false;

    statusText.textContent =
        "STOPPED";

    analysisStatus.textContent =
        "STOPPED";

}


// ==========================================
// PROCESS FRAME
// ==========================================

async function processFrame() {

    if (!running || processing) {

        return;

    }

    processing = true;

    const requestStart =
        performance.now();

    const captureCanvas =
        document.createElement("canvas");

    captureCanvas.width =
        video.videoWidth;

    captureCanvas.height =
        video.videoHeight;

    const captureCtx =
        captureCanvas.getContext("2d");

    captureCtx.drawImage(

        video,

        0,

        0,

        captureCanvas.width,

        captureCanvas.height

    );

    const image =
        captureCanvas.toDataURL(
            "image/jpeg",
            0.7
        );

    try {

        const response =
            await fetch(

                "/api/process_frame",

                {

                    method: "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body: JSON.stringify({

                        image: image

                    })

                }

            );

        const data =
            await response.json();

        // ======================================
        // FRONTEND END-TO-END LATENCY
        // ======================================

        const requestEnd =
            performance.now();

        const latency =
            requestEnd - requestStart;

        benchmarkFrames++;

        frontendLatencies.push(
            latency
        );

        // Keep last 100 measurements
        if (
            frontendLatencies.length > 100
        ) {

            frontendLatencies.shift();

        }

        const elapsed =
            (
                performance.now()
                - benchmarkStartTime
            ) / 1000;

        const fps =
            benchmarkFrames / elapsed;

        const averageLatency =
            frontendLatencies.reduce(
                (a, b) => a + b,
                0
            )
            / frontendLatencies.length;

        // ======================================
        // PRINT RESULTS
        // ======================================

        if (
            benchmarkFrames % 20 === 0
        ) {

            console.log(
                "\n========== PRESENCEAI FRONTEND =========="
            );

            console.log(
                "Frames:",
                benchmarkFrames
            );

            console.log(
                "Actual FPS:",
                fps.toFixed(2)
            );

            console.log(
                "Average end-to-end latency:",
                averageLatency.toFixed(2),
                "ms"
            );

            if (data.benchmark) {

                console.log(
                    "Backend API latency:",
                    data.benchmark.latency_ms,
                    "ms"
                );

                console.log(
                    "MediaPipe latency:",
                    data.benchmark.mediapipe_latency_ms,
                    "ms"
                );

            }

            console.log(
                "=========================================="
            );

        }

        updateUI(data);

        if (data.landmarks) {

            drawLandmarks(
                data.landmarks
            );

        }

    }

    catch (error) {

        console.error(
            "Processing error:",
            error
        );

    }

    processing = false;

    if (running) {

        setTimeout(
            processFrame,
            150
        );

    }

}


// ==========================================
// UPDATE UI
// ==========================================

function updateUI(data) {

    const metrics =
        data.metrics;

    const score =
        Math.round(
            metrics.score * 100
        );

    confidenceScore.textContent =
        score + "%";

    confidenceBar.style.width =
        score + "%";

    confidenceStatus.textContent =
        metrics.status;

    analysisStatus.textContent =
        metrics.status;

    if (
        metrics.status ===
        "NO FACE DETECTED"
    ) {

        faceStatus.textContent =
            "NO FACE";

    }

    else {

        faceStatus.textContent =
            "FACE DETECTED";

    }

    updateMetric(

        headPitch,

        headPitchBar,

        metrics.sub_scores.head_pitch

    );

    updateMetric(

        mouthCurve,

        mouthCurveBar,

        metrics.sub_scores.mouth_curve

    );

    updateMetric(

        browHeight,

        browHeightBar,

        metrics.sub_scores.brow_height

    );

    updateMetric(

        eyeOpenness,

        eyeOpennessBar,

        metrics.sub_scores.eye_openness

    );

    updateMetric(

        eyeAttention,

        eyeAttentionBar,

        metrics.sub_scores.eye_look

    );

}


// ==========================================
// UPDATE METRIC
// ==========================================

function updateMetric(
    valueElement,
    barElement,
    value
) {

    const percentage =
        Math.round(
            value * 100
        );

    valueElement.textContent =
        percentage + "%";

    barElement.style.width =
        percentage + "%";

}


// ==========================================
// DRAW LANDMARKS
// ==========================================

function drawLandmarks(
    landmarks
) {

    ctx.clearRect(

        0,

        0,

        canvas.width,

        canvas.height

    );

    if (
        !meshVisible ||
        !landmarks.length
    ) {

        return;

    }

    ctx.fillStyle =
        "#57c7bb";

    for (
        const point of landmarks
    ) {

        const x =
            point.x *
            canvas.width;

        const y =
            point.y *
            canvas.height;

        ctx.beginPath();

        ctx.arc(

            x,

            y,

            1.2,

            0,

            Math.PI * 2

        );

        ctx.fill();

    }

}


// ==========================================
// RESET
// ==========================================

function resetUI() {

    running = false;

    benchmarkFrames = 0;

    benchmarkStartTime = null;

    frontendLatencies = [];

    confidenceScore.textContent =
        "0%";

    confidenceBar.style.width =
        "0%";

    confidenceStatus.textContent =
        "WAITING";

    analysisStatus.textContent =
        "WAITING";

    faceStatus.textContent =
        "NO FACE";

    const metrics = [

        [
            headPitch,
            headPitchBar
        ],

        [
            mouthCurve,
            mouthCurveBar
        ],

        [
            browHeight,
            browHeightBar
        ],

        [
            eyeOpenness,
            eyeOpennessBar
        ],

        [
            eyeAttention,
            eyeAttentionBar
        ]

    ];

    metrics.forEach(
        ([value, bar]) => {

            value.textContent =
                "0%";

            bar.style.width =
                "0%";

        }
    );

    ctx.clearRect(

        0,

        0,

        canvas.width,

        canvas.height

    );

    statusText.textContent =
        "SYSTEM READY";

}


// ==========================================
// BUTTONS
// ==========================================

startButton.addEventListener(

    "click",

    startAnalysis

);

stopButton.addEventListener(

    "click",

    stopAnalysis

);

resetButton.addEventListener(

    "click",

    resetUI

);


// ==========================================
// MESH TOGGLE
// ==========================================

meshToggle.addEventListener(

    "click",

    () => {

        meshVisible =
            !meshVisible;

        if (meshVisible) {

            meshToggle.innerHTML =
                '<i class="bi bi-eye"></i> MESH ON';

        }

        else {

            meshToggle.innerHTML =
                '<i class="bi bi-eye-slash"></i> MESH OFF';

            ctx.clearRect(

                0,

                0,

                canvas.width,

                canvas.height

            );

        }

    }

);
