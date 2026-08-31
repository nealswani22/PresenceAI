const video = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const confidenceScore = document.getElementById("confidenceScore");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceStatus = document.getElementById("confidenceStatus");

const headPitch = document.getElementById("headPitch");
const headPitchBar = document.getElementById("headPitchBar");

const mouthCurve = document.getElementById("mouthCurve");
const mouthCurveBar = document.getElementById("mouthCurveBar");

const browHeight = document.getElementById("browHeight");
const browHeightBar = document.getElementById("browHeightBar");

const eyeOpenness = document.getElementById("eyeOpenness");
const eyeOpennessBar = document.getElementById("eyeOpennessBar");

const eyeAttention = document.getElementById("eyeAttention");
const eyeAttentionBar = document.getElementById("eyeAttentionBar");

const statusText = document.getElementById("statusText");
const analysisStatus = document.getElementById("analysisStatus");
const faceStatus = document.getElementById("faceStatus");

const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const resetButton = document.getElementById("resetButton");

const meshToggle = document.getElementById("meshToggle");

let stream;
let running = false;
let processing = false;
let meshVisible = true;

const connections = [
    [10, 109], [109, 67], [67, 103], [103, 54],
    [54, 21], [21, 162], [162, 127], [127, 234],
    [234, 93], [93, 132], [132, 58], [58, 172],
    [172, 136], [136, 150], [150, 149], [149, 176],
    [176, 148], [148, 152], [152, 377], [377, 400],
    [400, 378], [378, 379], [379, 365], [365, 397],
    [397, 288], [288, 361], [361, 323], [323, 454],
    [454, 356], [356, 389], [389, 251], [251, 284],
    [284, 332], [332, 297], [297, 338], [338, 10],

    [33, 7], [7, 163], [163, 144], [144, 145],
    [145, 153], [153, 154], [154, 155], [155, 133],
    [133, 173], [173, 157], [157, 158], [158, 159],
    [159, 160], [160, 161], [161, 246], [246, 33],

    [362, 382], [382, 381], [381, 380], [380, 374],
    [374, 373], [373, 390], [390, 249], [249, 263],
    [263, 466], [466, 388], [388, 387], [387, 386],
    [386, 385], [385, 384], [384, 398], [398, 362],

    [61, 146], [146, 91], [91, 181], [181, 84],
    [84, 17], [17, 314], [314, 405], [405, 321],
    [321, 375], [375, 291], [291, 308], [308, 324],
    [324, 318], [318, 402], [402, 317], [317, 14],
    [14, 87], [87, 178], [178, 88], [88, 95],
    [95, 78], [78, 191], [191, 80], [80, 81],
    [81, 82], [82, 13], [13, 312], [312, 311],
    [311, 310], [310, 415], [415, 308]
];

async function startCamera() {

    try {

        stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });

        video.srcObject = stream;

        video.onloadedmetadata = () => {

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

        };

    }

    catch (error) {

        console.error("Camera error:", error);

        statusText.textContent = "CAMERA ERROR";

    }

}

async function startAnalysis() {

    if (!stream) {
        await startCamera();
    }

    running = true;

    statusText.textContent = "ANALYZING";
    analysisStatus.textContent = "RUNNING";

    processFrame();

}

function stopAnalysis() {

    running = false;

    statusText.textContent = "STOPPED";
    analysisStatus.textContent = "STOPPED";

}

async function processFrame() {

    if (!running || processing) {
        return;
    }

    processing = true;

    const captureCanvas = document.createElement("canvas");

    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;

    const captureCtx = captureCanvas.getContext("2d");

    captureCtx.drawImage(
        video,
        0,
        0,
        captureCanvas.width,
        captureCanvas.height
    );

    const image = captureCanvas.toDataURL(
        "image/jpeg",
        0.7
    );

    try {

        const response = await fetch(
            "/api/process_frame",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    image: image
                })
            }
        );

        const data = await response.json();

        updateUI(data);

        if (data.landmarks) {
            drawLandmarks(data.landmarks);
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

function updateUI(data) {

    const metrics = data.metrics;

    const score = Math.round(
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
        metrics.status === "NO FACE DETECTED"
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

function updateMetric(
    valueElement,
    barElement,
    value
) {

    const percentage = Math.round(
        value * 100
    );

    valueElement.textContent =
        percentage + "%";

    barElement.style.width =
        percentage + "%";

}

function drawLandmarks(landmarks) {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    if (!meshVisible || !landmarks.length) {
        return;
    }

    ctx.strokeStyle =
        "#57c7bb";

    ctx.lineWidth =
        0.7;

    for (const [start, end] of connections) {

        if (
            !landmarks[start] ||
            !landmarks[end]
        ) {
            continue;
        }

        ctx.beginPath();

        ctx.moveTo(
            landmarks[start].x * canvas.width,
            landmarks[start].y * canvas.height
        );

        ctx.lineTo(
            landmarks[end].x * canvas.width,
            landmarks[end].y * canvas.height
        );

        ctx.stroke();

    }

}

function resetUI() {

    running = false;

    confidenceScore.textContent = "0%";
    confidenceBar.style.width = "0%";

    confidenceStatus.textContent = "WAITING";

    analysisStatus.textContent = "WAITING";

    faceStatus.textContent = "NO FACE";

    const metrics = [
        [headPitch, headPitchBar],
        [mouthCurve, mouthCurveBar],
        [browHeight, browHeightBar],
        [eyeOpenness, eyeOpennessBar],
        [eyeAttention, eyeAttentionBar]
    ];

    metrics.forEach(
        ([value, bar]) => {

            value.textContent = "0%";
            bar.style.width = "0%";

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