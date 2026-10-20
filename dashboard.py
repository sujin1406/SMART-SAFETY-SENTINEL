from pathlib import Path
import tempfile

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO


# =========================================================
# SMART SAFETY SENTINEL
# PHASE 3: VIDEO + LIVE CAMERA WORKER TRACKING AND PPE DETECTION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

PERSON_MODEL_PATH = BASE_DIR / "yolo11n.pt"
PPE_MODEL_PATH = BASE_DIR / "models" / "buildvision-hazards-v0.3.pt"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Only these classes explicitly indicate a missing item in this model.
# This is a frame-level alert: PPE boxes are not reliably assigned to a worker yet.
MISSING_PPE_CLASSES = {"no-hardhat", "no-safety vest", "no-mask", "no-goggles", "no-gloves"}

def is_missing_ppe(class_name):
    """Return True only for model labels explicitly indicating missing PPE."""
    normalized = " ".join(str(class_name).lower().replace("_", "-").split())
    return normalized in MISSING_PPE_CLASSES


# ---------------- STREAMLIT CONFIGURATION ----------------

st.set_page_config(
    page_title="Smart Safety Sentinel",
    page_icon="🏗️",
    layout="wide"
)


# ---------------- DASHBOARD STYLE ----------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --bg:#080d17; --panel:#111a2a; --panel2:#151f32; --line:#26344b; --text:#edf3ff; --muted:#94a3b8; --cyan:#42d6c5; --blue:#5a8cff; --red:#ff4d5e; }
    .stApp { background: radial-gradient(ellipse at 15% 0%, #152747 0%, #0b1220 38%, #080d17 76%); color:var(--text); font-family:'DM Sans',sans-serif; }
    [data-testid="stHeader"] { background:rgba(8,13,23,.82); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#0c1423,#090f1b); border-right:1px solid #202c40; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#b8c5d9; }
    [data-testid="stSidebar"] .stRadio label { padding:9px 12px; border-radius:10px; }
    [data-testid="stSidebar"] .stRadio label:hover { background:#17253a; }
    .block-container { max-width:1500px; padding-top:2rem; padding-bottom:2.5rem; }
    h1,h2,h3,h4 { font-family:'Space Grotesk','DM Sans',sans-serif !important; letter-spacing:-.025em; color:#f4f7ff; }
    h2 { margin-top:.4rem; }
    p, li { color:#c4cee0; }
    .hero { position:relative; overflow:hidden; padding:28px 30px; border:1px solid #2a3a53; border-radius:22px; background:linear-gradient(115deg,rgba(24,42,68,.96),rgba(15,25,42,.92)); box-shadow:0 18px 50px rgba(0,0,0,.22); margin-bottom:24px; }
    .hero:after { content:''; position:absolute; width:250px;height:250px;right:-70px;top:-120px;border-radius:50%;background:rgba(66,214,197,.10);filter:blur(2px); }
    .hero-kicker { color:var(--cyan); text-transform:uppercase; letter-spacing:.17em; font-size:11px; font-weight:700; margin-bottom:9px; }
    .hero-title { color:#f5f8ff; font:700 clamp(26px,3vw,38px)/1.1 'Space Grotesk',sans-serif; margin:0 0 10px; }
    .hero-subtitle { color:#aab9ce; font-size:14px; margin:0; max-width:720px; }
    .hero-pills { display:flex; flex-wrap:wrap; gap:8px; margin-top:20px; }
    .pill { border:1px solid #354761; border-radius:999px; background:rgba(8,16,29,.48); padding:6px 10px; color:#d9e6f8; font-size:11px; font-weight:600; }
    .section-kicker { color:var(--cyan); font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.14em; margin:18px 0 5px; }
    .section-note { color:var(--muted); font-size:13px; margin-top:-2px; margin-bottom:16px; }
    [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(21,32,51,.96),rgba(14,23,38,.96)); border:1px solid #283850; border-radius:16px; padding:17px 18px; box-shadow:0 8px 24px rgba(0,0,0,.12); }
    [data-testid="stMetricLabel"] { color:#9eafc7 !important; font-size:12px !important; }
    [data-testid="stMetricValue"] { color:#f2f7ff !important; font-family:'Space Grotesk',sans-serif; }
    [data-testid="stMetricDelta"] { font-size:11px; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-color:#293950 !important; border-radius:16px !important; }
    .stButton > button, .stDownloadButton > button { border-radius:11px; min-height:43px; font-weight:700; border:1px solid #344762; transition:all .18s ease; }
    .stButton > button[kind="primary"] { background:linear-gradient(100deg,#36cbbb,#278baf); color:#06131b; border:0; box-shadow:0 6px 18px rgba(54,203,187,.18); }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--cyan); transform:translateY(-1px); }
    [data-testid="stFileUploader"] { border:1px dashed #3b526f; background:rgba(17,28,45,.62); border-radius:14px; padding:10px; }
    [data-testid="stDataFrame"] { border:1px solid #283850; border-radius:12px; overflow:hidden; }
    .stAlert { border-radius:12px; }
    hr { border-color:#25334a !important; }
    .sentinel-alert { animation:sentinelBlink .7s steps(2,start) infinite; border:2px solid #ff6571; border-radius:13px; padding:14px 18px; color:white; font-weight:800; font-size:17px; text-align:center; margin:10px 0 16px; letter-spacing:.02em; }
    @keyframes sentinelBlink { 0%,100% { background:#5c111b; box-shadow:0 0 10px rgba(255,40,60,.22); } 50% { background:#b91f32; box-shadow:0 0 24px rgba(255,40,60,.52); } }
    .status-card { padding:15px 17px; border:1px solid #2b4057; border-radius:14px; background:rgba(17,29,46,.72); min-height:105px; }
    .status-label { font-size:11px; text-transform:uppercase; letter-spacing:.12em; color:#91a4be; font-weight:700; margin-bottom:7px; }
    .status-value { font-size:17px; color:#f1f6ff; font-weight:700; }
    .status-small { color:#9eafc7; font-size:12px; margin-top:5px; }
    .feature-card { padding:18px; min-height:148px; border:1px solid #2a3b53; border-radius:16px; background:linear-gradient(145deg,rgba(20,32,51,.92),rgba(13,22,37,.9)); margin-bottom:10px; }
    .feature-icon { font-size:22px; margin-bottom:12px; }
    .feature-title { font-weight:700; color:#edf4ff; font-size:15px; margin-bottom:6px; }
    .feature-copy { color:#9eafc7; font-size:12px; line-height:1.55; }
    .footer-note { color:#74859d; font-size:11px; text-align:center; padding:12px 0 2px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- MODEL LOADING ----------------

@st.cache_resource
def load_person_model():
    if not PERSON_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Person model not found: {PERSON_MODEL_PATH}"
        )

    return YOLO(str(PERSON_MODEL_PATH))


@st.cache_resource
def load_ppe_model():
    if not PPE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"PPE model not found: {PPE_MODEL_PATH}"
        )

    return YOLO(str(PPE_MODEL_PATH))


# ---------------- HEADER ----------------

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Construction intelligence · Safety operations</div>
      <div class="hero-title">SMART SAFETY SENTINEL</div>
      <p class="hero-subtitle">AI-assisted site monitoring for worker detection, PPE visibility, and faster review of potential safety risks.</p>
      <div class="hero-pills"><span class="pill">● Vision AI</span><span class="pill">YOLO + ByteTrack</span><span class="pill">Live & recorded analysis</span><span class="pill">Human review recommended</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- NAVIGATION ----------------

with st.sidebar:
    st.markdown("""<div style="padding:10px 4px 20px"><div style="font-size:10px;letter-spacing:.18em;color:#42d6c5;font-weight:800">SENTINEL / CONTROL</div><div style="font-family:'Space Grotesk',sans-serif;font-size:21px;font-weight:700;color:#f3f7ff;margin-top:7px">Safety Console</div><div style="font-size:12px;color:#8193ad;margin-top:5px">Construction site monitoring</div></div>""", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;letter-spacing:.14em;color:#70829c;font-weight:700;margin:0 0 8px 9px'>WORKSPACE</div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Overview", "Video Analysis", "Live Webcam Monitoring", "AI Model Status", "Project Information"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("<div class='status-card'><div class='status-label'>System profile</div><div class='status-value'>Phase 3 · Monitoring</div><div class='status-small'>Local inference · CPU/GPU depends on device</div></div>", unsafe_allow_html=True)
    st.caption("Safety alerts are model predictions and should be reviewed by a person.")


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown('<div class="section-kicker">Operations overview</div>', unsafe_allow_html=True)
    st.subheader("Monitoring at a glance")
    st.markdown('<div class="section-note">Choose a workflow below to inspect a recording or start a live monitoring session.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="status-card"><div class="status-label">Worker analytics</div><div class="status-value">YOLO11n + ByteTrack</div><div class="status-small">Person detection and temporary track IDs</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="status-card"><div class="status-label">PPE intelligence</div><div class="status-value">Custom hazard model</div><div class="status-small">Detects supported PPE and missing-PPE labels</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="status-card"><div class="status-label">Analysis modes</div><div class="status-value">Video + Live camera</div><div class="status-small">Local webcam, USB camera, or RTSP stream</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Quick actions</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="feature-card"><div class="feature-icon">🎞️</div><div class="feature-title">Review a recorded site video</div><div class="feature-copy">Upload MP4, AVI, MOV, or MKV footage, run detection, review the annotated output, and export detection details.</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="feature-card"><div class="feature-icon">📡</div><div class="feature-title">Monitor a live camera</div><div class="feature-copy">Run a time-limited monitoring session with worker tracking, frame-level PPE alerts, and a downloadable event table.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Important interpretation</div>', unsafe_allow_html=True)
    st.info("Worker IDs are tracking estimates, not verified identities. PPE detections are individual object boxes and are not yet assigned to a specific worker. A missing-PPE label is an AI prediction that requires review.")


# =========================================================
# VIDEO ANALYSIS
# =========================================================

elif page == "Video Analysis":

    st.markdown('<div class="section-kicker">Recorded footage</div>', unsafe_allow_html=True)
    st.subheader("Video analysis")
    st.markdown('<div class="section-note">Inspect construction footage, review frame-level detections, and export results.</div>', unsafe_allow_html=True)

    st.write(
        "Upload a video to detect workers and safety equipment."
    )

    uploaded_file = st.file_uploader(
        "Choose a construction video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    confidence = st.slider(
        "Detection confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05
    )

    max_width = st.select_slider(
        "Processing resolution",
        options=[480, 640, 800, 960],
        value=640
    )

    if uploaded_file is not None:

        st.video(uploaded_file)

        run_analysis = st.button(
            "🚀 Start AI Analysis",
            type="primary"
        )

        if run_analysis:

            progress = st.progress(0)
            status = st.empty()

            input_path = None
            output_path = None

            try:

                # Load models
                status.info("Loading AI models...")

                person_model = load_person_model()
                ppe_model = load_ppe_model()

                # Display PPE model classes
                st.subheader("🦺 PPE Model Classes")

                if ppe_model is not None:
                    class_names = ppe_model.names

                    for class_id, class_name in class_names.items():
                        st.write(f"Class {class_id}: {class_name}")

                # Save uploaded video temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_file.name).suffix
                ) as temp_file:

                    temp_file.write(uploaded_file.getbuffer())
                    input_path = Path(temp_file.name)

                # Open video
                cap = cv2.VideoCapture(str(input_path))

                if not cap.isOpened():
                    raise RuntimeError(
                        "Unable to open the uploaded video."
                    )

                fps = cap.get(cv2.CAP_PROP_FPS)

                if fps <= 0:
                    fps = 25.0

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                original_width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )

                original_height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                if total_frames <= 0:
                    total_frames = 1

                scale = min(
                    1.0,
                    max_width / max(original_width, 1)
                )

                output_width = max(
                    1, int(original_width * scale)
                )

                output_height = max(
                    1, int(original_height * scale)
                )

                output_path = OUTPUT_DIR / "annotated_video.mp4"

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    fps,
                    (output_width, output_height)
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        "Unable to create the output video."
                    )

                frame_number = 0
                total_person_detections = 0
                total_ppe_detections = 0
                max_persons_in_frame = 0
                max_ppe_boxes_in_frame = 0
                unique_track_ids = set()
                event_rows = []

                status.info(
                    "Analyzing frames. Processing time depends "
                    "on video length and computer performance."
                )

                # Process video frame by frame
                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    frame_number += 1

                    if scale < 1.0:
                        frame = cv2.resize(
                            frame,
                            (output_width, output_height)
                        )

                    # Person detection and tracking
                    person_results = person_model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        classes=[0],
                        conf=confidence,
                        verbose=False
                    )

                    # Separate PPE detection
                    ppe_results = ppe_model.predict(
                        frame,
                        conf=confidence,
                        verbose=False
                    )

                    annotated = frame.copy()
                    frame_has_violation = False
                    persons_this_frame = 0
                    ppe_boxes_this_frame = 0

                    # Draw person detections
                    for result in person_results:

                        boxes = result.boxes

                        if boxes is None:
                            continue

                        for box in boxes:

                            xyxy = box.xyxy[0].cpu().numpy().astype(int)

                            x1, y1, x2, y2 = xyxy

                            total_person_detections += 1
                            persons_this_frame += 1

                            track_id = None

                            if (
                                box.id is not None
                                and len(box.id) > 0
                            ):
                                track_id = int(
                                    box.id[0].item()
                                )

                                unique_track_ids.add(track_id)

                            if track_id is not None:
                                label = f"Worker ID {track_id}"
                            else:
                                label = "Worker"

                            cv2.rectangle(
                                annotated,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                2
                            )

                            cv2.putText(
                                annotated,
                                label,
                                (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )

                    # Draw PPE detections
                    for result in ppe_results:

                        boxes = result.boxes

                        if boxes is None:
                            continue

                        names = result.names

                        for box in boxes:

                            xyxy = box.xyxy[0].cpu().numpy().astype(int)

                            x1, y1, x2, y2 = xyxy

                            class_id = int(box.cls[0].item())

                            label_name = names.get(
                                class_id,
                                str(class_id)
                            )

                            score = float(box.conf[0].item())

                            label = f"{label_name} {score:.2f}"
                            violation = is_missing_ppe(label_name)
                            if violation:
                                frame_has_violation = True

                            total_ppe_detections += 1
                            ppe_boxes_this_frame += 1
                            box_color = (0, 0, 255) if violation else (0, 165, 255)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3 if violation else 2)
                            cv2.putText(
                                annotated, label, (x1, max(25, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2
                            )

                            event_rows.append(
                                {
                                    "frame": frame_number,
                                    "timestamp_seconds": round(
                                        frame_number / fps, 2
                                    ),
                                    "class": label_name,
                                    "confidence": round(score, 3),
                                    "x1": int(x1),
                                    "y1": int(y1),
                                    "x2": int(x2),
                                    "y2": int(y2),
                                }
                            )

                    max_persons_in_frame = max(max_persons_in_frame, persons_this_frame)
                    max_ppe_boxes_in_frame = max(max_ppe_boxes_in_frame, ppe_boxes_this_frame)

                    # Flash a red frame border when any explicit missing-PPE class is detected.
                    if frame_has_violation and (frame_number // 5) % 2 == 0:
                        h, w = annotated.shape[:2]
                        cv2.rectangle(annotated, (4, 4), (w - 5, h - 5), (0, 0, 255), 10)
                        cv2.putText(annotated, "SAFETY ALERT: MISSING PPE DETECTED", (12, 32),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    writer.write(annotated)

                    if frame_number % 5 == 0:
                        progress.progress(
                            min(
                                frame_number / total_frames,
                                1.0
                            )
                        )

                cap.release()
                writer.release()

                progress.progress(1.0)

                status.success("Video analysis completed!")

                # Detection summary
                st.subheader("📊 Analysis Results")

                c1, c2, c3 = st.columns(3)
                c1.metric("Estimated Unique Workers", len(unique_track_ids))
                c2.metric("Max People in One Frame", max_persons_in_frame)
                c3.metric("Max PPE Boxes in One Frame", max_ppe_boxes_in_frame)
                st.caption(
                    "PPE boxes are individual detections (for example, helmet + vest + gloves) and can be more numerous than people. The max-per-frame metric avoids adding the same item again for every video frame. The PPE model does not yet link each item to a specific worker."
                )
                if any(is_missing_ppe(row.get("class", "")) for row in event_rows):
                    st.markdown('<div class="sentinel-alert">⚠️ SAFETY ALERT — MISSING PPE CLASS DETECTED</div>', unsafe_allow_html=True)
                    st.error("The model detected one or more missing-PPE labels. Review the annotated video; this is a model prediction, not a confirmed worker-level violation.")
                else:
                    st.success("No explicit missing-PPE class was detected in the analyzed frames. This does not guarantee full PPE compliance.")

                # Display annotated video directly in the dashboard
                st.subheader("🎬 Annotated Video")

                if output_path.exists():
                    video_bytes = output_path.read_bytes()

                    if video_bytes:
                        st.video(video_bytes, format="video/mp4")

                        st.download_button(
                            label="⬇️ Download Annotated Video",
                            data=video_bytes,
                            file_name="annotated_video.mp4",
                            mime="video/mp4",
                        )
                    else:
                        st.error("The annotated video file is empty.")
                else:
                    st.error("Annotated video file was not found.")

                # Export PPE detection details
                if event_rows:
                    st.subheader("🦺 PPE Detection Details")
                    dataframe = pd.DataFrame(event_rows)
                    st.dataframe(dataframe, use_container_width=True)

                    # -------------------------------------------------
                    # PPE CLASS SUMMARY
                    # Counts detections by model class. These are
                    # detections across frames, not unique workers.
                    # -------------------------------------------------
                    st.subheader("📋 PPE Detection Summary")

                    class_counts = (
                        dataframe["class"]
                        .value_counts()
                        .rename_axis("Class")
                        .reset_index(name="Detection Count")
                    )
                    st.dataframe(
                        class_counts,
                        use_container_width=True,
                        hide_index=True,
                    )

                    positive_ppe_classes = [
                        "hardhat",
                        "safety vest",
                        "gloves",
                        "safety shoes",
                        "goggles",
                    ]
                    missing_ppe_classes = [
                        "no-hardhat",
                        "no-safety vest",
                        "no-mask",
                        "no-goggles",
                        "no-gloves",
                    ]

                    positive_count = int(
                        dataframe["class"].isin(positive_ppe_classes).sum()
                    )
                    missing_count = int(
                        dataframe["class"].isin(missing_ppe_classes).sum()
                    )

                    summary_col1, summary_col2 = st.columns(2)
                    summary_col1.metric(
                        "Protective Equipment Detections",
                        positive_count,
                    )
                    summary_col2.metric(
                        "Missing PPE Detections",
                        missing_count,
                    )

                    st.caption(
                        "These are object detections across video frames, "
                        "not counts of unique workers or confirmed violations. "
                        "A missing-PPE class is a model prediction and should "
                        "be reviewed before taking safety action."
                    )

                    csv_data = dataframe.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download PPE Detection CSV",
                        data=csv_data,
                        file_name="ppe_detection_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning(
                        "No PPE objects were detected at the selected confidence "
                        "threshold. This does not prove that workers lack PPE."
                    )

            except Exception as error:
                st.error(f"Video analysis failed: {error}")

            finally:
                # Release temporary resources after errors too
                if "cap" in locals() and cap.isOpened():
                    cap.release()

                if "writer" in locals() and writer.isOpened():
                    writer.release()

                if input_path is not None:
                    input_path.unlink(missing_ok=True)

# =========================================================
# LIVE WEBCAM / USB CAMERA / CCTV MONITORING
# =========================================================

elif page == "Live Webcam Monitoring":

    st.markdown('<div class="section-kicker">Live operations</div>', unsafe_allow_html=True)
    st.subheader("Live camera monitoring")
    st.markdown('<div class="section-note">Run a bounded monitoring session and review the resulting safety detections.</div>', unsafe_allow_html=True)
    st.write(
        "Monitor live frames with person tracking and PPE detection. "
        "For local testing, start with your laptop's built-in webcam."
    )

    camera_type = st.selectbox(
        "Choose camera source",
        ["Laptop webcam", "External USB webcam", "CCTV / RTSP stream"],
    )

    source_value = None
    if camera_type == "Laptop webcam":
        source_value = 0
        st.caption("Uses the default webcam connected to the computer running Streamlit.")
    elif camera_type == "External USB webcam":
        camera_index = st.selectbox("USB camera index", [1, 2, 3], index=0)
        source_value = int(camera_index)
        st.caption("If your USB camera does not open, try another index (1, 2, or 3).")
    else:
        rtsp_url = st.text_input(
            "CCTV RTSP URL",
            placeholder="rtsp://username:password@camera-ip:554/stream1",
            help="Enter the RTSP stream URL supplied by your CCTV/NVR. Keep credentials private.",
        )
        source_value = rtsp_url.strip() if rtsp_url.strip() else None

    live_confidence = st.slider(
        "Live detection confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        key="live_confidence",
    )
    live_width = st.select_slider(
        "Live processing width",
        options=[480, 640, 800, 960],
        value=640,
        key="live_width",
    )
    duration_seconds = st.slider(
        "Monitoring duration (seconds)",
        min_value=10,
        max_value=120,
        value=30,
        step=10,
        help="This run processes the camera for a fixed duration, then releases the camera.",
    )

    st.warning(
        "Camera monitoring runs on the machine hosting this Python app. "
        "When deployed remotely, camera index 0 refers to the server's camera, "
        "not automatically your laptop webcam. Browser-camera access requires a separate integration."
    )

    start_live = st.button("▶️ Start Live Monitoring", type="primary", key="start_live_monitoring")

    if start_live:
        if source_value is None:
            st.error("Please enter a valid CCTV RTSP URL.")
        else:
            camera = None
            try:
                with st.spinner("Loading AI models..."):
                    person_model = load_person_model()
                    ppe_model = load_ppe_model()

                # CAP_FFMPEG can help with RTSP; OpenCV chooses the backend for local cameras.
                if isinstance(source_value, str):
                    camera = cv2.VideoCapture(source_value, cv2.CAP_FFMPEG)
                else:
                    camera = cv2.VideoCapture(source_value, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)

                if not camera.isOpened():
                    raise RuntimeError(
                        "Could not open this camera source. Check camera permissions, "
                        "the USB index, or the CCTV RTSP URL."
                    )

                preview = st.empty()
                live_status = st.empty()
                live_progress = st.progress(0)
                live_person_count = 0
                live_ppe_count = 0
                live_max_persons_in_frame = 0
                live_max_ppe_boxes_in_frame = 0
                live_track_ids = set()
                live_rows = []
                frame_number = 0
                start_time = __import__("time").monotonic()
                time_module = __import__("time")

                live_status.info(f"Monitoring for up to {duration_seconds} seconds...")

                while time_module.monotonic() - start_time < duration_seconds:
                    ok, frame = camera.read()
                    if not ok:
                        live_status.warning("No frame received. The stream may have disconnected.")
                        break

                    frame_number += 1
                    height, width = frame.shape[:2]
                    scale = min(1.0, live_width / max(width, 1))
                    if scale < 1.0:
                        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

                    person_results = person_model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        classes=[0],
                        conf=live_confidence,
                        verbose=False,
                    )
                    ppe_results = ppe_model.predict(
                        frame,
                        conf=live_confidence,
                        verbose=False,
                    )
                    annotated = frame.copy()
                    frame_has_violation = False
                    persons_this_frame = 0
                    ppe_boxes_this_frame = 0

                    for result in person_results:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            live_person_count += 1
                            persons_this_frame += 1
                            track_id = None
                            if box.id is not None and len(box.id) > 0:
                                track_id = int(box.id[0].item())
                                live_track_ids.add(track_id)
                            label = f"Worker ID {track_id}" if track_id is not None else "Worker"
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                annotated, label, (x1, max(25, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                            )

                    for result in ppe_results:
                        if result.boxes is None:
                            continue
                        names = result.names
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            class_id = int(box.cls[0].item())
                            class_name = names.get(class_id, str(class_id))
                            score = float(box.conf[0].item())
                            live_ppe_count += 1
                            ppe_boxes_this_frame += 1
                            violation = is_missing_ppe(class_name)
                            if violation:
                                frame_has_violation = True
                            box_color = (0, 0, 255) if violation else (0, 165, 255)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3 if violation else 2)
                            cv2.putText(
                                annotated, f"{class_name} {score:.2f}",
                                (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, box_color, 2,
                            )
                            live_rows.append({
                                "frame": frame_number,
                                "elapsed_seconds": round(time_module.monotonic() - start_time, 2),
                                "class": class_name,
                                "confidence": round(score, 3),
                                "x1": int(x1), "y1": int(y1),
                                "x2": int(x2), "y2": int(y2),
                            })

                    live_max_persons_in_frame = max(live_max_persons_in_frame, persons_this_frame)
                    live_max_ppe_boxes_in_frame = max(live_max_ppe_boxes_in_frame, ppe_boxes_this_frame)

                    # Flash a red border on alternating frames while a missing-PPE label is present.
                    if frame_has_violation:
                        live_status.error("⚠️ SAFETY ALERT: Missing-PPE class detected in current frame")
                        if (frame_number // 3) % 2 == 0:
                            h, w = annotated.shape[:2]
                            cv2.rectangle(annotated, (4, 4), (w - 5, h - 5), (0, 0, 255), 10)
                            cv2.putText(annotated, "SAFETY ALERT: MISSING PPE", (12, 32),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        live_status.info(f"Monitoring for up to {duration_seconds} seconds...")

                    # Streamlit image expects RGB, while OpenCV frames are BGR.
                    preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    elapsed = time_module.monotonic() - start_time
                    live_progress.progress(min(elapsed / duration_seconds, 1.0))

                if camera is not None:
                    camera.release()
                    camera = None
                live_progress.progress(1.0)
                live_status.success("Live monitoring session finished and camera released.")

                m1, m2, m3 = st.columns(3)
                m1.metric("Estimated Unique Workers", len(live_track_ids))
                m2.metric("Max People in One Frame", live_max_persons_in_frame)
                m3.metric("Max PPE Boxes in One Frame", live_max_ppe_boxes_in_frame)
                st.caption(
                    "PPE boxes count detected items, not people. One worker may have several PPE boxes, and the same item is detected again in later frames. These metrics show the maximum in a single frame; PPE items are not yet linked to specific workers."
                )
                if any(is_missing_ppe(row.get("class", "")) for row in live_rows):
                    st.markdown('<div class="sentinel-alert">⚠️ SAFETY ALERT — MISSING PPE CLASS DETECTED</div>', unsafe_allow_html=True)
                    st.error("Review the red boxes and video frames. The alert is based on model predictions and is not a confirmed worker-level violation.")

                if live_rows:
                    live_df = pd.DataFrame(live_rows)
                    st.subheader("🦺 Live PPE Detection Details")
                    st.dataframe(live_df, use_container_width=True, hide_index=True)
                    summary = live_df["class"].value_counts().rename_axis("Class").reset_index(name="Detection Count")
                    st.subheader("📋 Live PPE Detection Summary")
                    st.dataframe(summary, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ Download Live Detection CSV",
                        data=live_df.to_csv(index=False).encode("utf-8"),
                        file_name="live_ppe_detection_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No PPE detections were recorded during this session. This does not prove workers lack PPE.")

            except Exception as error:
                st.error(f"Live monitoring failed: {error}")
            finally:
                if camera is not None:
                    camera.release()


# =========================================================
# AI MODEL STATUS
# =========================================================

elif page == "AI Model Status":

    st.markdown('<div class="section-kicker">System diagnostics</div>', unsafe_allow_html=True)
    st.subheader("AI model status")
    st.markdown('<div class="section-note">Load each model to confirm that the local weights are accessible.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Person Detection")

        if st.button("Load Person Model"):

            try:

                with st.spinner("Loading person model..."):
                    model = load_person_model()

                st.success("Person model loaded")

                st.json(model.names)

            except Exception as error:

                st.error(str(error))

    with col2:

        st.markdown("### PPE Detection")

        if st.button("Load PPE Model"):

            try:

                with st.spinner("Loading PPE model..."):
                    model = load_ppe_model()

                st.success("PPE model loaded")

                st.json(model.names)

            except Exception as error:

                st.error(str(error))


# =========================================================
# PROJECT INFORMATION
# =========================================================

elif page == "Project Information":

    st.markdown('<div class="section-kicker">Documentation</div>', unsafe_allow_html=True)
    st.subheader("About Smart Safety Sentinel")

    st.write(
        """
        Smart Safety Sentinel is a computer-vision prototype
        that analyzes construction videos to detect people
        and personal protective equipment.
        """
    )

    st.markdown(
        """
        **Technology stack**

        - Python
        - YOLO11n
        - Custom PPE detection model
        - ByteTrack
        - OpenCV
        - Streamlit
        - Pandas

        **Planned future features**

        - Worker-level PPE compliance analysis
        - Repeated-violation alerts
        - SQLite event history
        - Risk summaries
        - Safety analytics and reports
        """
    )

    st.warning(
        "PPE object detections are not yet linked to individual "
        "workers. Compliance conclusions will be added in a "
        "later phase and require careful validation."
    )


# ---------------- FOOTER ----------------

st.divider()

st.markdown('<div class="footer-note">SMART SAFETY SENTINEL · PHASE 3 · AI-ASSISTED CONSTRUCTION SAFETY MONITORING</div>', unsafe_allow_html=True)
