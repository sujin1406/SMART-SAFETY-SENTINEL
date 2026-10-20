from pathlib import Path
import tempfile
from collections import defaultdict, deque
from datetime import datetime
from html import escape
import sqlite3
import time

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration
import av


# =========================================================
# SMART SAFETY SENTINEL
# PHASE 8: SIMPLIFIED DASHBOARD
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

PERSON_MODEL_PATH = BASE_DIR / "yolo11n.pt"
PPE_MODEL_PATH = BASE_DIR / "models" / "buildvision-hazards-v0.3.pt"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
INCIDENT_DB_PATH = OUTPUT_DIR / "sentinel_incidents.db"
EVIDENCE_DIR = OUTPUT_DIR / "incident_evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)


def init_incident_db():
    """Create the local incident history table if it does not exist."""
    with sqlite3.connect(INCIDENT_DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                worker_id TEXT,
                violation TEXT NOT NULL,
                confidence REAL,
                evidence_path TEXT,
                status TEXT NOT NULL DEFAULT 'Open',
                notes TEXT DEFAULT ''
            )
        """)
        connection.commit()


def record_incident(source, worker_id, violation, confidence, frame, session_key):
    """Save one incident per worker/violation per monitoring session, with evidence."""
    dedupe_key = (str(worker_id) if worker_id is not None else "unassigned", str(violation))
    if dedupe_key in session_key:
        return None
    session_key.add(dedupe_key)
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    evidence_path = EVIDENCE_DIR / f"incident_{stamp}.jpg"
    saved = cv2.imwrite(str(evidence_path), frame)
    if not saved:
        evidence_path = None
    with sqlite3.connect(INCIDENT_DB_PATH) as connection:
        cursor = connection.execute(
            "INSERT INTO incidents (created_at, source, worker_id, violation, confidence, evidence_path, status) VALUES (?, ?, ?, ?, ?, ?, 'Open')",
            (created_at, source, str(worker_id) if worker_id is not None else "Unassigned", str(violation), float(confidence) if confidence is not None else None, str(evidence_path) if evidence_path else None),
        )
        connection.commit()
        return cursor.lastrowid


def load_incidents():
    init_incident_db()
    with sqlite3.connect(INCIDENT_DB_PATH) as connection:
        return pd.read_sql_query("SELECT * FROM incidents ORDER BY id DESC", connection)


def update_incident_status(incident_id, status, notes):
    with sqlite3.connect(INCIDENT_DB_PATH) as connection:
        connection.execute("UPDATE incidents SET status = ?, notes = ? WHERE id = ?", (status, notes, int(incident_id)))
        connection.commit()


init_incident_db()

# Only these classes explicitly indicate a missing item in this model.
# This is a frame-level alert: PPE boxes are not reliably assigned to a worker yet.
MISSING_PPE_CLASSES = {"no-hardhat", "no-safety vest", "no-mask", "no-goggles", "no-gloves"}

def is_missing_ppe(class_name):
    """Return True only for model labels explicitly indicating missing PPE."""
    normalized = " ".join(str(class_name).lower().replace("_", "-").split())
    return normalized in MISSING_PPE_CLASSES


def box_iou(a, b):
    """Intersection-over-union for xyxy boxes."""
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def associate_ppe_to_person(ppe_box, people):
    """Associate a PPE box to the most plausible person using geometry.

    `people` is a list of dictionaries with `box` and `track_id`. This is a
    heuristic, not identity-level proof; overlap/crowding can cause mistakes.
    """
    px1, py1, px2, py2 = map(float, ppe_box)
    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
    best_person, best_score = None, -1.0
    for person in people:
        x1, y1, x2, y2 = map(float, person["box"])
        # Extend the person box slightly above the head and sideways for PPE boxes.
        width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
        ex1, ey1, ex2, ey2 = x1 - 0.08 * width, y1 - 0.18 * height, x2 + 0.08 * width, y2 + 0.03 * height
        center_inside = ex1 <= pcx <= ex2 and ey1 <= pcy <= ey2
        overlap = box_iou(ppe_box, (x1, y1, x2, y2))
        if center_inside or overlap >= 0.01:
            score = (2.0 if center_inside else 0.0) + overlap
            if score > best_score:
                best_person, best_score = person, score
    return best_person



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
        ["Overview", "Video Analysis", "Live Webcam Monitoring", "Supervisor Dashboard", "Incident Management", "Safety Analytics", "Safety Reports", "AI Model Status", "Project Information"],
        label_visibility="collapsed",
    )
    st.divider()
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
# SUPERVISOR DASHBOARD — PHASE 7
# =========================================================

if page == "Supervisor Dashboard":
    st.markdown('<div class="section-kicker">Supervisor workspace</div>', unsafe_allow_html=True)
    st.subheader("Site safety oversight")
    st.markdown('<div class="section-note">A concise overview of recorded AI flags, review workload, and recurring PPE labels. All figures reflect the incident database and should be verified against evidence.</div>', unsafe_allow_html=True)

    supervisor_df = load_incidents()
    if supervisor_df.empty:
        st.info("No incident records are available yet. Run Video Analysis or Live Webcam Monitoring to populate the supervisor dashboard.")
    else:
        supervisor_df["created_at"] = pd.to_datetime(supervisor_df["created_at"], errors="coerce", utc=True)
        supervisor_df["status"] = supervisor_df["status"].fillna("Open").astype(str)
        supervisor_df["violation"] = supervisor_df["violation"].fillna("Unknown").astype(str)
        supervisor_df["worker_id"] = supervisor_df["worker_id"].fillna("Unassigned").astype(str)
        supervisor_df["source"] = supervisor_df["source"].fillna("Unknown").astype(str)
        valid_dates = supervisor_df["created_at"].dropna().dt.date
        if not valid_dates.empty:
            d1, d2 = st.columns(2)
            with d1:
                supervisor_start = st.date_input("From date", value=min(valid_dates), min_value=min(valid_dates), max_value=max(valid_dates), key="supervisor_from")
            with d2:
                supervisor_end = st.date_input("To date", value=max(valid_dates), min_value=min(valid_dates), max_value=max(valid_dates), key="supervisor_to")
            if supervisor_start > supervisor_end:
                st.error("The start date must be on or before the end date.")
                st.stop()
            supervisor_df = supervisor_df[supervisor_df["created_at"].dt.date.between(supervisor_start, supervisor_end)].copy()

        sf1, sf2 = st.columns(2)
        with sf1:
            selected_sources = st.multiselect("Monitoring source", sorted(supervisor_df["source"].unique()), default=sorted(supervisor_df["source"].unique()), key="supervisor_sources")
        with sf2:
            selected_statuses = st.multiselect("Incident status", ["Open", "Acknowledged", "Resolved"], default=["Open", "Acknowledged", "Resolved"], key="supervisor_statuses")
        supervisor_df = supervisor_df[supervisor_df["source"].isin(selected_sources) & supervisor_df["status"].isin(selected_statuses)].copy()

        total_flags = len(supervisor_df)
        open_flags = int((supervisor_df["status"] == "Open").sum())
        acknowledged_flags = int((supervisor_df["status"] == "Acknowledged").sum())
        resolved_flags = int((supervisor_df["status"] == "Resolved").sum())
        resolution_rate = (resolved_flags / total_flags * 100) if total_flags else 0.0
        worker_count = supervisor_df.loc[supervisor_df["worker_id"].ne("Unassigned"), "worker_id"].nunique()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Recorded flags", total_flags)
        m2.metric("Open review", open_flags)
        m3.metric("Acknowledged", acknowledged_flags)
        m4.metric("Resolved", resolved_flags)
        m5.metric("Tracked worker IDs", worker_count)
        st.caption(f"Incident resolution rate: {resolution_rate:.1f}% — a workflow metric, not a measure of overall site safety.")

        st.markdown("### Priority review queue")
        priority_queue = supervisor_df[supervisor_df["status"].isin(["Open", "Acknowledged"])].copy()
        if not priority_queue.empty:
            priority_queue["priority"] = priority_queue["status"].map({"Open": 0, "Acknowledged": 1}).fillna(2)
            priority_queue = priority_queue.sort_values(["priority", "created_at"], ascending=[True, False])
            priority_cols = [c for c in ["id", "created_at", "worker_id", "violation", "confidence", "status", "source", "notes"] if c in priority_queue.columns]
            priority_view = priority_queue[priority_cols].copy()
            priority_view["created_at"] = priority_view["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            st.dataframe(priority_view, use_container_width=True, hide_index=True)
            st.download_button("Download priority review queue (CSV)", data=priority_view.to_csv(index=False).encode("utf-8"), file_name="sentinel_priority_review_queue.csv", mime="text/csv")
        else:
            st.success("No open or acknowledged records match the current filters.")

        left, right = st.columns(2)
        with left:
            st.markdown("### Most frequent violation labels")
            violation_counts = supervisor_df["violation"].value_counts().rename_axis("Violation").reset_index(name="Recorded flags")
            if violation_counts.empty:
                st.info("No violations match these filters.")
            else:
                st.bar_chart(violation_counts.set_index("Violation"), use_container_width=True)
        with right:
            st.markdown("### Review status distribution")
            status_counts = supervisor_df["status"].value_counts().reindex(["Open", "Acknowledged", "Resolved"], fill_value=0)
            if int(status_counts.sum()) == 0:
                st.info("No statuses match these filters.")
            else:
                st.bar_chart(status_counts, use_container_width=True)

        st.markdown("### Worker-ID incident history")
        st.caption("Worker IDs are temporary tracker IDs, not verified identities. They may change between sessions and must not be used to identify a person.")
        assigned = supervisor_df[supervisor_df["worker_id"].ne("Unassigned")]
        if assigned.empty:
            st.info("No incidents with an assigned worker tracking ID are available for these filters.")
        else:
            worker_summary = (assigned.groupby("worker_id").agg(Recorded_flags=("id", "count"), Open=("status", lambda x: int((x == "Open").sum())), Acknowledged=("status", lambda x: int((x == "Acknowledged").sum())), Resolved=("status", lambda x: int((x == "Resolved").sum())), Latest_flag=("created_at", "max")).reset_index().sort_values("Recorded_flags", ascending=False))
            worker_summary["Latest_flag"] = worker_summary["Latest_flag"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            st.dataframe(worker_summary, use_container_width=True, hide_index=True)
            chosen_worker = st.selectbox("Inspect a tracked worker ID", worker_summary["worker_id"].tolist(), key="supervisor_worker_detail")
            worker_detail = assigned[assigned["worker_id"] == chosen_worker].copy()
            detail_cols = [c for c in ["id", "created_at", "violation", "status", "confidence", "source", "notes"] if c in worker_detail.columns]
            worker_detail = worker_detail[detail_cols].sort_values("created_at", ascending=False)
            worker_detail["created_at"] = worker_detail["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            st.dataframe(worker_detail, use_container_width=True, hide_index=True)

        st.warning("AI-generated flags can be false positives or incorrectly associated with a worker. Review saved evidence and the original footage before making personnel or compliance decisions.")


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

                # Open the uploaded recording (not the webcam).
                cap = cv2.VideoCapture(str(input_path))
                if not cap.isOpened():
                    cap.release()
                    raise RuntimeError("Unable to open the uploaded video. Try an MP4/H.264 file.")

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
                video_violation_history = defaultdict(lambda: deque(maxlen=5))
                event_rows = []
                recorded_incidents = set()

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

                    # Collect tracked people for worker-level PPE association.
                    people_this_frame = []
                    visible_track_ids = set()
                    for result in person_results:
                        boxes = result.boxes
                        if boxes is None:
                            continue
                        for box in boxes:
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            x1, y1, x2, y2 = map(int, xyxy)
                            track_id = None
                            if box.id is not None and len(box.id) > 0:
                                track_id = int(box.id[0].item())
                                unique_track_ids.add(track_id)
                                visible_track_ids.add(track_id)
                            people_this_frame.append({"box": (x1, y1, x2, y2), "track_id": track_id, "violation": False})
                            persons_this_frame += 1
                            label = f"Worker ID {track_id}" if track_id is not None else "Worker (untracked)"
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Collect PPE detections and associate them with a person when possible.
                    violating_track_ids = set()
                    unassigned_violation = False
                    for result in ppe_results:
                        boxes = result.boxes
                        if boxes is None:
                            continue
                        names = result.names
                        for box in boxes:
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            x1, y1, x2, y2 = map(int, xyxy)
                            class_id = int(box.cls[0].item())
                            label_name = names.get(class_id, str(class_id))
                            score = float(box.conf[0].item())
                            violation = is_missing_ppe(label_name)
                            assigned_person = associate_ppe_to_person((x1, y1, x2, y2), people_this_frame)
                            assigned_id = assigned_person["track_id"] if assigned_person else None
                            if violation:
                                frame_has_violation = True
                                if assigned_person is not None:
                                    assigned_person["violation"] = True
                                    if assigned_id is not None:
                                        violating_track_ids.add(assigned_id)
                                else:
                                    unassigned_violation = True
                            total_ppe_detections += 1
                            ppe_boxes_this_frame += 1
                            box_color = (0, 0, 255) if violation else (0, 165, 255)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3 if violation else 2)
                            tag = f"{label_name} {score:.2f}"
                            if assigned_id is not None:
                                tag += f" | W{assigned_id}"
                            cv2.putText(annotated, tag, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            event_rows.append({
                                "frame": frame_number,
                                "timestamp_seconds": round(frame_number / fps, 2),
                                "class": label_name,
                                "confidence": round(score, 3),
                                "associated_worker_id": assigned_id,
                                "association_status": "heuristic match" if assigned_id is not None else "unassigned",
                                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                            })

                    # Confirm worker-level warnings only after repeated detections.
                    for tid in visible_track_ids:
                        video_violation_history[tid].append(tid in violating_track_ids)
                    confirmed_track_ids = {tid for tid in visible_track_ids if sum(video_violation_history[tid]) >= 3}
                    for person in people_this_frame:
                        x1, y1, x2, y2 = person["box"]
                        tid = person["track_id"]
                        confirmed = tid is not None and tid in confirmed_track_ids
                        if confirmed:
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            cv2.putText(annotated, "PPE VIOLATION - REVIEW", (x1, min(annotated.shape[0]-8, y2+22)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                    # Persist confirmed worker-level violations once per worker/class in this run.
                    for tid in confirmed_track_ids:
                        classes_for_worker = [row["class"] for row in event_rows if row.get("associated_worker_id") == tid and is_missing_ppe(row.get("class", ""))]
                        violation_name = ", ".join(sorted(set(classes_for_worker))) or "Missing PPE (repeated detection)"
                        record_incident("Video: " + uploaded_file.name, tid, violation_name, None, annotated, recorded_incidents)
                    if unassigned_violation:
                        unassigned_classes = [row["class"] for row in event_rows[-max(1, ppe_boxes_this_frame):] if is_missing_ppe(row.get("class", "")) and row.get("associated_worker_id") is None]
                        if unassigned_classes:
                            record_incident("Video: " + uploaded_file.name, None, ", ".join(sorted(set(unassigned_classes))), None, annotated, recorded_incidents)

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
                c1.metric("Unique Tracked IDs", len(unique_track_ids))
                c2.metric("Max People in One Frame", max_persons_in_frame)
                c3.metric("Max PPE Boxes in One Frame", max_ppe_boxes_in_frame)
                st.caption(
                    "PPE boxes are object detections, not worker counts. PPE-to-worker matching uses bounding-box geometry and can be wrong when workers overlap; IDs are temporary tracking IDs. Worker warning requires a missing-PPE label in at least 3 of the last 5 visible frames."
                )
                if any(is_missing_ppe(row.get("class", "")) for row in event_rows):
                    st.markdown('<div class="sentinel-alert">⚠️ SAFETY ALERT — MISSING PPE CLASS DETECTED</div>', unsafe_allow_html=True)
                    st.error("The model detected one or more missing-PPE labels. Review the annotated video. Worker boxes turn red only after repeated detections associated with that tracked person; unmatched detections remain frame-level alerts.")
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
# LIVE WEBCAM MONITORING (BROWSER CAMERA VIA WEBRTC)
# =========================================================

elif page == "Live Webcam Monitoring":

    st.markdown('<div class="section-kicker">Live operations</div>', unsafe_allow_html=True)
    st.subheader("Live camera monitoring")
    st.markdown('<div class="section-note">Use this page from a browser to grant access to the webcam on the device you are using.</div>', unsafe_allow_html=True)
    st.write("The browser sends camera frames to this Streamlit app using WebRTC. The YOLO models process the frames on the Render server.")

    live_confidence = st.slider(
        "Detection confidence threshold", min_value=0.10, max_value=0.90,
        value=0.35, step=0.05, key="browser_live_confidence"
    )
    live_width = st.select_slider(
        "Live processing width", options=[320, 480, 640, 800],
        value=480, key="browser_live_width"
    )

    st.info("Open this page using HTTPS in Chrome or Edge, click START, and allow camera permission when your browser asks. The first frame may take longer while the models load.")

    class BrowserPPEProcessor(VideoProcessorBase):
        def __init__(self):
            self.frame_count = 0
            self.last_error = None

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            h, w = image.shape[:2]
            scale = min(1.0, float(live_width) / max(w, 1))
            if scale < 1.0:
                image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))

            annotated = image.copy()
            try:
                person_model = load_person_model()
                ppe_model = load_ppe_model()
                person_results = person_model.predict(image, classes=[0], conf=live_confidence, verbose=False)
                ppe_results = ppe_model.predict(image, conf=live_confidence, verbose=False)
                people = []
                for result in person_results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        people.append({"box": (int(x1), int(y1), int(x2), int(y2)), "track_id": None})
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        cv2.putText(annotated, "Person", (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

                violations = 0
                for result in ppe_results:
                    if result.boxes is None:
                        continue
                    names = result.names
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        class_id = int(box.cls[0].item())
                        class_name = str(names.get(class_id, class_id))
                        score = float(box.conf[0].item())
                        missing = is_missing_ppe(class_name)
                        color = (0, 0, 255) if missing else (0, 165, 255)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, f"{class_name} {score:.2f}", (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
                        if missing:
                            violations += 1

                if violations:
                    ah, aw = annotated.shape[:2]
                    cv2.rectangle(annotated, (3, 3), (aw - 4, ah - 4), (0, 0, 255), 7)
                    cv2.rectangle(annotated, (0, 0), (aw, min(42, ah)), (0, 0, 170), -1)
                    cv2.putText(annotated, f"SAFETY REVIEW: {violations} missing-PPE label(s)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            except Exception as exc:
                self.last_error = str(exc)
                cv2.putText(annotated, "Detection error - check app logs", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
    rtc_configuration = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]}
        ]
    })
    webrtc_ctx = webrtc_streamer(
        key="sentinel-browser-webcam",
        video_processor_factory=BrowserPPEProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.caption("Green boxes indicate person detections. PPE boxes are colored red when the model label explicitly indicates missing PPE; orange boxes indicate other PPE detections. These are AI-assisted detections, not proof of compliance. PPE-to-worker assignment is not confirmed in this browser preview.")
    if webrtc_ctx.state.playing:
        st.success("Browser webcam is active. If no video appears, check browser camera permissions and reload the page.")
    else:
        st.info("Click START above to begin browser webcam monitoring.")


# =========================================================
# INCIDENT MANAGEMENT
# =========================================================

elif page == "Incident Management":
    st.markdown('<div class="section-kicker">Safety event register</div>', unsafe_allow_html=True)
    st.subheader("Incident management")
    st.markdown('<div class="section-note">Review recorded potential PPE incidents, inspect evidence, and update their status.</div>', unsafe_allow_html=True)
    incidents = load_incidents()
    if incidents.empty:
        st.info("No incidents have been recorded yet. Run Video Analysis or Live Webcam Monitoring; confirmed repeated missing-PPE detections will be saved here.")
    else:
        total, open_count, reviewed_count = st.columns(3)
        total.metric("Recorded incidents", len(incidents))
        open_count.metric("Open", int((incidents["status"] == "Open").sum()))
        reviewed_count.metric("Acknowledged / Resolved", int(incidents["status"].isin(["Acknowledged", "Resolved"]).sum()))
        status_filter = st.selectbox("Filter by status", ["All", "Open", "Acknowledged", "Resolved"])
        shown = incidents if status_filter == "All" else incidents[incidents["status"] == status_filter]
        st.dataframe(shown.drop(columns=["evidence_path"], errors="ignore"), use_container_width=True, hide_index=True)
        csv_bytes = incidents.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export incident history (CSV)", csv_bytes, "sentinel_incidents.csv", "text/csv")
        st.markdown("### Review an incident")
        incident_options = {f"#{int(row.id)} · {row.created_at} · {row.violation}": int(row.id) for row in incidents.itertuples()}
        selected_label = st.selectbox("Choose incident", list(incident_options.keys()))
        selected_id = incident_options[selected_label]
        selected = incidents[incidents["id"] == selected_id].iloc[0]
        evidence = selected.get("evidence_path")
        if isinstance(evidence, str) and evidence and Path(evidence).exists():
            st.image(evidence, caption=f"Evidence for incident #{selected_id}", use_container_width=True)
            st.download_button("⬇️ Download evidence image", Path(evidence).read_bytes(), f"incident_{selected_id}.jpg", "image/jpeg")
        else:
            st.caption("No evidence image is available for this incident.")
        with st.form("incident_review_form"):
            new_status = st.selectbox("Status", ["Open", "Acknowledged", "Resolved"], index=["Open", "Acknowledged", "Resolved"].index(str(selected["status"])) if str(selected["status"]) in ["Open", "Acknowledged", "Resolved"] else 0)
            notes = st.text_area("Review notes", value=str(selected.get("notes") or ""), placeholder="Add the human review outcome or corrective action.")
            save_review = st.form_submit_button("Save incident review", type="primary")
            if save_review:
                update_incident_status(selected_id, new_status, notes)
                st.success("Incident review saved.")
                st.rerun()
    st.caption("Incidents are stored locally in outputs/sentinel_incidents.db with evidence images in outputs/incident_evidence. Treat detections as AI-generated flags requiring human verification.")


# =========================================================
# SAFETY ANALYTICS
# =========================================================

elif page == "Safety Analytics":
    st.markdown("""
    <style>
    .analytics-chart-card { background: rgba(22, 30, 43, 0.58); border: 1px solid rgba(148,163,184,.20); border-radius: 16px; padding: 20px; min-height: 235px; }
    .analytics-bar-row { margin: 0 0 17px 0; }
    .analytics-bar-heading { display:flex; justify-content:space-between; gap:12px; align-items:center; margin-bottom:7px; color:var(--text-color); font-size:.91rem; }
    .analytics-bar-heading span { overflow-wrap:anywhere; }
    .analytics-bar-heading strong { font-variant-numeric:tabular-nums; }
    .analytics-bar-track { width:100%; height:10px; border-radius:99px; background:rgba(148,163,184,.17); overflow:hidden; }
    .analytics-bar-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#3b82f6,#22d3ee); }
    .status-stack { display:flex; height:15px; width:100%; overflow:hidden; border-radius:99px; background:rgba(148,163,184,.16); margin:4px 0 25px; }
    .status-segment { height:100%; transition:width .2s ease; }
    .status-open { background:#f87171; } .status-acknowledged { background:#fbbf24; } .status-resolved { background:#34d399; } .status-unknown { background:#94a3b8; }
    .status-row { display:flex; align-items:center; gap:10px; padding:12px 2px; border-bottom:1px solid rgba(148,163,184,.13); color:var(--text-color); }
    .status-row:last-child { border-bottom:0; }
    .status-dot { width:9px; height:9px; border-radius:50%; flex:0 0 9px; }
    .status-dot-open { background:#f87171; } .status-dot-acknowledged { background:#fbbf24; } .status-dot-resolved { background:#34d399; } .status-dot-unknown { background:#94a3b8; }
    .status-label { flex:1; } .status-share { color:#94a3b8; font-size:.86rem; min-width:55px; text-align:right; }
    .status-row strong { min-width:24px; text-align:right; font-variant-numeric:tabular-nums; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Safety intelligence</div>', unsafe_allow_html=True)
    st.subheader("Safety analytics & reporting")
    st.markdown('<div class="section-note">Explore recorded incident patterns, follow resolution progress, and export a filtered report.</div>', unsafe_allow_html=True)

    incidents = load_incidents()
    if incidents.empty:
        st.info("There is no incident history to analyze yet. Run a video or live monitoring session first.")
    else:
        incidents["created_at"] = pd.to_datetime(incidents["created_at"], errors="coerce", utc=True)
        incidents["date"] = incidents["created_at"].dt.date
        valid_dates = incidents["date"].dropna()
        if valid_dates.empty:
            filtered = incidents.copy()
        else:
            min_date, max_date = min(valid_dates), max(valid_dates)
            date_range = st.date_input("Incident date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                filtered = incidents[(incidents["date"] >= start_date) & (incidents["date"] <= end_date)].copy()
            else:
                filtered = incidents.copy()

        f1, f2 = st.columns(2)
        with f1:
            statuses = st.multiselect("Statuses", ["Open", "Acknowledged", "Resolved"], default=["Open", "Acknowledged", "Resolved"])
        with f2:
            sources = sorted(filtered["source"].dropna().astype(str).unique().tolist()) if not filtered.empty else []
            selected_sources = st.multiselect("Sources", sources, default=sources)

        if "status" in filtered.columns:
            filtered = filtered[filtered["status"].isin(statuses)]
        if "source" in filtered.columns:
            filtered = filtered[filtered["source"].astype(str).isin(selected_sources)]

        total = len(filtered)
        open_total = int((filtered["status"] == "Open").sum()) if total else 0
        resolved_total = int((filtered["status"] == "Resolved").sum()) if total else 0
        resolution_rate = (resolved_total / total * 100) if total else 0
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Incidents in filter", total)
        k2.metric("Open", open_total)
        k3.metric("Resolved", resolved_total)
        k4.metric("Resolution rate", f"{resolution_rate:.1f}%")

        if filtered.empty:
            st.warning("No incidents match the selected filters.")
        else:
            chart_left, chart_right = st.columns(2, gap="large")
            with chart_left:
                st.markdown("### Incidents by violation")
                st.caption("Ranked by the number of recorded flags")
                violation_counts = filtered["violation"].fillna("Unknown").astype(str).value_counts().sort_values(ascending=True)
                max_violation = max(int(violation_counts.max()), 1) if not violation_counts.empty else 1
                violation_html = ['<div class="analytics-chart-card">']
                for label, count in violation_counts.items():
                    pct = max(3, int(int(count) / max_violation * 100))
                    violation_html.append(
                        '<div class="analytics-bar-row">'
                        f'<div class="analytics-bar-heading"><span>{escape(str(label))}</span><strong>{int(count)}</strong></div>'
                        f'<div class="analytics-bar-track"><div class="analytics-bar-fill violation-fill" style="width:{pct}%"></div></div>'
                        '</div>'
                    )
                violation_html.append('</div>')
                st.markdown(''.join(violation_html), unsafe_allow_html=True)
            with chart_right:
                st.markdown("### Incident status")
                st.caption("Current workflow state for filtered incidents")
                status_order = ["Open", "Acknowledged", "Resolved"]
                status_counts = filtered["status"].fillna("Unknown").astype(str).value_counts()
                for status_name in status_order:
                    if status_name not in status_counts.index:
                        status_counts.loc[status_name] = 0
                extras = [name for name in status_counts.index if name not in status_order]
                ordered_status = status_order + extras
                total_status = max(int(status_counts.sum()), 1)
                segments = []
                for status_name in ordered_status:
                    count = int(status_counts.get(status_name, 0))
                    if count:
                        segments.append(f'<div class="status-segment status-{escape(status_name.lower().replace(" ", "-"))}" style="width:{count / total_status * 100:.2f}%" title="{escape(status_name)}: {count}"></div>')
                status_html = ['<div class="analytics-chart-card">', '<div class="status-stack">', *segments, '</div>']
                for status_name in ordered_status:
                    count = int(status_counts.get(status_name, 0))
                    share = count / total_status * 100 if total_status else 0
                    status_html.append(
                        '<div class="status-row">'
                        f'<span class="status-dot status-dot-{escape(status_name.lower().replace(" ", "-"))}"></span>'
                        f'<span class="status-label">{escape(status_name)}</span>'
                        f'<span class="status-share">{share:.1f}%</span>'
                        f'<strong>{count}</strong>'
                        '</div>'
                    )
                status_html.append('</div>')
                st.markdown(''.join(status_html), unsafe_allow_html=True)

            st.markdown("### Filtered incident register")
            display_df = filtered.copy()
            display_df["created_at"] = display_df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            st.dataframe(display_df.drop(columns=["evidence_path", "date"], errors="ignore"), use_container_width=True, hide_index=True)
            report_df = filtered.copy()
            report_df["created_at"] = report_df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            st.download_button(
                "⬇️ Download filtered safety report (CSV)",
                data=report_df.drop(columns=["date"], errors="ignore").to_csv(index=False).encode("utf-8"),
                file_name="sentinel_safety_analytics.csv",
                mime="text/csv",
                type="primary",
            )
        st.caption("Analytics summarize recorded AI-generated flags, not verified injury events or definitive compliance findings. Review incidents before making safety decisions.")



# =========================================================
# SAFETY REPORTS — PHASE 6
# =========================================================

elif page == "Safety Reports":
    st.markdown('<div class="section-kicker">Supervisor workspace</div>', unsafe_allow_html=True)
    st.subheader("Safety reports & monitoring history")
    st.markdown('<div class="section-note">Build a shareable report from recorded AI-generated incident flags. Select a date range, review the summary, then export a CSV or standalone HTML report.</div>', unsafe_allow_html=True)

    report_incidents = load_incidents()
    if report_incidents.empty:
        st.info("No incident records are available yet. Run Video Analysis or Live Webcam Monitoring to create records.")
    else:
        report_incidents["created_at"] = pd.to_datetime(report_incidents["created_at"], errors="coerce", utc=True)
        report_incidents["date"] = report_incidents["created_at"].dt.date
        valid_report_dates = report_incidents["date"].dropna()
        if valid_report_dates.empty:
            report_filtered = report_incidents.copy()
        else:
            earliest, latest = min(valid_report_dates), max(valid_report_dates)
            report_range = st.date_input("Report period", value=(earliest, latest), min_value=earliest, max_value=latest, key="safety_report_period")
            if isinstance(report_range, tuple) and len(report_range) == 2:
                report_start, report_end = report_range
                report_filtered = report_incidents[(report_incidents["date"] >= report_start) & (report_incidents["date"] <= report_end)].copy()
            else:
                report_filtered = report_incidents.copy()

        report_statuses = st.multiselect("Include statuses", ["Open", "Acknowledged", "Resolved"], default=["Open", "Acknowledged", "Resolved"], key="safety_report_statuses")
        report_filtered = report_filtered[report_filtered["status"].isin(report_statuses)].copy()

        total_reported = len(report_filtered)
        open_reported = int((report_filtered["status"] == "Open").sum()) if total_reported else 0
        ack_reported = int((report_filtered["status"] == "Acknowledged").sum()) if total_reported else 0
        resolved_reported = int((report_filtered["status"] == "Resolved").sum()) if total_reported else 0
        resolution_pct = (resolved_reported / total_reported * 100) if total_reported else 0.0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recorded flags", total_reported)
        m2.metric("Open", open_reported)
        m3.metric("Acknowledged", ack_reported)
        m4.metric("Resolved", f"{resolved_reported} · {resolution_pct:.1f}%")

        if report_filtered.empty:
            st.warning("No records match this report's date and status filters.")
        else:
            st.markdown("### Monitoring source summary")
            source_summary = (report_filtered.groupby("source", dropna=False)
                             .agg(Recorded_flags=("id", "count"), Open=("status", lambda x: int((x == "Open").sum())),
                                  Acknowledged=("status", lambda x: int((x == "Acknowledged").sum())),
                                  Resolved=("status", lambda x: int((x == "Resolved").sum())))
                             .reset_index().sort_values("Recorded_flags", ascending=False))
            st.dataframe(source_summary, use_container_width=True, hide_index=True)

            st.markdown("### Most frequent flagged violations")
            violation_summary = (report_filtered["violation"].fillna("Unknown").astype(str).value_counts()
                                 .rename_axis("Violation label").reset_index(name="Recorded flags"))
            st.dataframe(violation_summary, use_container_width=True, hide_index=True)

            st.markdown("### Report preview")
            preview_cols = [c for c in ["created_at", "source", "worker_id", "violation", "status", "notes"] if c in report_filtered.columns]
            preview = report_filtered[preview_cols].copy()
            preview["created_at"] = preview["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            st.dataframe(preview, use_container_width=True, hide_index=True)

            csv_export = report_filtered.drop(columns=["date"], errors="ignore").copy()
            csv_export["created_at"] = csv_export["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button("⬇️ Download detailed report (CSV)", data=csv_export.to_csv(index=False).encode("utf-8"), file_name=f"sentinel_safety_report_{stamp}.csv", mime="text/csv", type="primary")

            period_text = "Selected reporting period"
            if valid_report_dates.size:
                period_text = f"{min(report_filtered['date'])} to {max(report_filtered['date'])}" if not report_filtered.empty else "No records"
            violation_html = violation_summary.to_html(index=False, border=0, classes="report-table")
            source_html = source_summary.to_html(index=False, border=0, classes="report-table")
            records_html = preview.to_html(index=False, border=0, classes="report-table")
            generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            html_report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Smart Safety Sentinel Report</title><style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:36px auto;padding:0 20px;color:#172033}}h1{{margin-bottom:4px}}.sub{{color:#58657a}}.metrics{{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}}.metric{{border:1px solid #d7deea;border-radius:10px;padding:14px 18px;min-width:140px}}.metric strong{{display:block;font-size:24px;margin-top:6px}}table{{width:100%;border-collapse:collapse;margin:12px 0 28px;font-size:13px}}th,td{{border:1px solid #dbe2ec;padding:9px;text-align:left;overflow-wrap:anywhere}}th{{background:#edf2f8}}.notice{{background:#fff7e6;border-left:4px solid #e0a126;padding:12px}}@media print{{body{{margin:10mm;max-width:none}}}}</style></head><body><h1>Smart Safety Sentinel</h1><div class="sub">Safety monitoring report · Generated {escape(generated_at)}</div><p><b>Reporting period:</b> {escape(period_text)}</p><div class="metrics"><div class="metric">Recorded flags<strong>{total_reported}</strong></div><div class="metric">Open<strong>{open_reported}</strong></div><div class="metric">Acknowledged<strong>{ack_reported}</strong></div><div class="metric">Resolved<strong>{resolved_reported}</strong></div><div class="metric">Resolution rate<strong>{resolution_pct:.1f}%</strong></div></div><h2>Monitoring source summary</h2>{source_html}<h2>Flagged violations</h2>{violation_html}<h2>Incident register</h2>{records_html}<p class="notice"><b>Important:</b> These are AI-generated flags, not verified injury events or definitive proof of PPE non-compliance. Review video/evidence and confirm findings with a qualified person before taking action.</p><p>Use your browser's Print command to print or save this report as PDF.</p></body></html>"""
            st.download_button("⬇️ Download printable report (HTML)", data=html_report.encode("utf-8"), file_name=f"sentinel_safety_report_{stamp}.html", mime="text/html")
        st.caption("Reports are generated from the local incident database. Exported files contain incident details; store and share them responsibly.")


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

        **Current prototype features**

        - Worker detection and temporary tracking IDs
        - PPE hazard labels and review-oriented alerts
        - SQLite incident history and evidence
        - Supervisor dashboard, safety analytics, and reports
        """
    )

    st.warning(
        "PPE object detections are not yet linked to individual "
        "workers. Compliance conclusions will be added in a "
        "later phase and require careful validation."
    )


# ---------------- FOOTER ----------------

st.divider()

st.markdown('<div class="footer-note">SMART SAFETY SENTINEL · PHASE 5 · AI-ASSISTED CONSTRUCTION SAFETY MONITORING</div>', unsafe_allow_html=True)
