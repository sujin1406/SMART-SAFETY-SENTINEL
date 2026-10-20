# Smart Safety Sentinel — Professional Web UI

This is a responsive frontend built with **HTML, CSS and vanilla JavaScript**. It does not use Python to render the interface and requires no npm/build step. It includes a polished operations dashboard, live browser-camera preview, video upload/preview, worker registry, event log, CSV exports, settings, API connection controls, and browser-side visual/audio alert demos.

## Run locally

Open `index.html` in a browser. For camera access, use `localhost` or HTTPS rather than opening a `file://` URL. One easy local static server is:

```bash
npx serve .
```

Alternatively, use the VS Code **Live Server** extension.

## Important: frontend vs AI backend

The browser UI **does not run the YOLO models**. The existing Python application still needs to run the model inference. This frontend includes an API adapter, but the existing Streamlit project does not automatically expose the API routes below. A backend/API adapter must be implemented before this UI can display actual YOLO results or submit videos for analysis. Until connected, camera/video preview, navigation, settings, CSV export, and the test alert work locally; detection data is not fabricated.

## API contract expected by `app.js`

Configure the base URL in **Settings → API connection**. Routes currently expected:

- `GET /api/health` — JSON such as `{ "person_model": "loaded", "ppe_model": "loaded", "device": "CPU" }`
- `GET /api/status` — status JSON, optionally with `ppe_observations`, `monitoring_status`, `device`, `unsafe`, and `active_alert`
- `GET /api/workers` — array of worker objects, or `{ "workers": [...] }`
- `GET /api/events` — array of event objects, or `{ "events": [...] }`
- `POST /api/video/analyze` — multipart form with `file`, `confidence`, `imgsz`; response may contain `summary`, `workers`, `events`, and `annotated_video_url`

Worker records can use fields such as:

```json
{
  "worker_id": 12,
  "helmet_status": "PPE detected",
  "safety_vest_status": "Missing-PPE prediction observed",
  "gloves_status": "Not observed / uncertain",
  "goggles_status": "PPE detected",
  "assessment": "UNSAFE",
  "reason": "no-safety vest prediction"
}
```

Event records can use `timestamp`, `worker_id`, `event`, `class`, `reason`, `source`, and `status`.

### Production notes

- Serve the frontend over HTTPS for camera access outside localhost.
- Configure CORS in the backend for the frontend's origin.
- Do not expose API secrets in a shared browser. The optional token is stored in local browser storage for prototype convenience.
- A local browser camera preview is not the same as server-side CCTV/RTSP inference.
- A red alert and browser tone are advisory. Model outputs and tracking IDs can be wrong; human verification is required.
- The currently identified PPE model does not have a positive mask class or a `no-safety shoes` class. The UI intentionally calls this limitation out.
