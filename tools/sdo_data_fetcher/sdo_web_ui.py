"""
Dependency-free local web UI for historical SDO data review.

Run with:
    python sdo_web_ui.py
"""

from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Dict
from urllib.parse import unquote, urlparse

from sdo_provider import SDOProviderClient, SDO_SOURCES, parse_target_datetime


HOST = "127.0.0.1"
PORT = 8765
OUTPUT_DIR = Path("sdo_data")
MAX_HOURS = 12
MAX_SAMPLES = 96
MAX_WIDTH = 2048

JOBS: Dict[str, Dict] = {}
JOBS_LOCK = threading.Lock()


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SDO Solar Moment Console</title>
  <style>
    :root {
      --intel-blue: #0068b5;
      --intel-light: #00c7fd;
      --intel-dark: #003c71;
      --ink: #07111f;
      --panel: #0c1d32;
      --panel-2: #102945;
      --white: #ffffff;
      --muted: #b8d9f1;
      --warning: #ffd166;
      --danger: #ff5f6d;
      --ok: #5cffc8;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--white);
      background:
        linear-gradient(90deg, rgba(0, 104, 181, 0.18) 1px, transparent 1px),
        linear-gradient(rgba(0, 199, 253, 0.12) 1px, transparent 1px),
        radial-gradient(circle at 20% 10%, rgba(0, 199, 253, 0.35), transparent 28rem),
        radial-gradient(circle at 80% 0%, rgba(0, 104, 181, 0.45), transparent 34rem),
        var(--ink);
      background-size: 28px 28px, 28px 28px, auto, auto, auto;
      font-family: "Courier New", Courier, monospace;
      min-height: 100vh;
    }

    header {
      border-bottom: 4px solid var(--intel-light);
      background: linear-gradient(135deg, var(--intel-blue), var(--intel-dark));
      padding: 22px clamp(18px, 4vw, 48px);
      box-shadow: 0 0 28px rgba(0, 199, 253, 0.36);
    }

    h1 {
      margin: 0;
      font-size: clamp(1.8rem, 4vw, 4rem);
      line-height: 0.95;
      letter-spacing: -0.08em;
      text-transform: uppercase;
    }

    .tagline {
      margin: 10px 0 0;
      color: var(--muted);
      max-width: 920px;
      font-size: 0.95rem;
    }

    main {
      display: grid;
      grid-template-columns: minmax(280px, 420px) 1fr;
      gap: 22px;
      padding: 24px clamp(14px, 3vw, 34px) 40px;
    }

    .panel {
      background: linear-gradient(180deg, rgba(16, 41, 69, 0.94), rgba(7, 17, 31, 0.96));
      border: 3px solid var(--intel-light);
      box-shadow: 8px 8px 0 var(--intel-blue), 0 0 22px rgba(0, 199, 253, 0.22);
      padding: 18px;
    }

    .panel h2 {
      margin: 0 0 16px;
      color: var(--intel-light);
      font-size: 1.05rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }

    label {
      display: block;
      color: var(--muted);
      font-size: 0.8rem;
      margin: 14px 0 7px;
      text-transform: uppercase;
    }

    input, select, button {
      width: 100%;
      border: 2px solid var(--intel-light);
      background: #050b13;
      color: var(--white);
      font: inherit;
      padding: 11px 12px;
      outline: none;
    }

    input:focus, select:focus {
      box-shadow: 0 0 0 3px rgba(0, 199, 253, 0.25);
    }

    button {
      margin-top: 18px;
      cursor: pointer;
      color: #00172a;
      background: linear-gradient(180deg, var(--intel-light), #74e5ff);
      border-color: var(--white);
      font-weight: 900;
      text-transform: uppercase;
      box-shadow: 4px 4px 0 var(--intel-blue);
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.65;
    }

    .split {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .toggle-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 0.85rem;
      text-transform: uppercase;
    }

    .toggle-row input {
      width: auto;
      accent-color: var(--intel-light);
    }

    .sources {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 10px;
      max-height: 210px;
      overflow: auto;
      padding-right: 6px;
    }

    .source-chip {
      display: flex;
      gap: 7px;
      align-items: center;
      border: 1px solid rgba(0, 199, 253, 0.6);
      padding: 7px;
      color: var(--white);
      background: rgba(0, 104, 181, 0.16);
      font-size: 0.78rem;
    }

    .source-chip input { width: auto; }

    .status-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }

    .meter {
      border: 2px solid rgba(0, 199, 253, 0.7);
      background: rgba(0, 60, 113, 0.42);
      padding: 12px;
    }

    .meter span {
      display: block;
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
    }

    .meter strong {
      display: block;
      margin-top: 5px;
      color: var(--white);
      font-size: 1.1rem;
      word-break: break-word;
    }

    .progress-shell {
      height: 22px;
      border: 2px solid var(--intel-light);
      background: #050b13;
      margin-bottom: 18px;
    }

    .progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--intel-blue), var(--intel-light));
      transition: width 0.25s ease;
    }

    .log {
      min-height: 72px;
      max-height: 170px;
      overflow: auto;
      border: 2px solid rgba(0, 199, 253, 0.5);
      background: #03070d;
      padding: 10px;
      color: var(--ok);
      font-size: 0.82rem;
      margin-bottom: 18px;
      white-space: pre-wrap;
    }

    .results {
      display: grid;
      gap: 18px;
    }

    .time-group {
      border: 2px solid rgba(255, 255, 255, 0.2);
      background: rgba(255, 255, 255, 0.04);
      padding: 12px;
    }

    .time-group h3 {
      margin: 0 0 12px;
      color: var(--warning);
      font-size: 0.95rem;
    }

    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
      gap: 12px;
    }

    .card {
      border: 2px solid var(--intel-blue);
      background: #06111e;
      padding: 10px;
    }

    .card img {
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
      background: #000;
      border: 1px solid rgba(0, 199, 253, 0.35);
    }

    .card h4 {
      margin: 8px 0 4px;
      color: var(--intel-light);
      font-size: 0.95rem;
    }

    .card p {
      margin: 4px 0;
      color: var(--muted);
      font-size: 0.72rem;
      line-height: 1.35;
    }

    .card a {
      color: var(--white);
      font-size: 0.75rem;
      margin-right: 10px;
    }

    .error { color: var(--danger); }

    @media (max-width: 920px) {
      main { grid-template-columns: 1fr; }
      .status-grid { grid-template-columns: 1fr 1fr; }
    }

    @media (max-width: 560px) {
      .split, .sources, .status-grid { grid-template-columns: 1fr; }
      .panel { box-shadow: 5px 5px 0 var(--intel-blue); }
    }
  </style>
</head>
<body>
  <header>
    <h1>SDO Solar Moment Console</h1>
    <p class="tagline">Intel-blue retro mission control for replaying SDO flare windows. Pick a start time, fetch forward samples, and review every available wavelength.</p>
  </header>

  <main>
    <section class="panel">
      <h2>Acquisition Controls</h2>
      <form id="fetch-form">
        <div class="split">
          <div>
            <label for="date">Target Date</label>
            <input id="date" name="date" type="date" required>
          </div>
          <div>
            <label for="time">Target Time</label>
            <input id="time" name="time" type="time" step="1" required>
          </div>
        </div>

        <label for="timezone-mode">Timezone Mode</label>
        <select id="timezone-mode" name="timezone_mode">
          <option value="utc">UTC</option>
          <option value="local">Local timezone</option>
        </select>

        <div class="split">
          <div>
            <label for="hours">Forward Hours</label>
            <input id="hours" name="hours" type="number" min="0.1" max="12" step="0.5" value="2" required>
          </div>
          <div>
            <label for="cadence">Cadence Minutes</label>
            <input id="cadence" name="cadence" type="number" min="1" max="180" step="1" value="30" required>
          </div>
        </div>

        <div class="split">
          <div>
            <label for="width">Image Width</label>
            <input id="width" name="width" type="number" min="128" max="2048" step="128" value="1024" required>
          </div>
          <div>
            <label for="image-type">Image Type</label>
            <select id="image-type" name="image_type">
              <option value="png">PNG</option>
              <option value="jpg">JPG</option>
              <option value="webp">WEBP</option>
            </select>
          </div>
        </div>

        <label class="toggle-row"><input id="all-sources" type="checkbox" checked> Fetch all wavelengths</label>
        <div id="source-list" class="sources"></div>

        <button id="submit" type="submit">Fetch Solar Moment</button>
      </form>
    </section>

    <section class="panel">
      <h2>Review Deck</h2>
      <div class="status-grid">
        <div class="meter"><span>Status</span><strong id="status">Idle</strong></div>
        <div class="meter"><span>Progress</span><strong id="progress-text">0 / 0</strong></div>
        <div class="meter"><span>Images</span><strong id="success-text">0 OK</strong></div>
        <div class="meter"><span>Failures</span><strong id="failure-text">0 ERR</strong></div>
      </div>
      <div class="progress-shell"><div id="progress-bar" class="progress-bar"></div></div>
      <div id="log" class="log">READY. Awaiting target time.</div>
      <div id="results" class="results"></div>
    </section>
  </main>

  <script>
    const sources = __SOURCES__;
    const form = document.getElementById('fetch-form');
    const sourceList = document.getElementById('source-list');
    const allSources = document.getElementById('all-sources');
    const submit = document.getElementById('submit');
    const statusEl = document.getElementById('status');
    const progressText = document.getElementById('progress-text');
    const successText = document.getElementById('success-text');
    const failureText = document.getElementById('failure-text');
    const progressBar = document.getElementById('progress-bar');
    const log = document.getElementById('log');
    const results = document.getElementById('results');

    function pad(value) { return String(value).padStart(2, '0'); }

    function setDefaultTime() {
      const now = new Date(Date.now() - 60 * 60 * 1000);
      document.getElementById('date').value = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
      document.getElementById('time').value = `${pad(now.getHours())}:${pad(now.getMinutes())}:00`;
    }

    function renderSources() {
      sourceList.innerHTML = sources.map((source) => `
        <label class="source-chip">
          <input type="checkbox" name="source" value="${source.key}" checked>
          <span>${source.key}<br>${source.wavelength}</span>
        </label>
      `).join('');
    }

    function appendLog(message, isError = false) {
      const prefix = new Date().toLocaleTimeString();
      log.textContent += `\n[${prefix}] ${message}`;
      if (isError) log.classList.add('error');
      log.scrollTop = log.scrollHeight;
    }

    function fileUrl(path) {
      return `/files/${encodeURIComponent(path)}`;
    }

    function metadataUrl(path) {
      if (!path) return '#';
      return `/files/${encodeURIComponent(path)}`;
    }

    function groupByRequested(items) {
      return items.reduce((groups, item) => {
        const key = item.requested_time || 'unknown';
        groups[key] = groups[key] || [];
        groups[key].push(item);
        return groups;
      }, {});
    }

    function renderResults(items) {
      const groups = groupByRequested(items);
      results.innerHTML = Object.keys(groups).sort().map((time) => `
        <section class="time-group">
          <h3>${time}</h3>
          <div class="cards">
            ${groups[time].map((item) => {
              const delta = item.delta_seconds === null || item.delta_seconds === undefined ? 'n/a' : `${Math.round(item.delta_seconds)}s`;
              return `
                <article class="card">
                  <a href="${fileUrl(item.filepath)}" target="_blank" rel="noreferrer"><img src="${fileUrl(item.filepath)}" alt="${item.source}"></a>
                  <h4>${item.source}</h4>
                  <p>${item.name} / ${item.wavelength}</p>
                  <p>OBS ${item.actual_observation_time || item.observation_time || 'unknown'}</p>
                  <p>DELTA ${delta}</p>
                  <p><a href="${fileUrl(item.filepath)}" target="_blank" rel="noreferrer">image</a><a href="${metadataUrl(item.metadata_filepath)}" target="_blank" rel="noreferrer">metadata</a></p>
                </article>
              `;
            }).join('')}
          </div>
        </section>
      `).join('');
    }

    function updateJob(job) {
      statusEl.textContent = job.status.toUpperCase();
      progressText.textContent = `${job.completed} / ${job.total}`;
      successText.textContent = `${job.results.length} OK`;
      failureText.textContent = `${job.errors.length} ERR`;
      const percent = job.total ? Math.round((job.completed / job.total) * 100) : 0;
      progressBar.style.width = `${percent}%`;
      renderResults(job.results);
    }

    async function pollJob(id) {
      const response = await fetch(`/api/job/${id}`);
      const job = await response.json();
      updateJob(job);
      if (job.status === 'running' || job.status === 'queued') {
        setTimeout(() => pollJob(id), 1200);
      } else {
        submit.disabled = false;
        appendLog(job.status === 'completed' ? 'Fetch complete.' : `Fetch ended: ${job.status}`, job.status === 'failed');
      }
    }

    allSources.addEventListener('change', () => {
      document.querySelectorAll('input[name="source"]').forEach((box) => {
        box.checked = allSources.checked;
      });
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      submit.disabled = true;
      log.classList.remove('error');
      log.textContent = 'TRANSMITTING FETCH REQUEST...';
      results.innerHTML = '';

      const selectedSources = Array.from(document.querySelectorAll('input[name="source"]:checked')).map((box) => box.value);
      const payload = {
        date: document.getElementById('date').value,
        time: document.getElementById('time').value,
        timezone_mode: document.getElementById('timezone-mode').value,
        hours: Number(document.getElementById('hours').value),
        cadence_minutes: Number(document.getElementById('cadence').value),
        width: Number(document.getElementById('width').value),
        image_type: document.getElementById('image-type').value,
        all_sources: allSources.checked,
        sources: selectedSources
      };

      try {
        const response = await fetch('/api/fetch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Request failed');
        appendLog(`Job ${data.job_id} accepted. Total downloads: ${data.total}`);
        pollJob(data.job_id);
      } catch (error) {
        submit.disabled = false;
        appendLog(error.message, true);
        statusEl.textContent = 'ERROR';
      }
    });

    setDefaultTime();
    renderSources();
  </script>
</body>
</html>
"""


def build_index() -> bytes:
    sources = [
        {
            "key": key,
            "name": value["name"],
            "wavelength": value["wavelength"],
            "description": value["description"],
        }
        for key, value in SDO_SOURCES.items()
    ]
    return INDEX_HTML.replace("__SOURCES__", json.dumps(sources)).encode("utf-8")


def json_response(handler: BaseHTTPRequestHandler, payload: Dict, status: int = HTTPStatus.OK):
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, message: str, status: int = HTTPStatus.BAD_REQUEST):
    body = message.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def calculate_total(hours: float, cadence_minutes: int, source_count: int) -> int:
    sample_count = 1
    elapsed = 0
    window_minutes = hours * 60
    while elapsed + cadence_minutes <= window_minutes:
        sample_count += 1
        elapsed += cadence_minutes
    return sample_count * source_count


def validate_fetch_payload(payload: Dict) -> Dict:
    date_value = str(payload.get("date", "")).strip()
    time_value = str(payload.get("time", "")).strip()
    timezone_mode = str(payload.get("timezone_mode", "utc")).strip().lower()
    image_type = str(payload.get("image_type", "png")).strip().lower()

    if not date_value or not time_value:
        raise ValueError("Date and time are required")
    if timezone_mode not in {"utc", "local"}:
        raise ValueError("timezone_mode must be utc or local")
    if image_type not in {"png", "jpg", "webp"}:
        raise ValueError("image_type must be png, jpg, or webp")

    hours = float(payload.get("hours", 1))
    cadence_minutes = int(payload.get("cadence_minutes", 15))
    width = int(payload.get("width", 1024))

    if hours <= 0 or hours > MAX_HOURS:
        raise ValueError(f"hours must be between 0 and {MAX_HOURS}")
    if cadence_minutes <= 0:
        raise ValueError("cadence_minutes must be greater than zero")
    if width <= 0 or width > MAX_WIDTH:
        raise ValueError(f"width must be between 1 and {MAX_WIDTH}")

    if payload.get("all_sources", True):
        sources = list(SDO_SOURCES.keys())
    else:
        sources = [str(source) for source in payload.get("sources", [])]
    if not sources:
        raise ValueError("Select at least one source")
    invalid = [source for source in sources if source not in SDO_SOURCES]
    if invalid:
        raise ValueError(f"Invalid sources: {invalid}")

    total = calculate_total(hours, cadence_minutes, len(sources))
    if total > MAX_SAMPLES * len(SDO_SOURCES):
        raise ValueError("Request is too large; reduce duration, cadence, or source count")

    target_time = f"{date_value}T{time_value}"
    start_dt = parse_target_datetime(target_time, timezone_mode=timezone_mode)

    return {
        "target_time": target_time,
        "start_time_utc": start_dt.isoformat(),
        "timezone_mode": timezone_mode,
        "hours": hours,
        "cadence_minutes": cadence_minutes,
        "width": width,
        "image_type": image_type,
        "sources": sources,
        "total": total,
    }


def run_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["status"] = "running"
        job["started_at"] = datetime.now(timezone.utc).isoformat()

    def progress(event: Dict):
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["completed"] = event.get("completed", job["completed"])
            if event["type"] == "result":
                job["results"].append(event["result"])
            elif event["type"] == "error":
                job["errors"].append(event["error"])

    try:
        with JOBS_LOCK:
            params = dict(JOBS[job_id]["params"])

        client = SDOProviderClient(output_dir=str(OUTPUT_DIR))
        manifest = client.download_samples(
            sources=params["sources"],
            start_time=params["target_time"],
            timezone_mode=params["timezone_mode"],
            hours=params["hours"],
            cadence_minutes=params["cadence_minutes"],
            width=params["width"],
            image_type=params["image_type"],
            output_subdir=f"web_{job_id}",
            progress_callback=progress,
        )

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["manifest"] = manifest
            JOBS[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["errors"].append({"error": str(exc)})
            JOBS[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()


class SDORequestHandler(BaseHTTPRequestHandler):
    server_version = "SDOWebUI/1.0"

    def log_message(self, format, *args):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {self.address_string()} {format % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = build_index()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/sources":
            json_response(self, {"sources": list(SDO_SOURCES.keys())})
            return

        if parsed.path.startswith("/api/job/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job:
                    payload = json.loads(json.dumps(job))
                else:
                    payload = None
            if not payload:
                json_response(self, {"error": "Job not found"}, HTTPStatus.NOT_FOUND)
                return
            json_response(self, payload)
            return

        if parsed.path.startswith("/files/"):
            self.serve_file(parsed.path[len("/files/"):])
            return

        text_response(self, "Not found", HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/fetch":
            text_response(self, "Not found", HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            params = validate_fetch_payload(payload)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        job_id = f"{int(time.time())}_{len(JOBS) + 1}"
        job = {
            "id": job_id,
            "status": "queued",
            "params": params,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "completed_at": None,
            "completed": 0,
            "total": params["total"],
            "results": [],
            "errors": [],
            "manifest": None,
        }

        with JOBS_LOCK:
            JOBS[job_id] = job

        thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
        thread.start()
        json_response(self, {"job_id": job_id, "total": params["total"]}, HTTPStatus.ACCEPTED)

    def serve_file(self, encoded_path: str):
        requested = Path(unquote(encoded_path))
        try:
            resolved = requested.resolve()
            output_root = OUTPUT_DIR.resolve()
            if output_root not in resolved.parents and resolved != output_root:
                raise ValueError("Path outside output directory")
            if not resolved.is_file():
                raise FileNotFoundError(str(requested))
        except Exception:
            text_response(self, "File not found", HTTPStatus.NOT_FOUND)
            return

        suffix = resolved.suffix.lower()
        content_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".json": "application/json; charset=utf-8",
        }.get(suffix, "application/octet-stream")

        data = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), SDORequestHandler)
    print(f"SDO Solar Moment Console running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SDO Solar Moment Console.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
