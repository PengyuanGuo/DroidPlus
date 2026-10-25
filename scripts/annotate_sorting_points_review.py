#!/usr/bin/env python3
"""Browser queue for clicking points on every flagged segment of a dataset; works over SSH.

Same per-segment write path as annotate_sorting_points_manual.py (backup, history,
human_verified=true), but collects all segments with needs_review=true across the
dataset and lets you step through them, choosing any frame inside each segment.
The server binds 127.0.0.1 only; reach it from your own machine with an SSH tunnel:

    ssh -L 8765:127.0.0.1:8765 vast-atomic     # then open http://localhost:8765/
"""
from __future__ import annotations

import argparse
import json
import logging
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

import cv2

if __package__:
    from .annotate_sorting_points import read_episode, read_rgb
    from .annotate_sorting_points_correction import prepare_correction
    from .annotate_sorting_points_manual import save_clicked_point
else:
    from annotate_sorting_points import read_episode, read_rgb
    from annotate_sorting_points_correction import prepare_correction
    from annotate_sorting_points_manual import save_clicked_point

LOG = logging.getLogger(__name__)

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Point review queue</title>
<style>
body {font:15px system-ui;margin:0;background:#171b22;color:#eee;display:flex;min-height:100vh}
#list {width:300px;flex:none;overflow-y:auto;max-height:100vh;border-right:1px solid #333}
#list div {padding:6px 10px;cursor:pointer;border-bottom:1px solid #222;font-size:13px}
#list div.active {background:#2b3a55} #list .saved {color:#6f6} #list .skipped {color:#999}
main {flex:1;padding:12px 20px;min-width:0}
canvas {display:block;max-width:100%;height:auto;cursor:crosshair;outline:1px solid #888}
button {padding:8px 16px;margin:8px 8px 8px 0;font-size:15px}
input[type=range] {width:100%} #status {white-space:pre-wrap;color:#fc6}
@media (max-width:800px) {body {flex-direction:column} #list {width:auto;max-height:30vh}}
</style></head><body>
<nav id="list"></nav>
<main>
<h3 id="target"></h3>
<p>Click the target (green). Red = current saved point, if any. <b>Enter</b> save &amp; next,
<b>S</b> skip, <b>&larr;/&rarr;</b> change frame, <b>N/P</b> next/previous segment.</p>
<canvas id="canvas"></canvas>
<p>Frame <span id="frame"></span> <input id="slider" type="range"></p>
<p id="coordinates">No point selected.</p>
<button id="save" disabled>Save point</button><button id="skip">Skip</button>
<button id="prev">Previous</button><button id="next">Next</button>
<p id="status" role="status"></p>
</main>
<script>
const config = __CONFIG__;
const items = config.items;
const canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d');
const slider = document.getElementById('slider'), status = document.getElementById('status');
const saveButton = document.getElementById('save');
let current = 0, frame = 0, point = null, busy = false;
const image = new Image();
function renderList() {
  const list = document.getElementById('list'); list.innerHTML = '';
  items.forEach((item, i) => {
    const div = document.createElement('div');
    div.textContent = (item.status === 'saved' ? '✓ ' : item.status === 'skipped' ? '– ' : '• ') +
      item.label + ' — ' + item.prompt;
    div.className = (i === current ? 'active ' : '') + item.status;
    div.onclick = () => select(i); list.appendChild(div);
  });
  const done = items.filter(i => i.status === 'saved').length;
  document.title = 'Point review ' + done + '/' + items.length;
}
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(image, 0, 0);
  const cross = (x, y, color) => {ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.beginPath();
    ctx.moveTo(x - 12, y); ctx.lineTo(x + 12, y); ctx.moveTo(x, y - 12); ctx.lineTo(x, y + 12); ctx.stroke();};
  const saved = items[current].point;
  if (saved && saved[0] >= 0) cross(saved[0] * canvas.width, saved[1] * canvas.height, '#ff3030');
  if (point) cross(point[0], point[1], '#00ff33');
}
function load() {
  const item = items[current];
  document.getElementById('frame').textContent = frame + ' (segment ' + item.start + '–' + (item.end - 1) + ')';
  image.src = '/image?item=' + current + '&frame=' + frame + '&t=' + config.token;
}
function select(i) {
  if (busy || i < 0 || i >= items.length) return;
  current = i; point = null; saveButton.disabled = true;
  const item = items[i];
  document.getElementById('target').textContent = (i + 1) + '/' + items.length + '  ' + item.label +
    '  segment ' + String(item.segment).padStart(2, '0') + ': ' + item.prompt + ' (' + (item.end - item.start) + ' frames)';
  slider.min = item.start; slider.max = item.end - 1; frame = item.frame; slider.value = frame;
  document.getElementById('coordinates').textContent = 'No point selected.';
  status.textContent = item.status === 'saved' ? 'Already saved in this session; saving again overwrites it.' : '';
  renderList(); load();
}
function nextPending(from) {
  for (let k = 1; k <= items.length; k++) {
    const j = (from + k) % items.length; if (items[j].status === 'pending') return j;
  }
  return -1;
}
image.onload = () => {canvas.width = image.naturalWidth; canvas.height = image.naturalHeight; draw();};
image.onerror = () => {status.textContent = 'Could not load image.';};
slider.oninput = () => {frame = Number(slider.value); load();};
canvas.addEventListener('click', event => {
  if (busy || !image.naturalWidth) return;
  const r = canvas.getBoundingClientRect();
  point = [Math.max(0, Math.min(canvas.width - 1, Math.floor((event.clientX-r.left)*canvas.width/r.width))),
           Math.max(0, Math.min(canvas.height - 1, Math.floor((event.clientY-r.top)*canvas.height/r.height)))];
  document.getElementById('coordinates').textContent = 'Pixel xy: [' + point.join(', ') +
    '] — normalized xy: [' + (point[0]/canvas.width).toFixed(6) + ', ' + (point[1]/canvas.height).toFixed(6) + ']';
  saveButton.disabled = false; draw();
});
async function save() {
  if (busy || !point) return;
  busy = true; saveButton.disabled = true;
  try {
    const response = await fetch('/save', {method:'POST',
      headers:{'Content-Type':'application/json', 'X-Annotation-Token':config.token},
      body:JSON.stringify({item:current, frame:frame, pixel_xy:point})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Save failed');
    items[current].status = 'saved'; items[current].point = result.point_pixel; items[current].frame = frame;
    busy = false;
    const n = nextPending(current);
    if (n < 0) {renderList(); point = null; draw(); status.textContent = 'All segments handled. You can stop the server (Ctrl+C).';}
    else {select(n); status.textContent = 'Saved previous segment.';}
  } catch(error) {status.textContent = error.message; saveButton.disabled = false;}
  finally {busy = false;}
}
function skip() {
  if (busy) return;
  if (items[current].status === 'pending') items[current].status = 'skipped';
  const n = nextPending(current); select(n < 0 ? Math.min(current + 1, items.length - 1) : n);
}
saveButton.onclick = save; document.getElementById('skip').onclick = skip;
document.getElementById('prev').onclick = () => select(current - 1);
document.getElementById('next').onclick = () => select(current + 1);
document.addEventListener('keydown', event => {
  const item = items[current];
  if (event.key === 'Enter') {event.preventDefault(); save();}
  else if (event.key === 's' || event.key === 'S') skip();
  else if (event.key === 'n' || event.key === 'N') select(current + 1);
  else if (event.key === 'p' || event.key === 'P') select(current - 1);
  else if (event.key === 'ArrowRight' && frame < item.end - 1) {frame++; slider.value = frame; load();}
  else if (event.key === 'ArrowLeft' && frame > item.start) {frame--; slider.value = frame; load();}
});
if (items.length) select(0); else status.textContent = 'No segments to review.';
</script></body></html>"""


def collect_items(dataset_root: Path, include_all: bool, methods: list[str] | None = None) -> list[dict]:
    items = []
    for report_path in sorted(dataset_root.rglob("molmo_point_annotations.json")):
        episode = report_path.parent
        report = json.loads(report_path.read_text())
        rows, _ = read_episode(episode / "steps.jsonl")
        for index, segment in enumerate(report["segments"]):
            if methods is not None:
                if segment.get("method") not in methods or segment.get("human_verified"):
                    continue
            elif not include_all and not segment.get("needs_review"):
                continue
            start, end = segment["start_frame"], segment["end_frame"]
            samples = [i for i in segment.get("sample_indices", []) if type(i) is int and start <= i < end]
            items.append({
                "label": str(episode.relative_to(dataset_root)), "segment": index,
                "prompt": segment["phase_prompt"], "start": start, "end": end,
                "frame": samples[0] if samples else start, "point": segment.get("point_pixel"),
                "status": "pending", "episode": episode,
                "images": [row["images"]["left"] for row in rows],
                "roi": report.get("roi_xy_size"),
            })
    return items


def frame_image(item: dict, frame: int) -> Path:
    if not item["start"] <= frame < item["end"]:
        raise ValueError("Frame is outside this segment")
    return item["episode"] / item["images"][frame]


def make_handler(items: list[dict], token: str, page: bytes):
    lock = Lock()  # one write at a time; prepare_correction re-reads files on every save

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code: int, content: bytes, content_type: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self.reply(200, page, "text/html; charset=utf-8")
                return
            if url.path != "/image":
                self.reply(404, b'{"error":"Not found"}')
                return
            query = parse_qs(url.query)
            try:
                if not secrets.compare_digest(query.get("t", [""])[0], token):
                    raise PermissionError
                item = items[int(query["item"][0])]
                rgb = read_rgb(frame_image(item, int(query["frame"][0])), item["roi"])
                ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise ValueError("Cannot encode image")
                self.reply(200, encoded.tobytes(), "image/jpeg")
            except PermissionError:
                self.reply(403, b'{"error":"Invalid session token"}')
            except (ValueError, KeyError, IndexError, OSError) as exc:
                self.reply(400, json.dumps({"error": str(exc)}).encode())

        def do_POST(self):
            if not secrets.compare_digest(self.headers.get("X-Annotation-Token", ""), token):
                self.reply(403, b'{"error":"Invalid session token"}')
                return
            if self.path != "/save":
                self.reply(404, b'{"error":"Not found"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request length")
                body = json.loads(self.rfile.read(length))
                item = items[int(body["item"])]
                image = frame_image(item, int(body["frame"]))
                with lock:
                    context = prepare_correction(image, item["segment"])
                    prediction, backup = save_clicked_point(context, body.get("pixel_xy"))
                item["status"], item["point"] = "saved", prediction["point_pixel"]
                LOG.info("%s segment %02d: saved %s from %s; backup=%s",
                         item["label"], item["segment"], prediction["point_pixel"], image.name, backup)
                self.reply(200, json.dumps({"point_pixel": prediction["point_pixel"], "backup": str(backup)}).encode())
            except (ValueError, OSError, KeyError, IndexError, TypeError, RuntimeError) as exc:
                LOG.error("Save failed: %s", exc)
                self.reply(400, json.dumps({"error": str(exc)}).encode())

        def log_message(self, format, *args):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="Queue every segment, not only needs_review=true")
    parser.add_argument("--method", action="append",
                        help="Queue segments with this report method instead of needs_review (repeatable), "
                             "e.g. auto_retry_molmo; already human-verified segments are skipped")
    parser.add_argument("--port", type=int, default=8765, help="Port on 127.0.0.1; forward it with ssh -L")
    parser.add_argument("--dry-run", action="store_true", help="List queued segments without starting the server")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not 0 <= args.port <= 65535:
        parser.error("port must be in [0, 65535]")
    try:
        items = collect_items(args.dataset_root.resolve(strict=True), args.all, args.method)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    for item in items:
        LOG.info("%s segment %02d: %s (%d frames)", item["label"], item["segment"], item["prompt"], item["end"] - item["start"])
    LOG.info("%d segments queued", len(items))
    if args.dry_run or not items:
        return
    token = secrets.token_urlsafe(24)
    public = [{k: v for k, v in item.items() if k not in ("episode", "images", "roi")} for item in items]
    config = {"items": public, "token": token}
    page = PAGE.replace("__CONFIG__", json.dumps(config).replace("<", "\\u003c")).encode()
    with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(items, token, page)) as server:
        port = server.server_port
        LOG.info("Serving on 127.0.0.1:%d. From your own computer run:", port)
        LOG.info("    ssh -L %d:127.0.0.1:%d vast-atomic", port, port)
        LOG.info("then open http://localhost:%d/  (Ctrl+C here to stop)", port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    saved = sum(item["status"] == "saved" for item in items)
    LOG.info("Saved %d of %d queued segments", saved, len(items))


if __name__ == "__main__":
    main()
