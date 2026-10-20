#!/usr/bin/env python3
"""Click a point in a local browser to correct one phase segment; no GPU needed."""
from __future__ import annotations

import argparse
import json
import logging
import math
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2

if __package__:
    from .annotate_sorting_points_correction import apply_correction, prepare_correction
else:
    from annotate_sorting_points_correction import apply_correction, prepare_correction

LOG = logging.getLogger(__name__)

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Manual point correction</title>
<style>
body {font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 16px;background:#171b22;color:#eee}
canvas {display:block;max-width:100%;height:auto;cursor:crosshair;outline:1px solid #888}
button {padding:10px 18px;margin:12px 10px 12px 0;font-size:16px}
#status {white-space:pre-wrap} code {overflow-wrap:anywhere}
</style></head><body>
<h2>Manual point correction</h2><p id="target"></p><p><code id="image-path"></code></p>
<p>Click the correct target. Click again to adjust it, then choose <b>Save point</b> (Enter).
Only this segment is updated. Cancel (Escape) leaves the files unchanged.</p>
<canvas id="canvas"></canvas>
<p id="coordinates">No point selected.</p>
<button id="save" disabled>Save point</button><button id="cancel">Cancel</button>
<p id="status" role="status"></p>
<script>
const config = __CONFIG__;
const canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d');
const saveButton = document.getElementById('save'), cancelButton = document.getElementById('cancel');
const status = document.getElementById('status');
let point = null, busy = false, finished = false;
document.getElementById('target').textContent = config.episode + ' / segment ' +
  String(config.segment).padStart(2, '0') + ': ' + config.prompt + ' (' + config.frames + ' frames)';
document.getElementById('image-path').textContent = config.image;
const image = new Image();
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(image, 0, 0);
  if (point) {
    const [x, y] = point; ctx.strokeStyle = '#00ff33'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(x - 12, y); ctx.lineTo(x + 12, y);
    ctx.moveTo(x, y - 12); ctx.lineTo(x, y + 12); ctx.stroke();
  }
}
image.onload = () => {canvas.width = image.naturalWidth; canvas.height = image.naturalHeight; draw();};
image.onerror = () => {status.textContent = 'Could not load image. Restart the script.';};
image.src = '/image.png';
canvas.addEventListener('click', event => {
  if (busy || finished || !image.complete || !image.naturalWidth) return;
  const r = canvas.getBoundingClientRect();
  point = [Math.max(0, Math.min(canvas.width - 1, Math.floor((event.clientX-r.left)*canvas.width/r.width))),
           Math.max(0, Math.min(canvas.height - 1, Math.floor((event.clientY-r.top)*canvas.height/r.height)))];
  document.getElementById('coordinates').textContent = 'Pixel xy: [' + point.join(', ') +
    '] — normalized xy: [' + (point[0]/canvas.width).toFixed(6) + ', ' + (point[1]/canvas.height).toFixed(6) + ']';
  saveButton.disabled = false; draw();
});
async function submit(action) {
  if (busy || finished || (action === 'save' && !point)) return;
  busy = true; saveButton.disabled = true; cancelButton.disabled = true;
  try {
    const response = await fetch('/' + action, {method:'POST',
      headers:{'Content-Type':'application/json', 'X-Annotation-Token':config.token},
      body:JSON.stringify({pixel_xy:point})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Correction failed');
    finished = true;
    status.textContent = action === 'save' ? 'Saved normalized point: ' + JSON.stringify(result.point_pixel) +
      '\\nUpdated ' + config.frames + ' frames. Backup: ' + result.backup + '\\nYou can close this tab.' :
      'Cancelled. Files were not changed. You can close this tab.';
  } catch(error) {status.textContent = error.message;}
  finally {busy = false; saveButton.disabled = finished || !point; cancelButton.disabled = finished;}
}
saveButton.onclick = () => submit('save'); cancelButton.onclick = () => submit('cancel');
document.addEventListener('keydown', event => {
  if (event.key === 'Enter') {event.preventDefault(); submit('save');}
  if (event.key === 'Escape') {event.preventDefault(); submit('cancel');}
});
</script></body></html>"""


def clicked_prediction(pixel_xy, width: int, height: int) -> dict:
    if not isinstance(pixel_xy, list) or len(pixel_xy) != 2:
        raise ValueError("Select one point on the image first")
    if width <= 0 or height <= 0 or any(type(v) not in (int, float) or not math.isfinite(v) for v in pixel_xy):
        raise ValueError("Invalid image dimensions or click coordinates")
    x, y = pixel_xy
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Clicked point is outside the displayed image")
    return {
        "pixel_xy": [x, y], "point_pixel": [x / width, y / height],
        "prompt_used": "human_click", "errors": [],
    }


def save_clicked_point(context: dict, pixel_xy) -> tuple[dict, Path]:
    height, width = context["rgb"].shape[:2]
    prediction = clicked_prediction(pixel_xy, width, height)
    settings = {
        "annotator": "human_click", "roi_xy_size": context["report"].get("roi_xy_size"),
        "point_coordinate_system": context["report"]["point_coordinate_system"],
        "image_width": width, "image_height": height,
    }
    backup = apply_correction(context, prediction, settings, method="human_click_correction", human_verified=True)
    return prediction, backup


def make_handler(context: dict, state: dict, token: str, page: bytes, image: bytes):
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
            if self.path == "/":
                self.reply(200, page, "text/html; charset=utf-8")
            elif self.path == "/image.png":
                self.reply(200, image, "image/png")
            else:
                self.reply(404, b'{"error":"Not found"}')

        def do_POST(self):
            supplied = self.headers.get("X-Annotation-Token", "")
            if not secrets.compare_digest(supplied, token):
                self.reply(403, b'{"error":"Invalid session token"}')
                return
            if state["done"]:
                self.reply(409, b'{"error":"This session already finished"}')
                return
            if self.path not in ("/save", "/cancel"):
                self.reply(404, b'{"error":"Not found"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request length")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Invalid request body")
                if self.path == "/cancel":
                    result = {"cancelled": True}
                else:
                    prediction, backup = save_clicked_point(context, body.get("pixel_xy"))
                    result = {"point_pixel": prediction["point_pixel"], "backup": str(backup)}
                    LOG.info("Saved pixel xy=%s; normalized xy=%s; backup=%s", prediction["pixel_xy"], prediction["point_pixel"], backup)
                state["result"] = result
                state["done"] = True
                self.reply(200, json.dumps(result).encode())
            except (ValueError, OSError, KeyError, RuntimeError) as exc:
                LOG.error("Correction failed: %s", exc)
                self.reply(400, json.dumps({"error": str(exc)}).encode())

        def log_message(self, format, *args):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Raw left JPEG or molmo_previews/segment_NN.jpg")
    parser.add_argument("--segment", type=int, choices=range(6), help="Optional segment check, 0–5")
    parser.add_argument("--port", type=int, default=0, help="Local browser port; default chooses an available port")
    parser.add_argument("--dry-run", action="store_true", help="Validate image/segment without starting UI or modifying files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not 0 <= args.port <= 65535:
        parser.error("port must be in [0, 65535]")
    try:
        context = prepare_correction(args.image, args.segment)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    start, end = context["segment"]["start_frame"], context["segment"]["end_frame"]
    LOG.info("Segment %02d: %s; %d frames [%d:%d]", context["segment_index"], context["segment"]["phase_prompt"], end-start, start, end)
    LOG.info("Original image: %s; saved ROI: %s", context["image"], context["report"].get("roi_xy_size"))
    if args.dry_run:
        return
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(context["rgb"], cv2.COLOR_RGB2BGR))
    if not ok:
        parser.error("Cannot encode image for browser")
    token = secrets.token_urlsafe(24)
    config = {
        "episode": context["episode"].name, "segment": context["segment_index"],
        "prompt": context["segment"]["phase_prompt"], "frames": end-start,
        "image": str(context["image"]), "token": token,
    }
    page = PAGE.replace("__CONFIG__", json.dumps(config).replace("<", "\\u003c")).encode()
    state = {"done": False, "result": None}
    handler = make_handler(context, state, token, page, encoded.tobytes())
    with HTTPServer(("127.0.0.1", args.port), handler) as server:
        server.timeout = 0.5
        LOG.info("Open http://127.0.0.1:%d/ in a browser on this computer", server.server_port)
        LOG.info("Click the target, then Save point (Enter). Cancel (Escape) or Ctrl+C leaves files unchanged before saving.")
        try:
            while not state["done"]:
                server.handle_request()
        except KeyboardInterrupt:
            LOG.info("Cancelled")
    if state["result"] and not state["result"].get("cancelled"):
        LOG.info("Updated %s and segment_%02d.jpg. The segment is marked human_verified=true.", context["annotated"], context["segment_index"])


if __name__ == "__main__":
    main()
