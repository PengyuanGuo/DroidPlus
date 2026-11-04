#!/usr/bin/env python3
"""Hand-label the crossword grid slots once per session and apply them to every episode of that session.

The board does not move within a session (left-frame shifts <= ~1 px), but it can between sessions.

    click: serve a page with reference left frames per session; click slot 1..N in order for each session;
           saves <dataset>/static_slot_points.json  {"sessions": {<session dir>: {"1": [x, y], ...}}}
    apply: write those points into every pre-place / place segment (grid_cell k -> slot k) through
           annotate_sorting_points_correction.apply_correction (backup + history kept, human_verified)

Usage (DroidPlus/.venv-molmo, no GPU):
    python scripts/annotate_static_slots.py click --dataset-root DATASET --slots 5 [--port 8765]
        open http://localhost:<port>/ through an ssh -L tunnel (or the editor's port forwarding)
    python scripts/annotate_static_slots.py apply --dataset-root DATASET [--dry-run]

Letter cubes: review all episodes (letters on the first frame, failed ones marked ?), drag / click to fix:
    python scripts/annotate_static_slots.py letters --dataset-root DATASET --port 8766   -> letter_edits.json
    python scripts/annotate_static_slots.py apply-letters --dataset-root DATASET [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Static slot points</title>
<style>body{font:15px system-ui;background:#171b22;color:#eee;margin:16px}
canvas{max-width:100%;cursor:crosshair;outline:1px solid #888}
button{padding:7px 12px;margin:4px 6px 4px 0;font-size:14px}.on{background:#4a7;color:#fff}.done{border:2px solid #4a7}
.row{margin:6px 0}</style></head><body>
<div class="row"><b>Session:</b> <span id="sess"></span></div>
<div class="row"><b>Frame:</b> <span id="frames"></span></div>
<h3 id="h"></h3>
<p>Click the cell centre where the placed cube sits, slots 1..__N__ in order. Z / Backspace = undo.
<button id="copy">Copy points from previous session</button> (then undo / re-click any that are off)</p>
<canvas id="c"></canvas>
<p id="pts"></p><button id="save">Save all sessions</button><span id="st"></span>
<script>
const N=__N__, S=__SESSIONS__; let pts=__PTS__; let si=0, fi=0;
S.forEach(s=>{if(!pts[s.name])pts[s.name]=[]});
const c=document.getElementById('c'), x=c.getContext('2d'), img=new Image();
const sd=document.getElementById('sess'), fd=document.getElementById('frames');
function buttons(){sd.innerHTML='';S.forEach((s,i)=>{const b=document.createElement('button');b.textContent=s.name.slice(-6)+` (${s.n} ep)`;
  b.className=(i==si?'on ':'')+(pts[s.name].length==N?'done':'');b.onclick=()=>{si=i;fi=0;load()};sd.appendChild(b)});
 fd.innerHTML='';S[si].frames.forEach((f,i)=>{const b=document.createElement('button');b.textContent=f.label;b.className=i==fi?'on':'';
  b.onclick=()=>{fi=i;load()};fd.appendChild(b)})}
function load(){buttons();img.src=S[si].frames[fi].src}
img.onload=()=>{c.width=img.width*2;c.height=img.height*2;draw()};
function draw(){const p=pts[S[si].name];x.drawImage(img,0,0,c.width,c.height);p.forEach((q,i)=>{const u=q[0]*c.width,v=q[1]*c.height;
 x.strokeStyle='#ff0';x.lineWidth=2;x.beginPath();x.moveTo(u-8,v);x.lineTo(u+8,v);x.moveTo(u,v-8);x.lineTo(u,v+8);x.stroke();
 x.fillStyle='#ff0';x.font='bold 18px sans-serif';x.fillText(i+1,u+6,v-6)});
 document.getElementById('h').textContent=`Session ${S[si].name}: `+(p.length<N?`click slot ${p.length+1}`:'all slots set');
 document.getElementById('pts').textContent=p.map((q,i)=>`slot ${i+1}: (${q[0].toFixed(4)}, ${q[1].toFixed(4)})`).join('   ');buttons()}
c.onclick=e=>{const p=pts[S[si].name];if(p.length>=N)return;const r=c.getBoundingClientRect();p.push([(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]);draw()};
function undo(){pts[S[si].name].pop();draw()}
document.onkeydown=e=>{if(e.key=='z'||e.key=='Backspace')undo()};
document.getElementById('copy').onclick=()=>{if(si>0){pts[S[si].name]=pts[S[si-1].name].map(q=>[...q]);draw()}};
document.getElementById('save').onclick=async()=>{const r=await fetch('/save',{method:'POST',body:JSON.stringify(pts)});
 document.getElementById('st').textContent=' '+(r.ok?'saved: ':'error: ')+(await r.text())};
load();
</script></body></html>"""


def jpeg(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def sessions(root: Path, groups: dict) -> list[dict]:
    """Per session: first / last left frames of its first and last episode. Per override group (episodes whose board
    moved): first / last frames of every episode in it."""
    out = []
    for session in sorted(p for p in root.iterdir() if p.is_dir() and any(p.glob("episode_*/steps.jsonl"))):
        eps = sorted(p.parent for p in session.glob("episode_*/steps.jsonl"))
        out.append({"name": session.name, "n": len(eps), "frames": episode_frames([eps[0], eps[-1]])})
    for name, eps in groups.items():
        out.append({"name": name, "n": len(eps), "frames": episode_frames([root / e for e in eps])})
    return out


def episode_frames(eps) -> list[dict]:
    frames = []
    for ep in dict.fromkeys(eps):
        rows = [json.loads(l) for l in (ep / "steps.jsonl").read_text().splitlines() if l.strip()]
        for tag, row in (("first", rows[0]), ("last", rows[-1])):
            frames.append({"label": f"{ep.parent.name[-6:]}/{ep.name[-3:]} {tag}", "src": jpeg(ep / row["images"]["left"])})
    return frames


def click(args) -> None:
    out = args.dataset_root / "static_slot_points.json"
    saved = json.loads(out.read_text()) if out.exists() else {}
    old, groups = saved.get("sessions", {}), saved.get("groups", {})
    pts = {s: [v[str(k)] for k in range(1, args.slots + 1)] for s, v in old.items()}
    page = (PAGE.replace("__N__", str(args.slots)).replace("__SESSIONS__", json.dumps(sessions(args.dataset_root, groups)))
            .replace("__PTS__", json.dumps(pts))).encode()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(page)

        def do_POST(self):
            pts = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            done = {s: p for s, p in pts.items() if len(p) == args.slots and all(0 <= v < 1 for q in p for v in q)}
            out.write_text(json.dumps({"coordinate_system": "normalized xy on full left RGB image", "annotator": "human_click",
                                       "groups": groups, "sessions": {s: {str(i + 1): q for i, q in enumerate(p)} for s, p in sorted(done.items())}},
                                      indent=1) + "\n")
            msg = f"{len(done)}/{len(pts)} sessions complete -> {out}"
            print(msg, flush=True)
            self.send_response(200); self.end_headers(); self.wfile.write(msg.encode())

        def log_message(self, *a):
            pass

    print(f"serving on http://127.0.0.1:{args.port}/ (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


def apply(args) -> None:
    from annotate_sorting_points_correction import apply_correction, prepare_correction
    from annotate_sorting_points_manual import clicked_prediction

    saved = json.loads((args.dataset_root / "static_slot_points.json").read_text())
    by_session = saved["sessions"]
    # override groups (board moved): episode -> that group's slots, if the group is labelled
    by_episode = {e: by_session[g] for g, eps in saved.get("groups", {}).items() if g in by_session for e in eps}
    reports = sorted(args.dataset_root.glob("*/episode_*/molmo_point_annotations.json"))
    missing = sorted({r.parent.parent.name for r in reports} - set(by_session))
    unlabelled = sorted(set(saved.get("groups", {})) - set(by_session))
    if unlabelled:
        raise SystemExit(f"override groups without slot points: {unlabelled}")
    if missing:
        raise SystemExit(f"no slot points for sessions {missing}")
    n = 0
    for report_path in reports:
        ep = report_path.parent
        slots = by_episode.get(str(ep.relative_to(args.dataset_root)), by_session[ep.parent.name])
        report = json.loads(report_path.read_text())
        rows = [json.loads(l) for l in (ep / "steps.jsonl").read_text().splitlines() if l.strip()]
        for index, seg in enumerate(report["segments"]):
            if seg["phase_index"] not in (2, 3):
                continue
            n += 1
            if args.dry_run:
                continue
            x, y = slots[str(int(seg["grid_cell"]))]
            frame = seg["start_frame"]  # preview frames of slot segments can lie in the next segment
            context = prepare_correction(ep / rows[frame]["images"]["left"], index)
            height, width = context["rgb"].shape[:2]
            prediction = clicked_prediction([float(x * width), float(y * height)], width, height)
            settings = {"annotator": "human_click_static_slot", "roi_xy_size": context["report"].get("roi_xy_size"),
                        "point_coordinate_system": context["report"]["point_coordinate_system"],
                        "image_width": width, "image_height": height}
            apply_correction(context, prediction, settings, method="static_slot_click", human_verified=True,
                             segment_fields={"target_method": "hand-labelled static slot (per session)"})
    print(f"{'would set' if args.dry_run else 'set'} {n} pre-place/place segments in {len(reports)} episodes")


LETTER_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Letter cube review</title>
<style>body{font:14px system-ui;background:#171b22;color:#eee;margin:0;display:flex;height:100vh}
#side{width:230px;overflow-y:auto;border-right:1px solid #444;padding:8px;flex:none}
#side div{padding:3px 6px;cursor:pointer;border-radius:4px;font-family:monospace;font-size:12px}
#side div.cur{background:#345}#main{flex:1;overflow:auto;padding:10px 16px}
canvas{max-width:100%;cursor:crosshair;outline:1px solid #888}
button{padding:6px 11px;margin:3px 5px 3px 0;font-size:14px}.on{outline:3px solid #ff0}
.miss{color:#f66}.edit{color:#6cf}.ok{color:#8c8}.seen{opacity:.55}</style></head><body>
<div id="side"></div><div id="main">
<h3 id="h"></h3>
<div><b>Letter</b> (keys 1-5): <span id="lb"></span></div>
<div><b>Frame</b>: <span id="fb"></span></div>
<p>Select a letter, then click its cube (on the table, where it is picked up). Drag a marker to move it.
N / P = next / previous episode (marks this one reviewed). R = reset this episode to MolmoPoint. <b>Save</b> often.</p>
<canvas id="c"></canvas>
<p><button id="prev">Prev (P)</button><button id="next">Next (N)</button><button id="reset">Reset episode (R)</button>
<button id="save">Save</button><span id="st"></span></p></div>
<script>
const E=__EPISODES__; let edits=__EDITS__, seen=new Set(__SEEN__);
const COL=['#ff4040','#ffa500','#40ff40','#40c0ff','#ff40ff'];
let ei=0, li=0, fi=0, drag=-1;
const c=document.getElementById('c'),x=c.getContext('2d'),img=new Image();
function pt(e,k){const d=edits[e.name]; return d&&d[k]?d[k]:(e.points[k][0]>=0?e.points[k]:null)}
function status(e){if(edits[e.name]&&Object.keys(edits[e.name]).length)return'edit';return e.points.some(p=>p[0]<0)?'miss':'ok'}
function side(){const s=document.getElementById('side');s.innerHTML='';E.forEach((e,i)=>{const d=document.createElement('div');
 const st=status(e);d.className=(i==ei?'cur ':'')+st+(seen.has(e.name)?' seen':'');
 d.textContent=(seen.has(e.name)?'\u2713 ':'  ')+e.name.slice(-18)+(st=='miss'?' ?':st=='edit'?' *':'');
 d.onclick=()=>{ei=i;fi=0;load()};s.appendChild(d)})}
function load(){const e=E[ei];document.getElementById('h').textContent=`${ei+1}/${E.length}  ${e.name}  (${seen.size} reviewed)`;
 const lb=document.getElementById('lb');lb.innerHTML='';e.letters.forEach((L,k)=>{const b=document.createElement('button');
  b.textContent=`${k+1} ${L}`+(pt(e,k)?'':' ?');b.style.color=COL[k];b.className=k==li?'on':'';b.onclick=()=>{li=k;load()};lb.appendChild(b)});
 const fb=document.getElementById('fb');fb.innerHTML='';e.frames.forEach((f,i)=>{const b=document.createElement('button');
  b.textContent=f.label;b.className=i==fi?'on':'';b.onclick=()=>{fi=i;load()};fb.appendChild(b)});
 img.src='/img?p='+encodeURIComponent(e.frames[fi].path);side()}
img.onload=()=>{c.width=img.width*2;c.height=img.height*2;draw()};
function draw(){const e=E[ei];x.drawImage(img,0,0,c.width,c.height);e.letters.forEach((L,k)=>{const p=pt(e,k);if(!p)return;
 const u=p[0]*c.width,v=p[1]*c.height;x.strokeStyle=COL[k];x.lineWidth=k==li?4:2;x.beginPath();x.arc(u,v,13,0,7);x.stroke();
 x.beginPath();x.moveTo(u-5,v);x.lineTo(u+5,v);x.moveTo(u,v-5);x.lineTo(u,v+5);x.stroke();
 x.font='bold 17px sans-serif';x.lineWidth=3;x.strokeStyle='#000';x.strokeText(`${k+1}${L}`,u+14,v-10);x.fillStyle=COL[k];x.fillText(`${k+1}${L}`,u+14,v-10)})}
function pos(ev){const r=c.getBoundingClientRect();return[(ev.clientX-r.left)/r.width,(ev.clientY-r.top)/r.height]}
function setp(k,p){const e=E[ei];(edits[e.name]=edits[e.name]||{})[k]=p}
c.onmousedown=ev=>{const e=E[ei],q=pos(ev);drag=-1;e.letters.forEach((L,k)=>{const p=pt(e,k);
 if(p&&Math.hypot((p[0]-q[0])*c.width,(p[1]-q[1])*c.height)<16)drag=k});
 if(drag<0){setp(li,q);const nxt=e.letters.findIndex((L,k)=>!pt(e,k));if(nxt>=0)li=nxt;load()}else{li=drag}};
c.onmousemove=ev=>{if(drag>=0){setp(drag,pos(ev));draw()}};
c.onmouseup=()=>{if(drag>=0){drag=-1;load()}};
function go(d){seen.add(E[ei].name);ei=Math.max(0,Math.min(E.length-1,ei+d));fi=0;li=0;load()}
document.getElementById('prev').onclick=()=>go(-1);document.getElementById('next').onclick=()=>go(1);
document.getElementById('reset').onclick=()=>{delete edits[E[ei].name];load()};
document.onkeydown=ev=>{if(ev.key>='1'&&ev.key<='5'){li=+ev.key-1;load()}else if(ev.key=='n')go(1);else if(ev.key=='p')go(-1);
 else if(ev.key=='r'){delete edits[E[ei].name];load()}};
document.getElementById('save').onclick=async()=>{const r=await fetch('/save',{method:'POST',body:JSON.stringify({edits,seen:[...seen]})});
 document.getElementById('st').textContent=' '+(await r.text())};
load();
</script></body></html>"""


def letter_episodes(root: Path) -> list[dict]:
    """Every episode: its pre-grasp letter points, plus the first frame and each cycle's pre-grasp start frame."""
    out = []
    for report_path in sorted(root.glob("*/episode_*/molmo_point_annotations.json")):
        ep = report_path.parent
        segs = json.loads(report_path.read_text())["segments"]
        rows = [json.loads(l) for l in (ep / "steps.jsonl").read_text().splitlines() if l.strip()]
        pre = [s for s in segs if s["phase_index"] == 0]
        frames = [{"label": "first frame", "path": str((ep / rows[0]["images"]["left"]).relative_to(root))}]
        frames += [{"label": f"{k + 1} {s.get('letter')} start", "path": str((ep / rows[min(s['start_frame'] + 2, s['end_frame'] - 1)]["images"]["left"]).relative_to(root))}
                   for k, s in enumerate(pre)]
        out.append({"name": str(ep.relative_to(root)), "letters": [s.get("letter") for s in pre],
                    "points": [s["point_pixel"] if not s["needs_review"] else [-1, -1] for s in pre], "frames": frames})
    return out


def letters(args) -> None:
    """Review page: all letter points of an episode on its first frame; edits saved to letter_edits.json
    as {episode: {cycle: [x, y]}} (cycle 0-based). apply-letters writes them to pre-grasp + grasp."""
    out = args.dataset_root / "letter_edits.json"
    saved = json.loads(out.read_text()) if out.exists() else {"edits": {}, "seen": []}
    page = (LETTER_PAGE.replace("__EPISODES__", json.dumps(letter_episodes(args.dataset_root)))
            .replace("__EDITS__", json.dumps(saved["edits"])).replace("__SEEN__", json.dumps(saved["seen"]))).encode()
    root = args.dataset_root

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            from urllib.parse import parse_qs, urlparse
            u = urlparse(self.path)
            if u.path == "/img":
                f = (root / parse_qs(u.query)["p"][0]).resolve()
                if root not in f.parents or f.suffix != ".jpg" or not f.is_file():
                    self.send_response(404); self.end_headers(); return
                self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.end_headers(); self.wfile.write(f.read_bytes())
                return
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(page)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            edits = {e: {k: p for k, p in v.items() if all(0 <= t < 1 for t in p)} for e, v in body["edits"].items()}
            edits = {e: v for e, v in edits.items() if v}
            out.write_text(json.dumps({"edits": edits, "seen": body["seen"]}, indent=1) + "\n")
            msg = f"saved: {sum(len(v) for v in edits.values())} letter edits in {len(edits)} episodes, {len(body['seen'])} reviewed"
            print(msg, flush=True)
            self.send_response(200); self.end_headers(); self.wfile.write(msg.encode())

        def log_message(self, *a):
            pass

    print(f"serving on http://127.0.0.1:{args.port}/ (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


def apply_letters(args) -> None:
    """Write each edited letter point into its pre-grasp segment and the following grasp segment."""
    from annotate_sorting_points_correction import apply_correction, prepare_correction
    from annotate_sorting_points_manual import clicked_prediction

    edits = json.loads((args.dataset_root / "letter_edits.json").read_text())["edits"]
    n = 0
    for rel, cycles in sorted(edits.items()):
        ep = args.dataset_root / rel
        rows = [json.loads(l) for l in (ep / "steps.jsonl").read_text().splitlines() if l.strip()]
        for cycle, (x, y) in sorted(cycles.items()):
            for index in (4 * int(cycle), 4 * int(cycle) + 1):
                seg = json.loads((ep / "molmo_point_annotations.json").read_text())["segments"][index]
                n += 1
                if args.dry_run:
                    continue
                context = prepare_correction(ep / rows[seg["start_frame"]]["images"]["left"], index)
                height, width = context["rgb"].shape[:2]
                prediction = clicked_prediction([float(x * width), float(y * height)], width, height)
                settings = {"annotator": "human_click_letter", "roi_xy_size": context["report"].get("roi_xy_size"),
                            "point_coordinate_system": context["report"]["point_coordinate_system"],
                            "image_width": width, "image_height": height}
                apply_correction(context, prediction, settings, method="human_click_correction", human_verified=True)
    print(f"{'would set' if args.dry_run else 'set'} {n} pre-grasp/grasp segments in {len(edits)} episodes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("click", "apply", "letters", "apply-letters"))
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.dataset_root = args.dataset_root.resolve()
    {"click": click, "apply": apply, "letters": letters, "apply-letters": apply_letters}[args.mode](args)


if __name__ == "__main__":
    main()
