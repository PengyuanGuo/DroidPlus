#!/usr/bin/env python3
"""Review page for recovery start cuts: pick the frame where the failed state begins (failure -> success only).

For every recovery episode (meta.json 'recovery') it shows the left + wrist frames of the whole raw recording
(steps.jsonl.orig when the prep step cut it), the gripper command, the arm height, the prep cut, a proposed start and
the saved pick. Saving writes the starts JSON read by scripts/trim_recovery_start.py (image numbers = raw frames).

Proposed start (rules from the user's crossword picks, 2026-11-11):
  * grasp (phase 1): the gripper closed on nothing and settled = last failed close + 20 frames, capped at the reopen
    (user picks: close + 17..21 in 16 of 20 episodes);
  * held place / stack (phase 3, start held): first frame the held object is down at its release height (within
    2 cm), i.e. the bad placement is reached, before the correction;
  * place with a staged drop (phase 3, a drop and a re-grasp): in the open stretch after the drop, 15 frames before
    the re-grasp close;
  * otherwise (no gripper failure found): the start is not proposed (0) and the episode is flagged.

    python scripts/recovery_cut_ui.py --dataset-root data/annotated/spacemouse_stacking_4phase_recovery \
        --starts scripts/recovery_starts/stacking.json --port 18769
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

PROJECT = Path(__file__).resolve().parents[1]
SETTLE = 20      # frames after the failed close
REGRASP_LEAD = 15
LOW_Z = 0.02


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def frame_no(row: dict) -> int:
    return int(re.search(r"(\d+)\.\w+$", row["images"]["wrist"])[1])


def flips(cmd):
    return [(i, c) for i, c in enumerate(cmd) if i and c != cmd[i - 1]]


def drawer_proposal(phase, cmd, closes, opens, pos, nums):
    """Drawer chicken-search recoveries (layout chicken_six_phase; phases 0 pre-open, 1 open-and-home, 3+4 grasp +
    pre-place), same idea as the crossword / stacking picks: start in the failed state, not on the way into it."""
    n = len(cmd)
    if phase == 3 and len(closes) >= 2:
        c, regrasp = closes[-2], closes[-1]
        reopen = next((i for i in opens if c < i < regrasp), None)
        if reopen is None:
            return 0, "grasp: no reopen between the closes: check by eye", True
        lift = (max(p[2] for p in pos[c:reopen + 1]) - pos[c][2]) if pos else 0.0
        if lift > 0.03:   # the chicken came up in the hand and was dropped: start after the drop (stacking rule)
            k = max(reopen, regrasp - REGRASP_LEAD)
            return k, (f"staged drop: closed {nums[c]}, lifted {lift * 100:.0f} cm, dropped {nums[reopen]}; "
                       f"{REGRASP_LEAD} before the re-grasp {nums[regrasp]}"), regrasp - reopen > 150
        k = min(c + SETTLE, reopen)
        long = regrasp - reopen > 150
        return k, (f"missed grasp: failed close {nums[c]} + {SETTLE} (reopen {nums[reopen]}, re-grasp {nums[regrasp]})"
                   + ("; long search before the re-grasp: check by eye" if long else "")), long
    if phase == 0 and len(closes) >= 2:
        c = closes[-2]
        reopen = next((i for i in opens if i > c), n - 1)
        k = min(c + SETTLE, reopen - 5)   # short closes (12-17 frames): keep a few closed frames before the reopen
        return k, f"staged close away from the handle at {nums[c]}, settled (+{k - c}, reopen {nums[reopen]})", False
    if phase in (0, 1) and pos:
        # the recording starts at the offset / mis-hooked pose (the failed state): keep it, unless the arm drives into
        # a second bad pose and backs out of it (a local minimum of the distance to the end pose, then >= 3 cm away)
        d = [sum((a - b) ** 2 for a, b in zip(p, pos[-1])) ** 0.5 for p in pos]
        best = None
        for i in range(1, n - 1):
            later = max(d[i:], default=0)
            back = max(d[i:next((j for j in range(i, n) if d[j] < d[i] - 0.02), n)], default=0)
            if back - d[i] >= 0.03 and d[i] == min(d[:i + 1]) and (best is None or d[i] < d[best]):
                best = i
        if best is not None and phase == 0:
            k = next(j for j in range(best + 1) if d[j] <= d[best] + 0.01)
            return k, (f"reaches a bad pose {d[best] * 100:.0f} cm from the end pose at {nums[best]}, backs out: "
                       f"start where it arrives ({nums[k]})"), True
        what = "offset start pose" if phase == 0 else "mis-hooked start"
        return 0, f"{what} is the failed state from frame 0: keep all" + (
            " (re-hooks mid-episode: check)" if best is not None else ""), best is not None
    return 0, "no failure found: check by eye", True


def load_episode(ep: Path, root: Path) -> dict:
    meta = json.loads((ep / "meta.json").read_text())
    rec = meta["recovery"]
    if (ep / "meta.json.pretrim").exists():
        rec_prep = json.loads((ep / "meta.json.pretrim").read_text())["recovery"]
    else:
        rec_prep = rec
    raw_p = ep / "steps.jsonl.orig" if (ep / "steps.jsonl.orig").exists() else ep / "steps.jsonl"
    raw = read_jsonl(raw_p)
    nums = [frame_no(r) for r in raw]
    cmd = [1 if r["action"]["positions"][7] > 0.5 else 0 for r in raw]
    held_until = rec.get("gripper_closed_until") or 0
    if rec.get("start_held"):
        cmd[:held_until] = [1] * held_until
    ee_p = ep / "ee_trajectory.json.orig" if (ep / "ee_trajectory.json.orig").exists() else ep / "ee_trajectory.json"
    pos = json.loads(ee_p.read_text()).get("position") or []
    pos = pos[len(pos) - len(raw):] if len(pos) >= len(raw) else []
    z = [p[2] for p in pos]
    phase = rec.get("phase")
    f = flips(cmd)
    closes = [i for i, c in f if c == 1]
    opens = [i for i, c in f if c == 0]
    prop, why, flag = 0, "no gripper failure found: check by eye", True
    if rec.get("layout") == "chicken_six_phase":
        phase = rec["phases"][0]
        prop, why, flag = drawer_proposal(phase, cmd, closes, opens, pos, nums)
    elif phase == 1 and len(closes) >= 2:
        c = closes[-2]
        reopen = next((i for i in opens if i > c), len(cmd) - 1)
        prop, why, flag = min(c + SETTLE, reopen), f"failed close at {nums[c]} + {SETTLE} (reopen {nums[reopen]})", False
    elif phase == 3 and rec.get("start_held") and len(opens) >= 2:
        drop = opens[-2]
        regrasp = next((i for i in closes if i > drop), None)
        if regrasp is not None:
            prop, why, flag = max(drop, regrasp - REGRASP_LEAD), f"after the staged drop at {nums[drop]}, {REGRASP_LEAD} before re-grasp {nums[regrasp]}", False
    elif phase == 3 and rec.get("start_held") and opens and z:
        rel = opens[-1]
        low = [i for i in range(rel) if z[i] <= z[rel] + LOW_Z]
        if low:
            prop, why, flag = low[0], f"held block first down at release height ({z[rel]:.3f} m + 2 cm), release at {nums[rel]}", False
    rel = str(ep.relative_to(root))
    prompt = (rec.get("prompts") or {}).get(str(phase)) or rec.get("phase_name") or ""
    if not prompt:
        prompt = (read_jsonl(ep / "steps.jsonl")[0].get("phase_prompt") or "") + ("  (+ pre-place)" if len(rec.get("phases") or []) > 1 else "")
    return {"ep": rel, "recording": rec.get("recording"), "phase": phase, "prompt": prompt,
            "held": bool(rec.get("start_held")), "nums": nums, "cmd": cmd,
            "z": [round(v, 4) for v in z], "prep_cut": nums[0] + (rec_prep.get("trim_start") or 0) if raw_p.name.endswith(".orig") else nums[0],
            "applied": nums[0] + (rec.get("trim_start") or 0) if raw_p.name.endswith(".orig") else nums[0],
            "proposal": nums[prop], "why": why, "flag": flag,
            "left": raw[0]["images"]["left"].split("/")[0], "wrist": raw[0]["images"]["wrist"].split("/")[0]}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Recovery cuts</title>
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#ddd;--acc:#2563eb;--ok:#15803d;--warn:#c2410c}
*{box-sizing:border-box}body{margin:0;font:14px system-ui,sans-serif;background:var(--bg);color:var(--fg)}
#wrap{display:grid;grid-template-columns:300px 1fr;height:100vh}
#side{overflow:auto;border-right:1px solid var(--line);background:var(--card)}
#side h1{font-size:15px;margin:12px}#side .sum{margin:0 12px 8px;color:var(--mut);font-size:12px}
.it{padding:6px 12px;border-top:1px solid #eee;cursor:pointer;font-size:12px}.it:hover{background:#f0f4ff}.it.cur{background:#dbe6ff}
.it b{font-weight:600}.st{float:right}.okc{color:var(--ok)}.fl{color:var(--warn)}
#main{overflow:auto;padding:12px 16px}
#hdr{font-size:13px;margin-bottom:6px}#hdr .p{font-weight:600;font-size:15px}
#imgs{display:flex;gap:8px;flex-wrap:wrap}#imgs figure{margin:0;flex:1;min-width:300px}#imgs img{width:100%;border-radius:4px;background:#000;display:block}
#imgs figcaption{font-size:11px;color:var(--mut)}
canvas{width:100%;height:120px;background:var(--card);border:1px solid var(--line);border-radius:4px;cursor:crosshair;margin-top:8px}
#ctl{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:8px}
button{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:4px;background:var(--card);cursor:pointer}
button.pri{background:var(--acc);color:#fff;border-color:var(--acc)}button:hover{filter:brightness(.95)}
#fr{font-variant-numeric:tabular-nums;font-weight:600;min-width:120px}
.leg{font-size:11px;color:var(--mut);margin-top:4px}.leg span{margin-right:12px}
.k{font-size:11px;color:var(--mut);margin-top:6px}#msg{color:var(--ok);font-size:12px;margin-left:8px}
</style></head><body><div id="wrap"><div id="side"><h1>Recovery cuts</h1><div class="sum" id="sum"></div><div id="list"></div></div>
<div id="main"><div id="hdr"></div><div id="imgs"><figure><img id="iL"><figcaption>left</figcaption></figure><figure><img id="iW"><figcaption>wrist</figcaption></figure></div>
<canvas id="tl" width="1600" height="240"></canvas>
<div class="leg"><span style="color:#111">■ gripper closed</span><span style="color:#2563eb">— arm height</span><span style="color:#999">┃ prep cut</span><span style="color:#c2410c">┃ proposal</span><span style="color:#15803d">┃ saved start</span><span>shaded = dropped</span></div>
<div id="ctl"><span id="fr"></span>
<button id="bPlay">▶ play (space)</button><button id="bSet" class="pri">Set start here (S)</button>
<button id="bProp">Use proposal (P)</button><button id="bKeep">Keep all frames</button>
<button id="bPrev">◀ prev (B)</button><button id="bNext">next (N) ▶</button><button id="bAll">Accept proposals for all unsaved</button><span id="msg"></span></div>
<div class="k">← / → one frame · shift+← / → ten frames · click the timeline to jump · Home = saved / proposed start</div></div></div>
<script>
let E=[],S={},cur=0,f=0,play=null;const $=id=>document.getElementById(id);
async function load(){E=await (await fetch('api/episodes')).json();S=await (await fetch('api/starts')).json();list();go(0)}
function start(e){return S[e.ep]!==undefined?S[e.ep]:null}
function list(){$('list').innerHTML=E.map((e,i)=>{const s=start(e);return `<div class="it ${i==cur?'cur':''}" onclick="go(${i})"><span class="st">${s!==null?'<span class=okc>✓ '+s+'</span>':(e.flag?'<span class=fl>check</span>':'')}</span><b>${e.ep.split('_').slice(-3).join('_')}</b><br>${e.prompt} · prop ${e.proposal}</div>`}).join('');
 const n=E.filter(e=>start(e)!==null).length;$('sum').textContent=`${n} / ${E.length} saved · grasp ${E.filter(e=>e.phase==1).length}, stack/place ${E.filter(e=>e.phase==3).length}`}
function go(i){cur=(i+E.length)%E.length;const e=E[cur],s=start(e);f=e.nums.indexOf(s!==null?s:e.proposal);if(f<0)f=0;
 $('hdr').innerHTML=`<div class=p>${e.prompt}</div>${e.ep} · ${e.recording} · ${e.held?'starts holding':'starts open'} · ${e.nums.length} raw frames · prep cut ${e.prep_cut} · applied now ${e.applied}<br>proposal <b>${e.proposal}</b>: ${e.why}`;
 list();show()}
function show(){const e=E[cur],n=e.nums[f];$('fr').textContent=`frame ${n} (${f+1}/${e.nums.length})`;
 const p=String(n).padStart(9,'0');$('iL').src=`img/${e.ep}/${e.left}/${p}.jpg`;$('iW').src=`img/${e.ep}/${e.wrist}/${p}.jpg`;draw();
 [1,2,3,5,8].forEach(k=>{if(f+k<e.nums.length){const q=String(e.nums[f+k]).padStart(9,'0');new Image().src=`img/${e.ep}/${e.wrist}/${q}.jpg`;new Image().src=`img/${e.ep}/${e.left}/${q}.jpg`}})}
function draw(){const e=E[cur],c=$('tl'),g=c.getContext('2d'),W=c.width,H=c.height,N=e.nums.length,x=i=>i/(N-1)*W;g.clearRect(0,0,W,H);
 const s=start(e),si=e.nums.indexOf(s!==null?s:e.proposal);g.fillStyle='rgba(0,0,0,.07)';g.fillRect(0,0,x(Math.max(si,0)),H);
 g.fillStyle='#111';for(let i=0;i<N;i++)if(e.cmd[i])g.fillRect(x(i),H-22,W/N+1,14);
 if(e.z.length){const lo=Math.min(...e.z),hi=Math.max(...e.z);g.strokeStyle='#2563eb';g.lineWidth=2;g.beginPath();e.z.forEach((v,i)=>{const y=12+(1-(v-lo)/(hi-lo+1e-9))*(H-50);i?g.lineTo(x(i),y):g.moveTo(x(i),y)});g.stroke()}
 const mk=(n,col,w)=>{const i=e.nums.indexOf(n);if(i<0)return;g.strokeStyle=col;g.lineWidth=w;g.beginPath();g.moveTo(x(i),0);g.lineTo(x(i),H);g.stroke()};
 mk(e.prep_cut,'#999',3);mk(e.proposal,'#c2410c',3);if(s!==null)mk(s,'#15803d',4);
 g.strokeStyle='#000';g.lineWidth=1;g.beginPath();g.moveTo(x(f),0);g.lineTo(x(f),H);g.stroke()}
async function save(v){const e=E[cur];const r=await fetch('api/start',{method:'POST',body:JSON.stringify({ep:e.ep,frame:v})});S=await r.json();$('msg').textContent=`saved ${e.ep.split('/').pop()}: ${v===null?'removed':v}`;list();draw()}
$('tl').onclick=ev=>{const r=ev.target.getBoundingClientRect();f=Math.round((ev.clientX-r.left)/r.width*(E[cur].nums.length-1));show()};
$('bSet').onclick=()=>save(E[cur].nums[f]);$('bProp').onclick=()=>save(E[cur].proposal);$('bKeep').onclick=()=>save(E[cur].nums[0]);
$('bNext').onclick=()=>go(cur+1);$('bPrev').onclick=()=>go(cur-1);
$('bAll').onclick=async()=>{for(const e of E)if(start(e)===null&&!e.flag){S=await (await fetch('api/start',{method:'POST',body:JSON.stringify({ep:e.ep,frame:e.proposal})})).json()}$('msg').textContent='proposals saved for unsaved, unflagged episodes';list();draw()};
$('bPlay').onclick=toggle;function toggle(){if(play){clearInterval(play);play=null;return}play=setInterval(()=>{if(f<E[cur].nums.length-1){f++;show()}else toggle()},1000/15)}
document.onkeydown=ev=>{const k=ev.key,N=E[cur].nums.length;if(k=='ArrowRight'){f=Math.min(N-1,f+(ev.shiftKey?10:1));show()}else if(k=='ArrowLeft'){f=Math.max(0,f-(ev.shiftKey?10:1));show()}
 else if(k==' '){ev.preventDefault();toggle()}else if(k=='s'||k=='S')$('bSet').click();else if(k=='p'||k=='P')$('bProp').click();else if(k=='n'||k=='N')go(cur+1);else if(k=='b'||k=='B')go(cur-1);
 else if(k=='Home'){const e=E[cur],s=start(e);f=Math.max(0,e.nums.indexOf(s!==null?s:e.proposal));show()}};
load();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--starts", type=Path, required=True)
    ap.add_argument("--port", type=int, default=18769)
    args = ap.parse_args()
    root = args.dataset_root.resolve()
    eps = sorted(p.parent for p in root.glob("*/episode_*/meta.json")
                 if "recovery" in json.loads(p.read_text()))
    data = [load_episode(ep, root) for ep in eps]
    known = {d["ep"] for d in data}
    lock = threading.Lock()
    if args.starts.exists():
        cfg = json.loads(args.starts.read_text())
    else:
        try:
            ds = str(root.relative_to(PROJECT))
        except ValueError:
            ds = str(root)
        cfg = {"_doc": "Recovery start frames (raw image numbers) picked in scripts/recovery_cut_ui.py; applied with "
                       "scripts/trim_recovery_start.py.", "dataset": ds, "starts": {}}
    cfg["_proposals"] = {d["ep"]: {"frame": d["proposal"], "why": d["why"], "flag": d["flag"]} for d in data}
    args.starts.write_text(json.dumps(cfg, indent=2))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body: bytes, ctype: str, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if "json" in ctype or "html" in ctype else "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == "/":
                return self.send(PAGE.encode(), "text/html; charset=utf-8")
            if path == "/api/episodes":
                return self.send(json.dumps(data).encode(), "application/json")
            if path == "/api/starts":
                return self.send(json.dumps(cfg["starts"]).encode(), "application/json")
            if path.startswith("/img/"):
                p = (root / path[5:]).resolve()
                if root in p.parents and p.suffix == ".jpg" and p.is_file():
                    return self.send(p.read_bytes(), "image/jpeg")
            self.send(b"not found", "text/plain", 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/start":
                return self.send(b"not found", "text/plain", 404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            ep, frame = body.get("ep"), body.get("frame")
            if ep not in known:
                return self.send(b"unknown episode", "text/plain", 400)
            with lock:
                if frame is None:
                    cfg["starts"].pop(ep, None)
                else:
                    cfg["starts"][ep] = int(frame)
                cfg["starts"] = dict(sorted(cfg["starts"].items()))
                args.starts.write_text(json.dumps(cfg, indent=2))
            self.send(json.dumps(cfg["starts"]).encode(), "application/json")

    print(f"{len(data)} recovery episodes, {sum(d['flag'] for d in data)} flagged; http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
