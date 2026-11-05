#!/usr/bin/env python3
"""Review page for stacking point labels: left MolmoPoint points + wrist projections, with the stacking_qa.py flags.

    python scripts/stacking_qa.py --dataset-root DATASET          # flags -> DATASET/stacking_qa.json
    python scripts/review_stacking.py --dataset-root DATASET --port 18765
    open http://localhost:18765

Per episode: stack order, flags, the 8 phase points on their left preview frames, the 4 wrist targets at three
frames each, and a frame player (left | wrist, both points drawn). Mark each point OK / needs fix; in fix mode a
click on a left image stores a proposed point. Everything is saved to DATASET/stacking_review.json (nothing in the
annotation files is changed; apply proposed points with the correction tool, then rerun relabel_stacking_prompts.py).
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

LOCK = threading.Lock()


def build(root: Path):
    qa = json.loads((root / "stacking_qa.json").read_text())
    return qa


class Handler(BaseHTTPRequestHandler):
    root: Path
    qa: dict

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store" if ctype == "application/json" else "max-age=3600")
        self.end_headers()
        self.wfile.write(b)

    def review(self):
        p = self.root / "stacking_review.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            return self.send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/api/list":
            with LOCK:
                rev = self.review()
            items = [{"ep": k, "variant": v["stack_variant"], "flags": len(v["flags"]), "frames": v["frames"]}
                     for k, v in self.qa.items()]
            return self.send(200, json.dumps({"episodes": items, "review": rev}))
        if u.path == "/api/ep":
            ep = q.get("ep", [""])[0]
            if ep not in self.qa:
                return self.send(404, "{}")
            rows = [json.loads(l) for l in (self.root / ep / "steps_molmo.jsonl").read_text().splitlines() if l.strip()]
            prompts = []
            frames = []
            for r in rows:
                if not prompts or prompts[-1] != r["phase_prompt"]:
                    prompts.append(r["phase_prompt"])
                frames.append([r["images"]["left"], r["images"]["wrist"], len(prompts) - 1, r.get("point_pixel"),
                               r.get("wrist_point_pixel") if r.get("wrist_point_visible") else None, bool(r.get("wrist_point_held"))])
            return self.send(200, json.dumps({"qa": self.qa[ep], "prompts": prompts, "frames": frames}))
        if u.path == "/img":
            p = (self.root / q.get("p", [""])[0]).resolve()
            if self.root not in p.parents or p.suffix.lower() not in (".jpg", ".jpeg", ".png") or not p.exists():
                return self.send(404, "{}")
            return self.send(200, p.read_bytes(), "image/jpeg")
        self.send(404, "{}")

    def do_POST(self):
        if urlparse(self.path).path != "/api/review":
            return self.send(404, "{}")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        with LOCK:
            rev = self.review()
            rev[body["ep"]] = body["data"]
            tmp = self.root / "stacking_review.json.tmp"
            tmp.write_text(json.dumps(rev, indent=1))
            tmp.replace(self.root / "stacking_review.json")
        self.send(200, "{}")


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Stacking label review</title>
<style>
:root{--bg:#12161c;--panel:#1b2129;--line:#2c3540;--ink:#e6ebf1;--muted:#93a0b0;--ok:#4cc38a;--fix:#f0883e;--flag:#f2c94c;
--red:#e5484d;--blue:#3e8ed0;--yellow:#f5d90a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13.5px/1.45 system-ui,sans-serif;display:flex;height:100vh}
#side{width:290px;border-right:1px solid var(--line);display:flex;flex-direction:column}
#side .top{padding:10px;border-bottom:1px solid var(--line);display:flex;flex-direction:column;gap:6px}
#list{overflow:auto;flex:1}
.ep{padding:6px 10px;border-bottom:1px solid var(--line);cursor:pointer;display:flex;gap:8px;align-items:center}
.ep:hover{background:var(--panel)}.ep.on{background:#24303d}
.ep .n{flex:1;font:12px ui-monospace,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.badge{font:11px ui-monospace,monospace;padding:0 6px;border-radius:8px;background:var(--flag);color:#111}
.dot{width:9px;height:9px;border-radius:50%;background:var(--line);flex:none}.dot.ok{background:var(--ok)}.dot.fix{background:var(--fix)}
#main{flex:1;overflow:auto;padding:14px 18px;display:flex;flex-direction:column;gap:14px}
h2{margin:0;font-size:17px}h3{margin:4px 0;font-size:14px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.stack{display:inline-flex;flex-direction:column-reverse;gap:2px;vertical-align:middle;margin-right:8px}.stack i{width:22px;height:12px;border-radius:2px}
.flags{background:#2a2410;border:1px solid #574a17;border-radius:6px;padding:8px 12px}.flags li{margin:2px 0}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:7px;display:flex;flex-direction:column;gap:5px}
.card.flagged{border-color:var(--flag)}.card.ok{border-color:var(--ok)}.card.fix{border-color:var(--fix)}
.card canvas{width:100%;border-radius:4px;cursor:zoom-in}.card.fixmode canvas{cursor:crosshair}
.meta{font-size:12px;color:var(--muted)}.p{font-weight:600}
.btns{display:flex;gap:5px;flex-wrap:wrap}
button{background:#2a3441;color:var(--ink);border:1px solid var(--line);border-radius:5px;padding:3px 9px;cursor:pointer;font:inherit}
button.on.okb{background:var(--ok);color:#111}button.on.fixb{background:var(--fix);color:#111}
.wgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.wthumbs{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
#player{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:8px;display:flex;flex-direction:column;gap:6px}
#pv{display:grid;grid-template-columns:1fr 1fr;gap:6px}#pv canvas{width:100%}
input[type=range]{width:100%}
textarea{width:100%;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:5px;min-height:40px}
select,input{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:4px;padding:3px}
.flagtxt{color:var(--flag);font-size:12px}
#zoom{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center;z-index:9}
#zoom canvas{max-width:95vw;max-height:92vh}
.kbd{font:11px ui-monospace,monospace;color:var(--muted)}
</style></head><body>
<div id="side"><div class="top">
<b>Stacking label review</b>
<select id="filt"><option value="flagged">flagged (second look)</option><option value="all">all episodes</option>
<option value="todo">not reviewed</option><option value="fix">marked needs fix</option></select>
<select id="var"><option value="">all stack orders</option></select>
<div class="meta" id="count"></div><div class="kbd">j / k: next / previous episode · space: play</div>
</div><div id="list"></div></div>
<div id="main"><div class="meta">Pick an episode.</div></div>
<div id="zoom"><canvas id="zc"></canvas></div>
<script>
const COL={red:'#e5484d',blue:'#3e8ed0',yellow:'#f5d90a'};
let EPS=[],REV={},CUR=null,DATA=null,timer=null,frame=0;
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const img=p=>'/img?p='+encodeURIComponent(CUR+'/'+p);
const IMC={};function loadImg(src){if(!IMC[src]){const i=new Image();i.src=src;IMC[src]=i;}return IMC[src];}
function rev(){return REV[CUR]||(REV[CUR]={reviewed:false,note:'',segments:{},wrist:{}});}
async function save(){await fetch('/api/review',{method:'POST',body:JSON.stringify({ep:CUR,data:rev()})});renderList();}
async function init(){const j=await (await fetch('/api/list')).json();EPS=j.episodes;REV=j.review;
 [...new Set(EPS.map(e=>e.variant))].sort().forEach(v=>$('#var').add(new Option(v.split('<').join(' < '),v)));renderList();}
function status(e){const r=REV[e.ep];if(!r)return'';const s=Object.values(r.segments||{}).concat(Object.values(r.wrist||{}));
 if(s.some(x=>x.status==='fix'))return'fix';return r.reviewed?'ok':'';}
function visible(){const f=$('#filt').value,v=$('#var').value;return EPS.filter(e=>(!v||e.variant===v)&&(f==='all'||(f==='flagged'&&e.flags)||(f==='todo'&&!(REV[e.ep]||{}).reviewed)||(f==='fix'&&status(e)==='fix')));}
function renderList(){const vs=visible();$('#count').textContent=`${vs.length} shown · ${EPS.filter(e=>e.flags).length} flagged · ${EPS.filter(e=>(REV[e.ep]||{}).reviewed).length}/${EPS.length} reviewed`;
 $('#list').innerHTML=vs.map(e=>`<div class="ep ${e.ep===CUR?'on':''}" data-ep="${esc(e.ep)}"><span class="dot ${status(e)}"></span><span class="n" title="${esc(e.ep)}">${esc(e.ep.replace('stacking_4phase_teleop_20261102_',''))}</span>${stackHtml(e.variant)}${e.flags?`<span class="badge">${e.flags}</span>`:''}</div>`).join('');
 $('#list').querySelectorAll('.ep').forEach(d=>d.onclick=()=>open(d.dataset.ep));}
function stackHtml(v){return `<span class="stack">${(v||'').split('<').map(c=>`<i style="background:${COL[c]}"></i>`).join('')}</span>`;}
function cross(ctx,x,y,w,h,c,r=10){const path=()=>{ctx.beginPath();ctx.arc(x*w,y*h,r,0,7);ctx.moveTo(x*w-r-6,y*h);ctx.lineTo(x*w+r+6,y*h);ctx.moveTo(x*w,y*h-r-6);ctx.lineTo(x*w,y*h+r+6);};
 path();ctx.strokeStyle='#000';ctx.lineWidth=5;ctx.stroke();path();ctx.strokeStyle=c;ctx.lineWidth=2.5;ctx.stroke();}
function draw(cv,src,pts){const im=loadImg(src);const go=()=>{cv.width=im.naturalWidth;cv.height=im.naturalHeight;const c=cv.getContext('2d');c.drawImage(im,0,0);
 pts.forEach(p=>p&&p.xy&&cross(c,p.xy[0],p.xy[1],cv.width,cv.height,p.c,p.r||10));};im.complete&&im.naturalWidth?go():im.addEventListener('load',go,{once:true});}
async function open(ep){CUR=ep;clearInterval(timer);timer=null;DATA=await (await fetch('/api/ep?ep='+encodeURIComponent(ep))).json();frame=0;renderList();renderEp();}
function renderEp(){const q=DATA.qa,r=rev();
 const segs=q.segments.map(s=>{const st=(r.segments[s.i]||{}).status||'',prop=(r.segments[s.i]||{}).point;
  return `<div class="card ${s.flags.length?'flagged':''} ${st}" data-i="${s.i}"><div class="p">${s.i+1}. ${esc(s.prompt)}</div>
  <canvas data-seg="${s.i}"></canvas>
  <div class="meta">frame ${s.preview_frame} · ${s.copied_from!=null?'copied from phase '+s.copied_from:`inliers ${s.inliers??'—'}/${s.required??'—'}`} · colour ${Math.round(100*s.colour_share)}%${s.stack_shift!=null?` · stack shift ${s.stack_shift}`:''}</div>
  ${s.flags.map(f=>`<div class="flagtxt">⚑ ${esc(f)}</div>`).join('')}${prop?`<div class="meta">proposed point ${prop.map(v=>v.toFixed(3)).join(', ')}</div>`:''}
  <div class="btns"><button class="okb ${st==='ok'?'on':''}" data-s="ok">OK</button><button class="fixb ${st==='fix'?'on':''}" data-s="fix">Needs fix</button>${st==='fix'?'<span class="meta">click the image to propose a point</span>':''}</div></div>`;}).join('');
 const wins=q.wrist_windows.map(w=>{const st=(r.wrist[w.k]||{}).status||'';const end=w.contact||w.end;
  const fr=[w.start,Math.round(w.start+0.5*(end-w.start)),Math.max(w.start,end-6)];
  return `<div class="card ${w.flags.length?'flagged':''} ${st}" data-k="${w.k}"><div class="p">wrist ${w.k+1}: ${w.role==='object'?'grasp':'stack target'} · ${esc(w.colour)} block</div>
  <div class="wthumbs">${fr.map(f=>`<canvas data-wf="${f}"></canvas>`).join('')}</div>
  <div class="meta">frames ${w.start}–${w.end} · contact ${w.contact??'none'} · visible ${Math.round(100*w.visible)}%${w.colour_hit!=null?` · on colour ${Math.round(100*w.colour_hit)}%`:''}${w.stack_dz_m!=null?` · vs 1st stack: ${(100*w.stack_dxy_m).toFixed(1)} cm side, ${(100*w.stack_dz_m).toFixed(1)} cm up`:''}</div>
  ${w.flags.map(f=>`<div class="flagtxt">⚑ ${esc(f)}</div>`).join('')}
  <div class="btns"><button class="okb ${st==='ok'?'on':''}" data-s="ok">OK</button><button class="fixb ${st==='fix'?'on':''}" data-s="fix">Needs fix</button></div></div>`;}).join('');
 $('#main').innerHTML=`<div><h2>${stackHtml(q.stack_variant)}${esc(CUR)}</h2><div class="meta">cyan = label · orange = wrist target while the block is held · ${q.frames} frames · stack (bottom → top): ${esc((q.stack_variant||'').split('<').join(' → '))}</div></div>
 ${q.flags.length?`<ul class="flags">${q.flags.map(f=>`<li>${esc(f)}</li>`).join('')}</ul>`:'<div class="meta">No automatic flags.</div>'}
 <div id="player"><div class="btns"><button id="play">▶ play</button><span class="meta" id="pinfo"></span></div>
 <div id="pv"><canvas id="pl"></canvas><canvas id="pw"></canvas></div><input type="range" id="scrub" min="0" max="${DATA.frames.length-1}" value="0"></div>
 <h3>Left camera: one point per phase (cyan = label, green = your proposed point)</h3><div class="grid">${segs}</div>
 <h3>Wrist camera: projected targets (start · middle · before contact)</h3><div class="wgrid">${wins}</div>
 <div class="card"><b>Episode</b><textarea id="note" placeholder="note">${esc(r.note)}</textarea>
 <div class="btns"><button class="okb ${r.reviewed?'on':''}" id="rev">${r.reviewed?'✓ reviewed':'Mark episode reviewed'}</button><button id="allok">All points OK + reviewed</button><button id="next">Next episode →</button></div></div>`;
 q.segments.forEach(s=>{const cv=document.querySelector(`canvas[data-seg="${s.i}"]`);const prop=(r.segments[s.i]||{}).point;
  draw(cv,img(DATA.frames[s.preview_frame][0]),[{xy:s.point,c:'#00e5ff'},prop&&{xy:prop,c:'#30d158'}]);
  cv.onclick=e=>{const st=(r.segments[s.i]||{}).status;if(st==='fix'){const b=cv.getBoundingClientRect();r.segments[s.i]={status:'fix',point:[(e.clientX-b.left)/b.width,(e.clientY-b.top)/b.height]};save();renderEp();}else zoom(cv);};});
 document.querySelectorAll('canvas[data-wf]').forEach(cv=>{const f=+cv.dataset.wf,fr=DATA.frames[f];draw(cv,img(fr[1]),[{xy:fr[4],c:fr[5]?'#ffa500':'#00e5ff',r:12}]);cv.onclick=()=>zoom(cv);});
 document.querySelectorAll('.card[data-i] button[data-s]').forEach(b=>b.onclick=()=>{const i=b.closest('.card').dataset.i,cur=(r.segments[i]||{});r.segments[i]={...cur,status:cur.status===b.dataset.s?'':b.dataset.s};save();renderEp();});
 document.querySelectorAll('.card[data-k] button[data-s]').forEach(b=>b.onclick=()=>{const k=b.closest('.card').dataset.k,cur=(r.wrist[k]||{});r.wrist[k]={status:cur.status===b.dataset.s?'':b.dataset.s};save();renderEp();});
 $('#note').onchange=e=>{r.note=e.target.value;save();};
 $('#rev').onclick=()=>{r.reviewed=!r.reviewed;save();renderEp();};
 $('#allok').onclick=()=>{q.segments.forEach(s=>{if(!(r.segments[s.i]||{}).status)r.segments[s.i]={status:'ok'};});q.wrist_windows.forEach(w=>{if(!(r.wrist[w.k]||{}).status)r.wrist[w.k]={status:'ok'};});r.reviewed=true;save();renderEp();};
 $('#next').onclick=()=>step(1);
 $('#scrub').oninput=e=>{frame=+e.target.value;drawFrame();};$('#play').onclick=togglePlay;drawFrame();}
function drawFrame(){const fr=DATA.frames[frame];if(!fr)return;draw($('#pl'),img(fr[0]),[{xy:fr[3],c:'#00e5ff'}]);draw($('#pw'),img(fr[1]),[{xy:fr[4],c:fr[5]?'#ffa500':'#00e5ff',r:12}]);
 $('#pinfo').textContent=`frame ${frame} · ${DATA.prompts[fr[2]]}${fr[5]?' · block held':''}${fr[4]?'':' · wrist point not visible'}`;$('#scrub').value=frame;
 for(let k=1;k<=8;k++){const n=DATA.frames[frame+k];if(n){loadImg(img(n[0]));loadImg(img(n[1]));}}}
function togglePlay(){if(timer){clearInterval(timer);timer=null;$('#play').textContent='▶ play';return;}$('#play').textContent='❚❚ pause';
 timer=setInterval(()=>{frame=(frame+2)%DATA.frames.length;drawFrame();},133);}
function zoom(cv){const z=$('#zc');z.width=cv.width;z.height=cv.height;z.getContext('2d').drawImage(cv,0,0);$('#zoom').style.display='flex';}
$('#zoom').onclick=()=>$('#zoom').style.display='none';
function step(d){const vs=visible();const i=vs.findIndex(e=>e.ep===CUR);const n=vs[Math.max(0,Math.min(vs.length-1,i+d))];if(n)open(n.ep);}
document.addEventListener('keydown',e=>{if(['TEXTAREA','INPUT','SELECT'].includes(e.target.tagName))return;if(e.key==='j')step(1);if(e.key==='k')step(-1);if(e.key===' '&&DATA){e.preventDefault();togglePlay();}if(e.key==='Escape')$('#zoom').style.display='none';});
$('#filt').onchange=renderList;$('#var').onchange=renderList;init();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--port", type=int, default=18765)
    a = ap.parse_args()
    Handler.root = a.dataset_root.resolve()
    Handler.qa = build(Handler.root)
    print(f"stacking review on http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
