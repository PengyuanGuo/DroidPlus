#!/usr/bin/env python3
"""Calibrate the static left (exo) camera from annotated episodes with wrist labels (no calibration board).

Every target window of `annotate_wrist_points.py` pairs a left-image point with a 3D point in the robot base frame:

    object window (pre-grasp, grasp)  left: the pre-grasp segment's point_pixel (MolmoPoint on the cube)
                                      3D:   fingertip centre at the durable close (wrist.windows[].target_base_m)
    place window (pre-place, place)   left: the pre-place segment's point_pixel (crossword: the placed cube, after release)
                                      3D:   fingertip centre at the release = target_base_m + [0, 0, place_drop_m]

The points lie near one plane, so the intrinsics are fixed (ZED factory calibration of the rectified left image,
--intrinsics) and only the pose is fitted: cv2.solvePnPRansac, then an iterative refinement on the inliers. A second
fit with fx / fy free checks the intrinsics. Residuals are reported per session (teleop folder); each session is
also held out of a refit once (leave-one-session-out). Last, the place targets are checked the way deployment uses
them: left pixel -> ray -> plane at the release height -> lowered by the drop -> projected into the window's wrist
frames, against the hindsight wrist labels.

Writes <dataset>/exo_camera_calibration.json (atomic_policy.wrist.geometry.ExoCamera + fit statistics).

Usage (any env with numpy + opencv + scipy, e.g. DroidPlus/.venv-molmo):
    python scripts/calibrate_exo_camera.py --dataset-root data/annotated/spacemouse_crossword_mars_4phase \
        --atomic-policy-root ../atomic_policy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

PROJECT = Path(__file__).resolve().parents[1]
# ZED serial 31262010, /usr/local/zed/settings/SN31262010.conf [LEFT_CAM_VGA] (rectified 672x376 frames)
ZED_LEFT_VGA = (267.0675, 267.025, 340.865, 190.6465)


def collect(dataset: Path):
    """Correspondences (one per target window) and the per-window data for the wrist check."""
    pairs, windows = [], []
    for ann_path in sorted(dataset.glob('*/episode_*/molmo_point_annotations.json')):
        ep = ann_path.parent
        ann = json.loads(ann_path.read_text())
        wrist = ann.get('wrist')
        if not wrist:
            continue
        drop = float(wrist.get('place_drop_m', 0.0))
        segs = ann['segments']
        for w in wrist['windows']:
            if w.get('target_base_m') is None:
                continue
            seg = segs[w['segments'][0]]
            uv = seg.get('point_pixel')
            if uv is None:
                continue
            X = np.asarray(w['target_base_m'], np.float64)
            if w['role'] == 'place':
                X = X + [0.0, 0.0, drop]
            rec = dict(session=ep.parent.name, episode=str(ep.relative_to(dataset)), role=w['role'], uv=np.asarray(uv),
                       X=X, drop=drop if w['role'] == 'place' else 0.0, start=w['start_frame'], end=w['end_frame'],
                       phase=seg['phase_index'])
            pairs.append(rec)
    return pairs


def fit_pose(X, px, K, rvec=None, tvec=None, ransac_px=8.0):
    if rvec is None:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(X, px, K, None, reprojectionError=ransac_px, iterationsCount=2000,
                                                 flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            raise RuntimeError('solvePnPRansac failed')
        inl = inl[:, 0]
    else:
        inl = np.arange(len(X))
    ok, rvec, tvec = cv2.solvePnP(X[inl], px[inl], K, None, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    err = np.linalg.norm(cv2.projectPoints(X, rvec, tvec, K, None)[0][:, 0] - px, axis=1)
    inl = np.nonzero(err <= ransac_px)[0]
    ok, rvec, tvec = cv2.solvePnP(X[inl], px[inl], K, None, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    err = np.linalg.norm(cv2.projectPoints(X, rvec, tvec, K, None)[0][:, 0] - px, axis=1)
    return rvec[:, 0], tvec[:, 0], err, err <= ransac_px


def fit_free_focal(X, px, K, rvec, tvec):
    """fx, fy and the pose free (principal point fixed), robust loss: the intrinsics check."""
    def res(p):
        Kp = K.copy(); Kp[0, 0], Kp[1, 1] = p[6], p[7]
        return (cv2.projectPoints(X, p[:3], p[3:6], Kp, None)[0][:, 0] - px).ravel()
    p0 = np.r_[rvec, tvec, K[0, 0], K[1, 1]]
    r = least_squares(res, p0, loss='soft_l1', f_scale=4.0)
    e = np.linalg.norm(r.fun.reshape(-1, 2), axis=1)
    return r.x[6], r.x[7], e


def stats(e):
    e = np.asarray(e)
    return {'n': int(len(e)), 'median_px': round(float(np.median(e)), 2), 'p90_px': round(float(np.percentile(e, 90)), 2),
            'over_25px': round(float(np.mean(e > 25)), 3)} if len(e) else {'n': 0}


def wrist_check(dataset: Path, pairs, exo, wrist_cam, panda_fk):
    """Place windows: left pixel -> exo ray -> plane at the release height -> minus the drop -> wrist frames, against
    the hindsight labels (both on the full wrist image, px). Also the 3D distance to the hindsight target."""
    errs, d3, cache = [], [], {}
    W, H = wrist_cam.width, wrist_cam.height
    for p in pairs:
        if p['role'] != 'place':
            continue
        Xe = exo.lift_to_plane(p['uv'], p['X'][2])
        if Xe is None:
            continue
        Xe = Xe - [0.0, 0.0, p['drop']]
        d3.append(float(np.linalg.norm((Xe - (p['X'] - [0, 0, p['drop']]))[:2])))
        if p['episode'] not in cache:
            cache[p['episode']] = [json.loads(l) for l in (dataset / p['episode'] / 'steps_molmo.jsonl').read_text().splitlines() if l.strip()]
        rows = cache[p['episode']]
        for r in rows[p['start']:p['end']]:
            if not r.get('wrist_point_visible'):
                continue
            uv, inside, _ = wrist_cam.project(Xe, panda_fk(r['state']['positions'][:7]))
            if inside:
                errs.append(float(np.hypot((uv[0] - r['wrist_point_pixel'][0]) * W, (uv[1] - r['wrist_point_pixel'][1]) * H)))
    return errs, d3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', type=Path, required=True)
    ap.add_argument('--atomic-policy-root', type=Path, default=PROJECT.parent / 'atomic_policy')
    ap.add_argument('--intrinsics', nargs=4, type=float, default=ZED_LEFT_VGA, metavar=('FX', 'FY', 'CX', 'CY'))
    ap.add_argument('--size', nargs=2, type=int, default=[672, 376], metavar=('W', 'H'))
    ap.add_argument('--ransac-px', type=float, default=8.0)
    ap.add_argument('--wrist-calibration', type=Path, default=None, help='default: <dataset>/wrist_camera_calibration.json')
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    sys.path.insert(0, str(args.atomic_policy_root.resolve()))
    from atomic_policy.wrist.geometry import ExoCamera, WristCamera, panda_fk

    ds = args.dataset_root.resolve()
    pairs = collect(ds)
    if len(pairs) < 12:
        raise SystemExit(f'only {len(pairs)} correspondences under {ds}; run annotate_wrist_points.py first')
    Wd, Ht = args.size
    fx, fy, cx, cy = args.intrinsics
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    X = np.array([p['X'] for p in pairs])
    px = np.array([p['uv'] for p in pairs]) * [Wd, Ht]
    sessions = np.array([p['session'] for p in pairs])
    roles = np.array([p['role'] for p in pairs])

    rvec, tvec, err, inl = fit_pose(X, px, K, ransac_px=args.ransac_px)
    print(f'{len(pairs)} correspondences ({(roles == "object").sum()} object, {(roles == "place").sum()} place), '
          f'{len(set(sessions))} sessions; z range {X[:, 2].min():.3f}..{X[:, 2].max():.3f} m')
    print(f'pose fit (fixed intrinsics): inliers {inl.sum()}/{len(inl)}, all {stats(err)}')
    by_role = {r: stats(err[roles == r]) for r in ('object', 'place')}
    print('  by role:', by_role)

    ffx, ffy, ferr = fit_free_focal(X[inl], px[inl], K, rvec, tvec)
    print(f'intrinsics check (fx, fy free): fx {ffx:.1f} ({100 * (ffx / fx - 1):+.1f}%), fy {ffy:.1f} '
          f'({100 * (ffy / fy - 1):+.1f}%), inlier median {np.median(ferr):.2f} px')

    per_session, holdout = {}, {}
    for s in sorted(set(sessions)):
        m = sessions == s
        per_session[s] = stats(err[m])
        if (~m).sum() >= 12:
            r2, t2, _, _ = fit_pose(X[~m], px[~m], K, ransac_px=args.ransac_px)
            e2 = np.linalg.norm(cv2.projectPoints(X[m], r2, t2, K, None)[0][:, 0] - px[m], axis=1)
            holdout[s] = stats(e2)
    print('per session (all-session fit) | held out of the fit:')
    for s in per_session:
        print(f'  {s}: {per_session[s]} | {holdout.get(s)}')
    ho = [holdout[s]['median_px'] for s in holdout]

    cam = ExoCamera(fx, fy, cx, cy, rvec.tolist(), tvec.tolist(), Wd, Ht)
    cam_pos = (-cv2.Rodrigues(rvec)[0].T @ tvec)
    print(f'camera centre in base: {np.round(cam_pos, 3).tolist()} m')

    wpath = args.wrist_calibration or ds / 'wrist_camera_calibration.json'
    check = {}
    if wpath.exists():
        e, d3 = wrist_check(ds, pairs, cam, WristCamera.load(wpath), panda_fk)
        check = {'wrist_px': stats(e), 'xy_cm_median': round(100 * float(np.median(d3)), 2),
                 'xy_cm_p90': round(100 * float(np.percentile(d3, 90)), 2)}
        print(f'place targets from the left point, in the wrist frames vs the hindsight labels: {check["wrist_px"]}; '
              f'3D xy error median {check["xy_cm_median"]} cm, p90 {check["xy_cm_p90"]} cm')

    files = sorted(ds.glob('*/episode_*/molmo_point_annotations.json'))
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    cam.info = {
        'method': 'board-free PnP on annotated targets (DroidPlus scripts/calibrate_exo_camera.py)',
        'date': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'dataset': str(ds), 'annotations_sha256': h.hexdigest(), 'n_annotation_files': len(files),
        'intrinsics': 'ZED factory [LEFT_CAM_VGA] (rectified, zero distortion), fixed in the fit',
        'n_pairs': len(pairs), 'n_inliers': int(inl.sum()), 'ransac_px': args.ransac_px,
        'reprojection_all': stats(err), 'reprojection_by_role': by_role,
        'free_focal_check': {'fx': round(float(ffx), 2), 'fy': round(float(ffy), 2)},
        'per_session': per_session, 'leave_one_session_out': holdout,
        'leave_one_session_out_median_px': round(float(np.median(ho)), 2) if ho else None,
        'camera_centre_base_m': np.round(cam_pos, 4).tolist(), 'place_check': check,
    }
    out = args.output or ds / 'exo_camera_calibration.json'
    cam.save(out)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
