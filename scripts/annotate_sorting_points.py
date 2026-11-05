#!/usr/bin/env python3
"""
Ground real sorting recordings with MolmoPoint; preserve original steps.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np

LOG = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parents[1]
# Handle prompts often land on the black table clamp left of the drawer ("L-shaped"
# is worse: the clamp is L-shaped too); the handle's black front panel is found reliably.
HANDLE = ("Point to the black panel on the front of the white drawer",
          "Point to the black handle on the front of the white drawer")
CLAMP = "Point to the black clamp on the table"
# Drawer/cube six-phase recordings: phase_prompt order -> (primary, fallback) pointing prompts.
DRAWER_PHASES = {
    "grasp_handle_open": HANDLE,
    "open_drawer_and_home": HANDLE,
    "pick_cube_and_lift": ("Point to the orange cube", "Point to the cube"),
    # "inside of the open drawer" returns no point; "the drawer" lands off-target.
    "carry_place_and_home": ("Point to where the cube should be placed in the drawer", "Point to the open drawer"),
    "grasp_handle_close": HANDLE,
    "close_drawer_and_home": HANDLE,
}


# Drawer chicken-search six-phase recordings (two drawer units side by side; the operator
# may open one drawer, find it empty, and open the other: phases 0,1 can repeat).
CHICKEN_HANDLES = "Point to the black drawer handles"
CHICKEN_SIDE_HANDLE = "Point to the black handle on the {side} white drawer"
GRIPPER = "Point to the robot gripper"
CHICKEN_PHASES = {
    "pre-open": (CHICKEN_HANDLES, CHICKEN_HANDLES),  # side-specific prompt chosen per segment
    "open-and-home": (CHICKEN_HANDLES, CHICKEN_HANDLES),  # copied from the preceding pre-open
    "pre-grasp": ("Point to the chicken leg", "Point to the orange chicken leg in the drawer"),
    "grasp": ("Point to the chicken leg", "Point to the orange chicken leg"),
    "pre-place": ("Point to the inside of the blue bin", "Point to the blue bin"),
    "place": ("Point to the inside of the blue bin", "Point to the blue bin"),
}
CHICKEN_NEXT = {0: {1}, 1: {0, 2}, 2: {3}, 3: {4}, 4: {5}}


# Letter-cube crossword recordings: plain "pre-grasp"/"grasp"/"pre-place"/"place" cycles, one per
# letter of --word, in order; cycle k moves letter word[k] into grid cell k+1.
LETTER_PHASES = ("pre-grasp", "grasp", "pre-place", "place")
ALL_CUBES = "Point to all the white letter cubes on the table"


def letter_prompts(letter: str) -> tuple[str, str]:
    return f"Point to the letter {letter} cube on the table", f"Point to the cube with the letter {letter}"


def cell_prompts(cell: int) -> tuple[str, str]:
    return f"Point to the blue grid cell labeled {cell}", f"Point to the square numbered {cell} on the blue grid"


# Descriptive renames written later for training ("grasp letter M", "pre-place to slot 1",
# "pre-open the left drawer", ...) map back to the base phase names used by this pipeline.
RENAMED_PROMPTS = [
    (r"(pre-grasp|grasp) letter [A-Z]", lambda m: m[1]),
    (r"(pre-place|place) to slot \d+", lambda m: m[1]),
    (r"pre-open the (left|right) drawer", lambda m: "pre-open"),
    (r"open the (left|right) drawer and home", lambda m: "open-and-home"),
    (r"(pre-grasp|grasp) chicken leg", lambda m: m[1]),
    (r"(pre-place|place) chicken leg to the blue bin", lambda m: m[1]),
]


def canonical_prompt(phase: int, text: str) -> str:
    for pattern, base in RENAMED_PROMPTS:
        match = re.fullmatch(pattern, text)
        if match:
            name = base(match)
            if (phase < len(LETTER_PHASES) and name == LETTER_PHASES[phase]) or \
                    (name in CHICKEN_PHASES and list(CHICKEN_PHASES).index(name) == phase):
                return name
    return text


def pointing_prompts(phase: int, text: str) -> tuple[str, str]:
    text = canonical_prompt(phase, text)
    if phase < len(LETTER_PHASES) and text == LETTER_PHASES[phase]:
        return ("letter cube (chosen per cycle)", "letter cube (chosen per cycle)")
    if text in CHICKEN_PHASES and list(CHICKEN_PHASES).index(text) == phase:
        return CHICKEN_PHASES[text]
    if text in DRAWER_PHASES and list(DRAWER_PHASES).index(text) == phase:
        return DRAWER_PHASES[text]
    # Four-phase collection prompts omit the destination; use this dataset's
    # sorting rule, never infer a destination from the object's current position.
    prefixes = {0: "pre-grasp", 1: "grasp", 2: "pre-place", 3: "place"}
    if phase in prefixes:
        match = re.fullmatch(rf"{prefixes[phase]} (banana|orange|apple|salt bottle)", text)
        if match:
            obj = match[1]
            if phase < 2:
                return f"Point to the {obj} on the table", f"Point to the {obj}"
            color = "white" if obj == "salt bottle" else "blue"
            return f"Point to the inside of the {color} bin", f"Point to the {color} bin"
    if phase == 0:
        match = re.fullmatch(r"grasp (banana|orange|apple|salt bottle)", text)
        if match:
            obj = match[1]
            return f"Point to the {obj} on the table", f"Point to the {obj}"
    if phase == 1:
        match = re.fullmatch(
            r"place (banana|orange|apple|salt bottle) to the (blue|white) bin", text,
        )
        if match:
            color = match[2]
            return f"Point to the inside of the {color} bin", f"Point to the {color} bin"
    raise ValueError(f"Unsupported phase/prompt: {phase!r}, {text!r}")


def read_episode(path: Path) -> tuple[list[dict], list[dict]]:
    rows = []
    segments = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON (remove plain-text order notes)") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a frame object")
        phase = row.get("phase_index")
        prompt = row.get("phase_prompt")
        if type(phase) is not int or phase not in range(len(DRAWER_PHASES)) or not isinstance(prompt, str):
            raise ValueError(f"{path}:{number}: missing/invalid phase_index or phase_prompt")
        primary, fallback = pointing_prompts(phase, prompt)
        if not segments or segments[-1]["phase_index"] != phase:
            segments.append({
                "phase_index": phase, "phase_prompt": prompt,
                "start_frame": len(rows), "end_frame": len(rows),
                "molmo_prompt": primary, "fallback_prompt": fallback,
            })
        if segments[-1]["phase_prompt"] != prompt:
            raise ValueError(f"{path}:{number}: prompt changed without a phase boundary")
        segments[-1]["end_frame"] = len(rows) + 1
        relative = row.get("images", {}).get("left")
        if not isinstance(relative, str) or not (path.parent / relative).is_file():
            raise ValueError(f"{path}:{number}: missing left-camera image {relative!r}")
        rows.append(row)
    first = canonical_prompt(segments[0]["phase_index"], segments[0]["phase_prompt"]) if segments else None
    if segments and first == LETTER_PHASES[0]:
        labels = [segment["phase_index"] for segment in segments]
        if not labels or len(labels) % 4 or labels != [0, 1, 2, 3] * (len(labels) // 4):
            raise ValueError(f"{path}: expected complete letter cycles 0,1,2,3; got {labels}")
        return rows, segments
    if segments and first == "pre-open":
        labels = [segment["phase_index"] for segment in segments]
        legal = all(b in CHICKEN_NEXT.get(a, set()) for a, b in zip(labels, labels[1:]))
        if not legal or labels[0] != 0 or labels[-1] != 5 or set(labels) != set(range(6)):
            raise ValueError(f"{path}: expected chicken phases (0,1)+ then 2,3,4,5; got {labels}")
        return rows, segments
    if segments and segments[0]["phase_prompt"] in DRAWER_PHASES:
        if [segment["phase_prompt"] for segment in segments] != list(DRAWER_PHASES):
            raise ValueError(f"{path}: expected six drawer phases {list(DRAWER_PHASES)} in order")
        return rows, segments
    four_phase = bool(segments and segments[0]["phase_prompt"].startswith("pre-grasp "))
    if four_phase:
        labels = [segment["phase_index"] for segment in segments]
        if len(labels) % 4 or labels != [0, 1, 2, 3] * (len(labels) // 4):
            raise ValueError(f"{path}: expected complete four-phase cycles 0,1,2,3")
        for start in range(0, len(segments), 4):
            obj = segments[start]["phase_prompt"].removeprefix("pre-grasp ")
            expected = [f"{prefix} {obj}" for prefix in ("pre-grasp", "grasp", "pre-place", "place")]
            if [s["phase_prompt"] for s in segments[start:start + 4]] != expected:
                raise ValueError(f"{path}: four-phase object/prompt mismatch")
        return rows, segments
    if [segment["phase_index"] for segment in segments] != [0, 1, 0, 1, 0, 1]:
        raise ValueError(f"{path}: expected six consecutive phase segments 0,1,0,1,0,1")
    for grasp, place in zip(segments[::2], segments[1::2]):
        obj = grasp["phase_prompt"].removeprefix("grasp ")
        color = "white" if obj == "salt bottle" else "blue"
        if place["phase_prompt"] != f"place {obj} to the {color} bin":
            raise ValueError(f"{path}: grasp/place object or destination mismatch")
    return rows, segments


def sample_indices(start: int, end: int, count: int, skip: int) -> list[int]:
    """Spread unique samples across the first quarter; cap skip for short phases."""
    stop = min(end, start + max(1, int(np.ceil((end - start) * 0.25))))
    first = min(start + skip, stop - 1)
    return np.unique(np.linspace(first, stop - 1, min(count, stop - first), dtype=int)).tolist()


def read_rgb(path: Path, roi: list[int] | None) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot decode image: {path}")
    if roi is not None:
        x, y, size = roi
        height, width = bgr.shape[:2]
        if x < 0 or y < 0 or size <= 0 or x + size > width or y + size > height:
            raise ValueError(f"ROI {roi} is outside {width}x{height} image: {path}")
        bgr = bgr[y:y + size, x:x + size]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def window(anchor: list[float], dx_range: tuple[float, float], dy_range: tuple[float, float]):
    def accept(point: np.ndarray) -> bool:
        dx, dy = point[0] - anchor[0], point[1] - anchor[1]
        return dx_range[0] <= dx <= dx_range[1] and dy_range[0] <= dy <= dy_range[1]
    return accept


def handle_window(anchor: list[float]):
    """Opened-drawer handle lies right of/below the closed handle; the table clamp is left of it."""
    return window(anchor, (-0.03, 0.10), (-0.02, 0.20))


def placement_window(anchor: list[float]):
    """The drawer opening sits just above the closed handle; the cube on the table or in the gripper does not."""
    return window(anchor, (-0.03, 0.10), (-0.10, 0.03))


def copy_segment(segment: dict, source: dict, args) -> dict:
    """Use the handle point from the grasp phase; the gripper hides the handle while it pulls/pushes."""
    return {
        **segment, "sample_indices": sample_indices(segment["start_frame"], segment["end_frame"], args.samples, args.skip_first_frames),
        "samples": [], "copied_from_phase": source["phase_index"], "valid_predictions": 0, "inliers": 0,
        "required_inliers": 0, "point_pixel": source["point_pixel"], "needs_review": source["needs_review"],
    }


def locate_clamp(grounder, episode: Path, rows: list[dict], segment: dict, args) -> list[float] | None:
    index = sample_indices(segment["start_frame"], segment["end_frame"], 1, args.skip_first_frames)[0]
    rgb = read_rgb(episode / rows[index]["images"]["left"], args.roi)
    try:
        pixel = np.asarray(grounder._point(rgb, CLAMP), dtype=np.float32)
    except Exception as exc:
        if isinstance(exc, MemoryError) or "cuda" in str(exc).lower():
            raise
        return None
    return (pixel / np.array([rgb.shape[1], rgb.shape[0]], dtype=np.float32)).tolist()


def ground_episode(grounder, episode: Path, rows: list[dict], segments: list[dict], args, log=lambda *a: None) -> list[dict]:
    results = []
    for number, segment in enumerate(segments):
        log(number, segment)
        if number == 0:
            # The closed handle sits right of the clamp; reject hits on the clamp itself.
            clamp = locate_clamp(grounder, episode, rows, segment, args)
            accept = None if clamp is None else (lambda point: point[0] - clamp[0] >= 0.025)
            result = {**ground_segment(grounder, episode, rows, segment, args, accept), "clamp_point": clamp}
        elif segment["phase_prompt"] in ("open_drawer_and_home", "close_drawer_and_home"):
            result = copy_segment(segment, results[-1], args)
        elif segment["phase_prompt"] == "carry_place_and_home" and not results[0]["needs_review"]:
            # The carried cube blocks the drawer early in this phase, so search the whole
            # phase plus the start of the next one, when the arm has moved away.
            after = segments[number + 1]
            frames = np.linspace(segment["start_frame"] + args.skip_first_frames, segment["end_frame"] - 1, args.samples * 2, dtype=int).tolist()
            frames += np.linspace(after["start_frame"], after["start_frame"] + (after["end_frame"] - after["start_frame"]) // 4, 4, dtype=int).tolist()
            result = ground_segment(grounder, episode, rows, segment, args, placement_window(results[0]["point_pixel"]), sorted(set(frames)))
        elif segment["phase_prompt"] == "grasp_handle_close" and not results[0]["needs_review"]:
            result = ground_segment(grounder, episode, rows, segment, args, handle_window(results[0]["point_pixel"]))
        else:
            result = ground_segment(grounder, episode, rows, segment, args)
        results.append(result)
    return results


def point_once(grounder, episode: Path, rows: list[dict], index: int, prompt: str, args, multi=False):
    """Normalized point(s) for one frame, or None on a model miss (device errors propagate)."""
    rgb = read_rgb(episode / rows[index]["images"]["left"], args.roi)
    scale = np.array([rgb.shape[1], rgb.shape[0]], dtype=np.float32)
    try:
        if multi:
            return [(np.asarray(p, dtype=np.float32) / scale).tolist() for p in grounder._point_multi(rgb, prompt)]
        return (np.asarray(grounder._point(rgb, prompt), dtype=np.float32) / scale).tolist()
    except Exception as exc:
        if isinstance(exc, MemoryError) or "cuda" in str(exc).lower() or "out of memory" in str(exc).lower():
            raise
        return None


def locate_chicken_handles(grounder, episode: Path, rows: list[dict], segment: dict, args) -> dict | None:
    """Both closed handles on early frames of the first pre-open: {'left': [x, y], 'right': [x, y]}."""
    for index in sample_indices(segment["start_frame"], segment["end_frame"], 3, args.skip_first_frames):
        points = point_once(grounder, episode, rows, index, CHICKEN_HANDLES, args, multi=True) or []
        if len(points) != 2:
            points = [point_once(grounder, episode, rows, index, CHICKEN_SIDE_HANDLE.format(side=s), args) for s in ("left", "right")]
        if all(points) and len(points) == 2:
            left, right = sorted(points)
            if right[0] - left[0] >= 0.05:
                return {"left": left, "right": right, "frame_index": index}
    return None


def ground_chicken_episode(grounder, episode: Path, rows: list[dict], segments: list[dict], args, log=lambda *a: None) -> list[dict]:
    results = []
    handles = locate_chicken_handles(grounder, episode, rows, segments[0], args)
    for number, segment in enumerate(segments):
        log(number, segment)
        prompt = segment["phase_prompt"]
        if prompt == "pre-open":
            if handles is None:
                result = {**ground_segment(grounder, episode, rows, segment, args), "needs_review": True,
                          "chicken_note": "could not locate both drawer handles"}
            else:
                # The gripper holds the target handle at the end of pre-open; pick the nearer handle by x.
                end = segment["end_frame"]
                tail = range(max(segment["start_frame"], end - 3), end)
                grips = [p for p in (point_once(grounder, episode, rows, i, GRIPPER, args) for i in tail) if p]
                side = None
                if grips:
                    gx = float(np.median([p[0] for p in grips]))
                    side = min(("left", "right"), key=lambda s: abs(handles[s][0] - gx))
                if side is None:
                    result = {**copy_segment(segment, {"phase_index": -1, "point_pixel": [-1.0, -1.0], "needs_review": True}, args),
                              "chicken_note": "gripper not found; drawer side unknown"}
                else:
                    anchor = handles[side]
                    seg = {**segment, "molmo_prompt": CHICKEN_SIDE_HANDLE.format(side=side), "fallback_prompt": CHICKEN_HANDLES}
                    result = ground_segment(grounder, episode, rows, seg, args, window(anchor, (-0.05, 0.05), (-0.06, 0.08)))
                    if result["needs_review"]:  # occluded by the approaching arm: use the clean closed-handle point
                        result.update(point_pixel=anchor, needs_review=False, chicken_note="handle point from episode anchor frame")
                    result.update(drawer_side=side, gripper_points=grips)
            result["drawer_handles"] = handles
        elif prompt == "open-and-home":
            result = copy_segment(segment, results[-1], args)
        elif prompt in ("grasp", "place") and not results[-1]["needs_review"]:
            # The gripper covers the target; accept hits near the previous phase's point, else copy it.
            result = ground_segment(grounder, episode, rows, segment, args, window(results[-1]["point_pixel"], (-0.06, 0.06), (-0.06, 0.06)))
            if result["needs_review"]:
                result = {**copy_segment(segment, results[-1], args), "preview_frame": result["preview_frame"]}
        else:
            result = ground_segment(grounder, episode, rows, segment, args)
        results.append(result)
    return results


def ground_letter_episode(grounder, episode: Path, rows: list[dict], segments: list[dict], args, log=lambda *a: None) -> list[dict]:
    word = args.word.upper()
    if len(segments) != 4 * len(word):
        raise ValueError(f"{episode}: {len(segments) // 4} cycles but --word {word!r} has {len(word)} letters")
    results = []
    # Locate all cubes on the first frames (all on the table, arm away). A pre-grasp hit must land on
    # one of them (rejects letters written on the whiteboard) and not on a cube an earlier cycle took.
    # Naming each letter on the first frame is unreliable (S/A confusions), so identity is left to the
    # per-phase prompt and only cube positions are fixed here.
    early = sample_indices(segments[0]["start_frame"], segments[0]["end_frame"], 3, args.skip_first_frames)
    cubes = []
    for index in early:
        cubes = point_once(grounder, episode, rows, index, ALL_CUBES, args, multi=True) or []
        if len(cubes) >= len(word):
            break
    taken = []  # start positions of cubes picked up in earlier cycles
    placed = []  # targets of earlier cycles; a later placement must not land on a filled cell
    for cycle, letter in enumerate(word):
        cell = cycle + 1
        pre_grasp, grasp, pre_place, place = segments[4 * cycle:4 * cycle + 4]
        primary, fallback = letter_prompts(letter)
        tag = {"letter": letter, "grid_cell": cell}
        # pre-grasp: the letter cube on the table, early in the approach.
        log(4 * cycle, pre_grasp)
        def on_free_cube(point):
            near = lambda q, r: np.hypot(*(point - np.asarray(q))) <= r
            return (not cubes or any(near(q, 0.035) for q in cubes)) and not any(near(q, 0.02) for q in taken)
        first = ground_segment(grounder, episode, rows, {**pre_grasp, "molmo_prompt": primary, "fallback_prompt": fallback}, args, on_free_cube)
        first.update(tag, table_cubes=cubes)
        if not first["needs_review"]:
            taken.append(first["point_pixel"])
        # grasp: the gripper covers the cube; accept hits near the pre-grasp point, else copy it.
        log(4 * cycle + 1, grasp)
        if first["needs_review"]:
            second = {**ground_segment(grounder, episode, rows, {**grasp, "molmo_prompt": primary, "fallback_prompt": fallback}, args), **tag}
        else:
            second = ground_segment(grounder, episode, rows, {**grasp, "molmo_prompt": primary, "fallback_prompt": fallback}, args,
                                    window(first["point_pixel"], (-0.06, 0.06), (-0.06, 0.06)))
            if second["needs_review"]:
                second = {**copy_segment(grasp, first, args), "preview_frame": second["preview_frame"]}
            second.update(tag)
        # pre-place/place: where the cube ends up. Find the placed cube after release (end of place,
        # start of the next cycle) near a rough numbered-cell estimate; small digits alone are unreliable.
        log(4 * cycle + 2, pre_place)
        anchor_frames = sample_indices(pre_place["start_frame"], pre_place["end_frame"], 3, args.skip_first_frames)
        anchors = [p for i in anchor_frames for p in [point_once(grounder, episode, rows, i, cell_prompts(cell)[0], args)] if p]
        anchor = np.median(np.asarray(anchors), axis=0).tolist() if anchors else None
        tail = max(1, (place["end_frame"] - place["start_frame"]) // 4)
        frames = np.linspace(place["end_frame"] - tail, place["end_frame"] - 1, 4, dtype=int).tolist()
        if cycle + 1 < len(word):
            nxt = segments[4 * cycle + 4]
            frames += np.linspace(nxt["start_frame"], nxt["start_frame"] + max(1, (nxt["end_frame"] - nxt["start_frame"]) // 4) - 1, 6, dtype=int).tolist()
        frames = sorted(set(frames))
        near_cell = (lambda point: True) if anchor is None else window(anchor, (-0.08, 0.08), (-0.08, 0.08))
        accept = lambda point: near_cell(point) and all(np.hypot(*(point - np.asarray(q))) > 0.02 for q in placed)
        third = ground_segment(grounder, episode, rows, {**pre_place, "molmo_prompt": primary, "fallback_prompt": fallback}, args, accept, frames)
        third.update(tag, grid_cell_anchor=anchor, target_method="placed cube after release")
        if third["needs_review"] and anchor is not None:
            # Placed cube not found: fall back to the numbered-cell estimate, but keep it flagged.
            third.update(point_pixel=anchor, target_method="numbered grid cell estimate (placed cube not found)")
        log(4 * cycle + 3, place)
        if not third["needs_review"]:
            placed.append(third["point_pixel"])
        fourth = {**copy_segment(place, third, args), **tag}
        results += [first, second, third, fourth]
    return results


# Block-stacking recordings: plain "pre-grasp"/"grasp"/"pre-place"/"place" cycles like the crossword,
# detected from meta.json's instruction ("... in the order of red, yellow, blue from bottom to top").
# Cycle k picks order[k + 1] and places it onto order[k] (the current top of the stack).
STACK_INSTRUCTION = re.compile(r"order of (\w+), (\w+), (\w+) from bottom to top")


def stack_order(episode: Path) -> list[str] | None:
    try:
        instruction = json.loads((episode / "meta.json").read_text()).get("instruction") or ""
    except (OSError, ValueError):
        return None
    match = STACK_INSTRUCTION.search(instruction) if "stack" in instruction.lower() else None
    return [color.lower() for color in match.groups()] if match else None


def block_prompts(color: str) -> tuple[str, str]:
    return f"Point to the {color} block", f"Point to the {color} cube"


def ground_stack_episode(grounder, episode: Path, rows: list[dict], segments: list[dict], args, log=lambda *a: None) -> list[dict]:
    order = stack_order(episode)
    if not order or len(segments) != 4 * (len(order) - 1):
        raise ValueError(f"{episode}: {len(segments) // 4} cycles but stacking order {order} needs {len(order or []) - 1}")
    near = lambda q, r: (lambda point: np.hypot(*(point - np.asarray(q))) <= r)
    results = []
    stack = None  # point of the first base block; later placements must land on this stack
    for cycle in range(len(order) - 1):
        moved, base = order[cycle + 1], order[cycle]
        pre_grasp, grasp, pre_place, place = segments[4 * cycle:4 * cycle + 4]
        grab = {"molmo_prompt": block_prompts(moved)[0], "fallback_prompt": block_prompts(moved)[1]}
        onto = {"molmo_prompt": block_prompts(base)[0], "fallback_prompt": block_prompts(base)[1]}
        tag = {"stack_cycle": cycle, "moved_block": moved, "target_block": base}
        # pre-grasp: the block to move, still on the table; it must not be part of the stack.
        log(4 * cycle, pre_grasp)
        off_stack = None if stack is None else (lambda point: not near(stack, 0.04)(point))
        first = {**ground_segment(grounder, episode, rows, {**pre_grasp, **grab}, args, off_stack), **tag}
        # grasp: the gripper covers the block; accept hits near the pre-grasp point, else copy it.
        log(4 * cycle + 1, grasp)
        if first["needs_review"]:
            second = ground_segment(grounder, episode, rows, {**grasp, **grab}, args)
        else:
            second = ground_segment(grounder, episode, rows, {**grasp, **grab}, args, window(first["point_pixel"], (-0.06, 0.06), (-0.06, 0.06)))
            if second["needs_review"]:
                second = {**copy_segment(grasp, first, args), "preview_frame": second["preview_frame"]}
        second.update(tag)
        # pre-place: the block it goes onto (top of the stack); the carried block is in the gripper.
        log(4 * cycle + 2, pre_place)
        accept = lambda point: not near(first["point_pixel"], 0.03)(point) if not first["needs_review"] else True
        if stack is not None:  # a stacked block sits above the base point in the image
            on_stack = window(stack, (-0.06, 0.06), (-0.12, 0.04))
            accept = (lambda a, b: lambda point: a(point) and b(point))(accept, on_stack)
        third = {**ground_segment(grounder, episode, rows, {**pre_place, **onto}, args, accept), **tag}
        if stack is None and not third["needs_review"]:
            stack = third["point_pixel"]
        # place: the gripper hides the target; accept hits near the pre-place point, else copy it.
        log(4 * cycle + 3, place)
        if third["needs_review"]:
            fourth = ground_segment(grounder, episode, rows, {**place, **onto}, args)
        else:
            fourth = ground_segment(grounder, episode, rows, {**place, **onto}, args, window(third["point_pixel"], (-0.06, 0.06), (-0.06, 0.06)))
            if fourth["needs_review"]:
                fourth = {**copy_segment(place, third, args), "preview_frame": fourth["preview_frame"]}
        fourth.update(tag)
        results += [first, second, third, fourth]
    return results


def ground_segment(grounder, episode: Path, rows: list[dict], segment: dict, args, accept=None, samples=None) -> dict:
    if samples is None:
        # A window rejects most distractor hits, so draw more frames to keep enough candidates.
        count = args.samples * 2 if accept is not None else args.samples
        samples = sample_indices(segment["start_frame"], segment["end_frame"], count, args.skip_first_frames)
    points = []
    attempts = []
    for index in samples:
        rgb = read_rgb(episode / rows[index]["images"]["left"], args.roi)
        height, width = rgb.shape[:2]
        attempt = {"frame_index": index, "seq": rows[index]["seq"], "errors": []}
        for prompt in (segment["molmo_prompt"], segment["fallback_prompt"]):
            try:
                pixel = np.asarray(grounder._point(rgb, prompt), dtype=np.float32)
                if pixel.shape != (2,) or not np.isfinite(pixel).all():
                    raise ValueError(f"Invalid point: {pixel}")
                if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
                    raise ValueError(f"Point outside image: {pixel}")
                normalized = pixel / np.array([width, height], dtype=np.float32)
                if accept is not None and not accept(normalized):
                    raise ValueError(f"Point outside expected region: {normalized.tolist()}")
                points.append(normalized)
                attempt.update({"prompt_used": prompt, "point_pixel": normalized.tolist()})
                break
            except Exception as exc:
                message = str(exc).lower()
                # Stop on device failures; never turn a broken GPU into failed labels.
                if isinstance(exc, MemoryError) or any(word in message for word in (
                    "out of memory", "cuda error", "device-side assert", "meta tensor",
                )):
                    raise
                attempt["errors"].append({"prompt": prompt, "error": str(exc)})
        attempts.append(attempt)
    kept = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    required = min(args.min_valid_samples, len(samples))
    if accept is not None:  # distractors are already excluded; two agreeing hits suffice
        required = min(required, 2)
    # With exactly `required` points the filter always drops the farthest one,
    # so short phases (3 samples) could never pass.
    if len(kept) > max(2, required):
        distances = np.linalg.norm(kept - np.median(kept, axis=0), axis=1)
        threshold = float(np.median(distances) + 2 * np.std(distances))
        kept = kept[distances <= threshold + 1e-8]
    success = len(kept) >= required
    point = kept.mean(axis=0).tolist() if success else [-1.0, -1.0]
    return {
        **segment, "sample_indices": samples, "samples": attempts,
        "valid_predictions": len(points), "inliers": len(kept),
        "required_inliers": required, "point_pixel": point, "needs_review": not success,
        # Draw the preview on the frame whose prediction is closest to the final point,
        # i.e. where the target is most clearly visible at the marked location.
        "preview_frame": min((a for a in attempts if "point_pixel" in a),
                             key=lambda a: np.hypot(*(np.asarray(a["point_pixel"]) - np.asarray(point))) if success else 0,
                             default={"frame_index": samples[0]})["frame_index"],
    }


def draw_point(image: np.ndarray, pixel: tuple[int, int]) -> None:
    """Open-centre cross (white outline, red core): the target under the point stays visible."""
    x, y = pixel
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        start, end = (x + 7 * dx, y + 7 * dy), (x + 20 * dx, y + 20 * dy)
        cv2.line(image, start, end, (255, 255, 255), 6)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(image, (x + 8 * dx, y + 8 * dy), (x + 19 * dx, y + 19 * dy), (0, 0, 255), 3)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def segment_label(segment: dict) -> str:
    """Top-left preview label; keep in sync with annotate_sorting_points_correction.segment_label."""
    prompt = segment["phase_prompt"]
    side = segment.get("drawer_side")
    label = prompt + (f" ({side} drawer)" if side and side not in prompt.split() else "")
    if segment.get("letter"):
        if segment["letter"] not in prompt.split():
            label += f" ({segment['letter']})" if "slot" in prompt.split() else f" {segment['letter']}"
        if "slot" not in prompt.split():
            label += f" -> cell {segment['grid_cell']}"
    if segment.get("moved_block"):
        label += f" {segment['moved_block']}" if segment["phase_index"] < 2 else f" {segment['moved_block']} onto {segment['target_block']}"
    return label


def write_episode(path: Path, rows: list[dict], segments: list[dict], args) -> None:
    preview_dir = path.parent / "molmo_previews"
    preview_dir.mkdir(exist_ok=True)
    for number, segment in enumerate(segments):
        for row in rows[segment["start_frame"]:segment["end_frame"]]:
            row["point_pixel"] = segment["point_pixel"]
        # Preview uses the exact input crop and the final phase-level point.
        # Pre-place targets are measured after release; still show a frame from inside the segment.
        frame = segment.get("preview_frame", segment["sample_indices"][0])
        if not segment["start_frame"] <= frame < segment["end_frame"]:
            frame = segment["end_frame"] - 1
        row = rows[frame]
        image = read_rgb(path.parent / row["images"]["left"], args.roi)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        x, y = segment["point_pixel"]
        if x >= 0:
            pixel = (min(width - 1, round(x * width)), min(height - 1, round(y * height)))
            draw_point(image, pixel)
        label = segment_label(segment)
        text = label if x >= 0 else "FAILED: " + label
        cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if not cv2.imwrite(str(preview_dir / f"segment_{number:02d}.jpg"), image):
            raise RuntimeError("Failed to write point preview")
    report = {
        "source_steps": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint": args.checkpoint, "dtype": args.dtype, "load_in_4bit": args.load_in_4bit,
        "max_image_size": args.max_image_size, "roi_xy_size": args.roi,
        "point_coordinate_system": "normalized xy on cropped left RGB image" if args.roi else "normalized xy on full left RGB image",
        "sampling": "first 25% of each consecutive phase segment; drawer place phase spans the whole phase plus the next phase start",
        "needs_review": any(s["needs_review"] for s in segments), "segments": segments,
    }
    atomic_text(path.parent / "steps_molmo.jsonl", "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    atomic_text(path.parent / "molmo_point_annotations.json", json.dumps(report, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT / "output_1017_phase_sorting")
    parser.add_argument("--atomic-policy-root", type=Path, default=PROJECT / "atomic_policy-main")
    geometry = parser.add_mutually_exclusive_group(required=True)
    geometry.add_argument("--roi", nargs=3, type=int, metavar=("X", "Y", "SIZE"), help="Fixed square crop in raw image pixels")
    geometry.add_argument("--full-frame", action="store_true", help="Point in full camera coordinates; useful for an initial test")
    parser.add_argument("--checkpoint", default="allenai/MolmoPoint-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-image-size", type=int, default=448)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--skip-first-frames", type=int, default=2)
    parser.add_argument("--min-valid-samples", type=int, default=3)
    parser.add_argument("--max-episodes", type=int, default=0, help="0 = all episodes")
    parser.add_argument("--dry-run", action="store_true", help="Validate labels, image samples and ROI without loading Molmo or saving annotations")
    parser.add_argument("--overwrite", action="store_true", help="Replace previously generated annotation files; originals are never modified")
    parser.add_argument("--word", help="Letter-cube mode: word spelled in order, one letter per pre-grasp/grasp/pre-place/place cycle")
    parser.add_argument("--skip-existing", action="store_true", help="Leave episodes that already have steps_molmo.jsonl untouched")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if min(args.samples, args.min_valid_samples, args.max_image_size) <= 0 or args.skip_first_frames < 0 or args.max_episodes < 0:
        parser.error("samples, min-valid-samples, max-image-size must be positive; skip/max-episodes cannot be negative")
    if args.min_valid_samples > args.samples:
        parser.error("min-valid-samples cannot exceed samples")
    root = args.dataset_root.resolve()
    paths = [root / "steps.jsonl"] if (root / "steps.jsonl").is_file() else sorted(root.rglob("episode_*/steps.jsonl"))
    if args.skip_existing:
        paths = [p for p in paths if not (p.parent / "steps_molmo.jsonl").exists()]
    if args.max_episodes:
        paths = paths[:args.max_episodes]
    if not paths:
        parser.error(f"No episode steps.jsonl files under {root}")
    total_frames = 0
    total_samples = 0
    # Validate the entire selection before loading the model or writing files.
    for path in paths:
        rows, segments = read_episode(path)
        if segments[0]["phase_prompt"] == LETTER_PHASES[0] and stack_order(path.parent):
            order = stack_order(path.parent)
            if len(segments) != 4 * (len(order) - 1):
                parser.error(f"{path}: {len(segments) // 4} cycles but stacking order {order} needs {len(order) - 1}")
        elif segments[0]["phase_prompt"] == LETTER_PHASES[0]:
            if not args.word:
                parser.error(f"{path}: letter-cube recording; pass --word (e.g. --word MARS)")
            if len(segments) != 4 * len(args.word):
                parser.error(f"{path}: {len(segments) // 4} cycles but --word {args.word!r} has {len(args.word)} letters")
        if not args.dry_run and not args.overwrite:
            for output in ("steps_molmo.jsonl", "molmo_point_annotations.json"):
                if (path.parent / output).exists():
                    parser.error(f"{path.parent / output} exists; use --overwrite to regenerate")
        for segment in segments:
            for index in sample_indices(segment["start_frame"], segment["end_frame"], args.samples, args.skip_first_frames):
                read_rgb(path.parent / rows[index]["images"]["left"], args.roi)
                total_samples += 1
        total_frames += len(rows)
    LOG.info("Validated %d episodes, %d frames, %d image samples", len(paths), total_frames, total_samples)
    if args.dry_run:
        return
    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("A working CUDA GPU is required by the repository's MolmoPoint grounder")
    sys.path.insert(0, str(args.atomic_policy_root.resolve()))
    from atomic_policy.vlm.grounder_molmo import MolmoPointGrounder

    grounder = MolmoPointGrounder(
        checkpoint=args.checkpoint, device=args.device, dtype=getattr(torch, args.dtype),
        load_in_4bit=args.load_in_4bit, max_image_size=args.max_image_size,
    )
    failed_segments = 0
    for episode_index, path in enumerate(paths):
        rows, segments = read_episode(path)
        log = lambda number, segment: LOG.info("Episode %d/%d %s segment %d: %s", episode_index + 1, len(paths), path.parent.name, number, segment["phase_prompt"])
        if segments[0]["phase_prompt"] == LETTER_PHASES[0] and stack_order(path.parent):
            results = ground_stack_episode(grounder, path.parent, rows, segments, args, log)
        elif segments[0]["phase_prompt"] == LETTER_PHASES[0]:
            results = ground_letter_episode(grounder, path.parent, rows, segments, args, log)
        elif segments[0]["phase_prompt"] == "pre-open":
            results = ground_chicken_episode(grounder, path.parent, rows, segments, args, log)
        elif segments[0]["phase_prompt"] in DRAWER_PHASES:
            results = ground_episode(grounder, path.parent, rows, segments, args, log)
        else:
            results = []
            for number, segment in enumerate(segments):
                log(number, segment)
                results.append(ground_segment(grounder, path.parent, rows, segment, args))
        for result in results:
            failed_segments += result["needs_review"]
            LOG.info("  phase %d point=%s, inliers=%d, needs_review=%s", result["phase_index"], result["point_pixel"], result["inliers"], result["needs_review"])
        write_episode(path, rows, results, args)
    LOG.info("Finished: %d episodes, %d failed segments. Inspect molmo_previews/ in each episode.", len(paths), failed_segments)
    if failed_segments:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
