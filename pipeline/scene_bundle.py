"""Turn a reconstruction into a scene bundle for the studio viewer (apps/studio, PlayCanvas).

    python pipeline/scene_bundle.py <sparse_dir> <web_splat.sog> <out_dir> [--title T]

Writes out_dir/manifest.json, visual/<splat>, navigation/grid.json, semantic/pois.json and
quality/report.json. Two things the studio viewer needs that COLMAP does not give:

Scale. COLMAP's units are arbitrary. Without a marker in the room we guess: the phone was held at
ASSUMED_CAMERA_HEIGHT_M above the floor, and the floor is the lowest dense layer of SfM points.
The manifest says scaleStatus "estimated"; expect +-15% on a real capture. When the model is already in
metres (e.g. a dataset solved with a calibrated rig), pass --metres-per-unit and scaleStatus is "known".

Walls. The navigation grid marks a cell blocked when SfM points sit between knee and head height
in it (desks, walls), widened by the player's clearance. Cells the camera walked through are always
free, since the person filming walked there. Cells far from the filmed path are "unknown": the
splat is unreliable there, so the viewer does not walk into them.

Frame. The studio viewer applies no reconstruction->world transform (its manifest schema and
apps/studio/scripts/validate_scene.py accept only an identity worldFromReconstruction). So the splat must
already be in the bundle's world frame (levelled, metres, floor at y=0): world_alignment() gives the
transform and bake_ply() applies it to the float PLY (positions, scales, rotations and the SH colour
coefficients) before compression; build() writes identity. run_pipeline does this; by hand:
    python pipeline/scene_bundle.py --bake <sparse_dir> splat_clean.ply splat_world.ply
To re-pick only the spawn of an existing bundle (no model needed): scene_bundle.py --respawn - - <bundle_dir>

Views. For each walkable cell the grid also lists which way the phone faced from nearby (a bitmask of
VIEW_SECTORS compass sectors). With ?guide=1 the viewer turns the visitor gently back when they rest looking
in a direction nobody filmed from there, since that is where the splat streaks (off by default).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from room_coverage import level_rotation, load_model  # noqa: E402

ASSUMED_CAMERA_HEIGHT_M = 1.45  # phone held at chest height while walking
CELL_M = 0.10                   # navigation grid resolution
CLEARANCE_M = 0.25              # player radius: blocked cells are widened by this
OBSTACLE_BAND_M = (0.25, 1.9)   # points in this height band block walking (floor and ceiling do not)
MIN_POINTS = 2                  # points per cell to count as an obstacle (single points are noise)
# Beyond this distance from the filmed path cells are "unknown". A splat is only sharp near the views
# it was trained from: at 1.5 m visitors reached spots that looked like smeared paint, and at 0.75 m the
# low views between desks still streaked, so keep it tight.
FILMED_RADIUS_M = 0.5
WALKED_RADIUS_M = 0.2           # cells this close to a camera position are always free
VIEW_SECTORS = 16               # compass sectors (22.5 degrees) for the per-cell filmed directions
VIEW_RADIUS_M = 1.0             # a camera within this distance counts as a view from the cell
MIN_VIEW_FRAMES = 3             # frames a sector needs to count as filmed (one or two glances still streak)
# Pose outliers. A phone walked at ~1 m/s and sampled at 2-3 fps moves 0.1-0.4 m per frame; a step of many times
# the median is SfM placing a frame in the wrong spot (often the first few frames, before the map has context),
# drawn on the nav grid as a long straight line through furniture. Such a step is never "walked"; and a stretch of
# the walk cut off by jumps on both sides that holds only a few frames is treated as mis-posed and not used to
# seed walkable cells, the view masks or the spawn.
JUMP_FACTOR = 8.0               # a step this many times the median step is a jump...
JUMP_MIN_M = 0.5                # ...if it is also at least this long
MIN_RUN_FRAMES = 10             # runs between jumps shorter than max(this, MIN_RUN_FRACTION of frames) are outliers
MIN_RUN_FRACTION = 0.05
# Dense evidence from the trained splat (world frame). The sparse SfM model has a few tens of thousands of points:
# a stray pair of them blocks a 0.6 m disc of floor, and open floor more than FILMED_RADIUS_M from the path is
# "unknown" even when it was clearly in view. With the splat as well:
#  - a sparse obstacle with no solid splat anywhere near it (SPLAT_SUPPORT_M) is SfM noise and is dropped;
#  - a cell holding SPLAT_MIN_SOLID solid (opacity >= SPLAT_SOLID_ALPHA), compact (largest axis <= SPLAT_MAX_SCALE_M)
#    splats in the obstacle band is an obstacle even with no SfM point in it;
#  - floor that was seen (SfM points or solid splats within FLOOR_BAND_M of y=0), within REACH_RADIUS_M of the
#    walked path, not blocked and connected to the filmed walkable area, is walkable, not unknown.
SPLAT_SOLID_ALPHA = 0.3
SPLAT_MAX_SCALE_M = 0.3
SPLAT_MIN_SOLID = 2
SPLAT_SUPPORT_M = 0.2
FLOOR_BAND_M = 0.15
FLOOR_FILL_M = 0.2              # gaps this wide between floor evidence are closed
REACH_RADIUS_M = 1.5
SPAWN_RAYS = 32                 # directions tried for the spawn's facing
SPAWN_RAY_MAX_M = 8.0


def estimate_scale(P: np.ndarray, C: np.ndarray) -> tuple[float, float, float]:
    """Return (metres per model unit, floor height, ceiling height) in the levelled frame."""
    cam_y = float(np.median(C[:, 1]))
    below, above = P[P[:, 1] < cam_y, 1], P[P[:, 1] > cam_y, 1]
    if len(below) < 20:
        raise ValueError("too few points below the camera path to find the floor")
    floor = float(np.percentile(below, 3))
    ceiling = float(np.percentile(above, 97)) if len(above) >= 20 else cam_y + (cam_y - floor)
    return ASSUMED_CAMERA_HEIGHT_M / (cam_y - floor), floor, ceiling


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def view_masks(Cw: np.ndarray, Fw: np.ndarray, lo: np.ndarray, w: int, h: int) -> np.ndarray:
    """Per cell, a bitmask of the compass sectors at least MIN_VIEW_FRAMES cameras within VIEW_RADIUS_M
    faced. Sector k covers headings atan2(x, z) in [k, k+1) * 360 / VIEW_SECTORS degrees."""
    heading = np.arctan2(Fw[:, 0], Fw[:, 2]) % (2 * math.pi)
    sector = np.minimum((heading / (2 * math.pi) * VIEW_SECTORS).astype(int), VIEW_SECTORS - 1)
    xs = lo[0] + (np.arange(w) + 0.5) * CELL_M
    zs = lo[2] + (np.arange(h) + 0.5) * CELL_M
    counts = np.zeros((VIEW_SECTORS, h, w), np.int32)
    for c, k in zip(Cw, sector):
        d2 = (xs[None, :] - c[0]) ** 2 + (zs[:, None] - c[2]) ** 2
        counts[k][d2 <= VIEW_RADIUS_M ** 2] += 1
    return ((counts >= MIN_VIEW_FRAMES) * (1 << np.arange(VIEW_SECTORS))[:, None, None]).sum(0)


def pose_outliers(Cw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cameras in capture order -> (outlier per camera, jump per consecutive step). See JUMP_FACTOR."""
    n = len(Cw)
    if n < 3:
        return np.zeros(n, bool), np.zeros(max(n - 1, 0), bool)
    step = np.linalg.norm(np.diff(Cw[:, [0, 2]], axis=0), axis=1)
    jump = step > max(JUMP_FACTOR * float(np.median(step)), JUMP_MIN_M)
    run = np.concatenate([[0], np.cumsum(jump)])
    size = np.bincount(run)
    keep = size >= max(MIN_RUN_FRAMES, MIN_RUN_FRACTION * n)
    if not keep.any():
        keep = size == size.max()
    return ~keep[run], jump


def splat_samples(path: Path | None) -> dict | None:
    """Positions, opacity (0-1) and largest axis (metres) of a world-frame splat: float 3DGS PLY or PlayCanvas
    compressed PLY. None for other formats (SOG) or a missing file, so callers fall back to SfM points only."""
    if path is None or not Path(path).is_file() or Path(path).suffix != ".ply":
        return None
    raw = Path(path).read_bytes()
    head = raw[:raw.find(b"end_header")]
    if b"element chunk" in head:
        from compressed_ply import decode
        d = decode(raw)
        return {"xyz": d["xyz"], "alpha": d["alpha"], "size": np.exp(d["log_scale"]).max(1)}
    from clean_splat import read_ply
    _, props, d = read_ply(Path(path))
    col = {k: i for i, k in enumerate(props)}
    xyz = d[:, [col["x"], col["y"], col["z"]]].astype(np.float64)
    alpha = 1 / (1 + np.exp(-d[:, col["opacity"]].astype(np.float64)))
    size = np.exp(d[:, [col[f"scale_{i}"] for i in range(3)]].astype(np.float64)).max(1)
    return {"xyz": xyz, "alpha": alpha, "size": size}


def walkable_mask(nav: dict) -> np.ndarray:
    w, h = nav["width"], nav["height"]
    occ = np.zeros(w * h, bool)
    occ[nav["blocked"]] = True
    occ[nav["unknown"]] = True
    return ~occ.reshape(h, w)


def _ray_lengths(walkable: np.ndarray, x: int, z: int, cell: float) -> np.ndarray:
    """Free distance (m) from cell (x, z) through walkable cells along SPAWN_RAYS headings; heading k is
    direction (sin a, cos a) in (x, z) with a = 2 pi k / SPAWN_RAYS, i.e. compass heading atan2(dx, dz)."""
    h, w = walkable.shape
    a = 2 * math.pi * np.arange(SPAWN_RAYS) / SPAWN_RAYS
    t = np.arange(1, int(SPAWN_RAY_MAX_M / cell * 2) + 1) * 0.5  # half-cell steps
    xs = np.floor(x + 0.5 + np.sin(a)[:, None] * t).astype(int)
    zs = np.floor(z + 0.5 + np.cos(a)[:, None] * t).astype(int)
    inside = (xs >= 0) & (xs < w) & (zs >= 0) & (zs < h)
    free = inside & walkable[np.clip(zs, 0, h - 1), np.clip(xs, 0, w - 1)]
    first_block = np.where(free.all(1), len(t), np.argmin(free, axis=1))
    return np.where(first_block > 0, t[np.maximum(first_block - 1, 0)], 0.0) * cell


def choose_spawn(nav: dict, cameras_xz: np.ndarray | None = None) -> dict:
    """Spawn cell and yaw: in the LARGEST walkable component (4-neighbour, as grid_stats and the route planner),
    on a cell a (non-outlier) camera stood in when there is one there (else one filmed from, else any), preferring
    room around it, facing the longest free line of sight (a filmed direction when one is nearly as long).
    Returns {"x", "z", "yaw"}: yaw in radians, the manifest's convention (forward = (-sin yaw, -cos yaw))."""
    from scipy import ndimage

    walkable = walkable_mask(nav)
    h, w, c = nav["height"], nav["width"], nav["cellSize"]
    labels, n = ndimage.label(walkable)
    if n == 0:
        raise ValueError("navigation grid has no walkable cell to spawn on")
    biggest = 1 + int(np.argmax(np.bincount(labels.ravel())[1:]))
    region = labels == biggest
    cand = np.zeros_like(region)
    if cameras_xz is not None and len(cameras_xz):
        ij = np.floor((np.asarray(cameras_xz) - [nav["origin"]["x"], nav["origin"]["z"]]) / c).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
        cand[ij[ok, 1], ij[ok, 0]] = True
        cand &= region
    views = np.asarray(nav["views"]).reshape(h, w) if "views" in nav else None
    if not cand.any() and views is not None:
        cand = region & (views > 0)
    if not cand.any():
        cand = region
    clear = ndimage.distance_transform_edt(np.pad(walkable, 1))[1:-1, 1:-1] * c  # metres to the nearest wall
    zz, xx = np.nonzero(cand)
    order = np.argsort(-clear[zz, xx], kind="stable")[:64]  # rays only for the roomiest candidates
    best = None
    for k in order:
        x, z = int(xx[k]), int(zz[k])
        rays = _ray_lengths(walkable, x, z, c)
        score = min(clear[z, x], 1.0) + 0.2 * rays.max()
        if best is None or score > best[0]:
            best = (score, x, z, rays)
    _, x, z, rays = best
    d = int(np.argmax(rays))
    if views is not None and views[z, x]:
        sectors = nav.get("viewSectors", VIEW_SECTORS)
        filmed = np.array([(views[z, x] >> int(k * sectors / SPAWN_RAYS)) & 1 for k in range(SPAWN_RAYS)], bool)
        if filmed.any() and rays[filmed].max() >= 0.6 * rays.max():
            d = int(np.flatnonzero(filmed)[np.argmax(rays[filmed])])
    a = 2 * math.pi * d / SPAWN_RAYS  # forward (sin a, cos a) = (-sin yaw, -cos yaw)
    return {"x": x, "z": z, "yaw": math.atan2(-math.sin(a), -math.cos(a))}


def navigation_grid(W: np.ndarray, Cw: np.ndarray, lo: np.ndarray, hi: np.ndarray, start: np.ndarray | None = None,
                    Fw: np.ndarray | None = None, splat: dict | None = None,
                    filmed_radius: float = FILMED_RADIUS_M, reach_radius: float = REACH_RADIUS_M) -> dict:
    """W, Cw: points and camera centres (capture order) in world metres (floor at y=0); Fw: camera forward
    directions; splat: splat_samples() of the world-frame splat (optional dense evidence, see SPLAT_*).
    Outlier poses (pose_outliers) seed nothing. With start None the spawn is choose_spawn's."""
    from scipy import ndimage

    outlier, jump = pose_outliers(Cw)

    w = int(math.ceil((hi[0] - lo[0]) / CELL_M))
    h = int(math.ceil((hi[2] - lo[2]) / CELL_M))

    def cells(xz):
        ij = np.floor((xz - [lo[0], lo[2]]) / CELL_M).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
        return ij[ok]

    def count(xyz):
        c = np.zeros((h, w), int)
        ij = cells(xyz[:, [0, 2]])
        np.add.at(c, (ij[:, 1], ij[:, 0]), 1)
        return c

    in_band = lambda X: (X[:, 1] > OBSTACLE_BAND_M[0]) & (X[:, 1] < OBSTACLE_BAND_M[1])
    obstacle = count(W[in_band(W)]) >= MIN_POINTS
    floor_seen = count(W[np.abs(W[:, 1]) < FLOOR_BAND_M]) > 0
    if splat is not None:
        X, solid = splat["xyz"], splat["alpha"] >= SPLAT_SOLID_ALPHA
        dense = count(X[in_band(X) & solid & (splat["size"] <= SPLAT_MAX_SCALE_M)]) >= SPLAT_MIN_SOLID
        support = ndimage.binary_dilation(count(X[in_band(X) & solid]) > 0, _disk(round(SPLAT_SUPPORT_M / CELL_M)))
        obstacle = (obstacle & support) | dense
        floor_seen |= count(X[(np.abs(X[:, 1]) < FLOOR_BAND_M) & solid]) > 0
    blocked = ndimage.binary_dilation(obstacle, _disk(round(CLEARANCE_M / CELL_M)))

    walked = np.zeros((h, w), bool)
    ij = cells(Cw[~outlier][:, [0, 2]])
    walked[ij[:, 1], ij[:, 0]] = True
    filmed = ndimage.binary_dilation(walked, _disk(round(filmed_radius / CELL_M)))
    reach = ndimage.binary_dilation(walked, _disk(round(reach_radius / CELL_M)))
    walked = ndimage.binary_dilation(walked, _disk(round(WALKED_RADIUS_M / CELL_M)))
    # Consecutive camera positions are walked ground too, even when the gap between them is > 1 cell; but not
    # across a jump or to/from an outlier pose (that line was never walked and may cross furniture).
    for k, (a, b) in enumerate(zip(Cw[:-1], Cw[1:])):
        if jump[k] or outlier[k] or outlier[k + 1]:
            continue
        for t in np.linspace(0, 1, max(2, int(np.linalg.norm((b - a)[[0, 2]]) / (CELL_M / 2)))):
            p = cells((a + (b - a) * t)[None, [0, 2]])
            if len(p):
                walked[p[0, 1], p[0, 0]] = True

    blocked &= ~walked
    # Seen floor near the path counts as filmed, where it joins the filmed walkable area (no new islands).
    floor_seen = ndimage.binary_closing(floor_seen, _disk(round(FLOOR_FILL_M / CELL_M))) & reach & ~blocked
    labels, _ = ndimage.label((filmed | floor_seen) & ~blocked)
    joined = np.isin(labels, np.unique(labels[filmed & ~blocked])) & (labels > 0)
    unknown = ~(filmed | (floor_seen & joined)) & ~blocked
    flat = lambda m: np.flatnonzero(m.ravel()).tolist()  # index = z * width + x
    grid = {"version": 1, "coordinateSystem": "x-z-grid",
            "origin": {"x": round(float(lo[0]), 3), "z": round(float(lo[2]), 3)},
            "cellSize": CELL_M, "width": w, "height": h,
            "blocked": flat(blocked), "unknown": flat(unknown),
            "start": {"x": 0, "z": 0}, "clearanceMetres": CLEARANCE_M}
    if Fw is not None:
        views = view_masks(Cw[~outlier], Fw[~outlier], lo, w, h)
        views[blocked | unknown] = 0
        grid.update(viewSectors=VIEW_SECTORS, views=views.ravel().tolist())
    grid["_floorEvidenceCells"] = int((floor_seen & joined & ~filmed).sum())  # popped by build(), not written
    if start is None:
        sp = choose_spawn(grid, Cw[~outlier][:, [0, 2]])
        grid["start"] = {"x": sp["x"], "z": sp["z"]}
    else:
        sx, sz = cells(start[None, [0, 2]])[0]
        grid["start"] = {"x": int(sx), "z": int(sz)}
    return grid


def grid_stats(nav: dict) -> dict:
    """Walkable area, its connected components (4-neighbour: the studio's planner only steps diagonally when
    both side cells are walkable, so that is its reachability), the spawn's component and unknown area."""
    from scipy import ndimage

    w, h, a = nav["width"], nav["height"], nav["cellSize"] ** 2
    blocked, unknown = np.zeros(w * h, bool), np.zeros(w * h, bool)
    blocked[nav["blocked"]] = True
    unknown[nav["unknown"]] = True
    walkable = (~blocked & ~unknown).reshape(h, w)
    labels, n = ndimage.label(walkable)
    spawn = labels[nav["start"]["z"], nav["start"]["x"]]
    return {"walkable_m2": round(float(walkable.sum() * a), 2), "components": int(n),
            "spawn_component_m2": round(float((labels == spawn).sum() * a), 2) if spawn else 0.0,
            "unknown_m2": round(float(unknown.sum() * a), 2)}


NAVGRID_COLOURS = {"walkable": (226, 240, 222), "blocked": (60, 64, 72), "unknown": (190, 196, 214)}  # RGB
NAVGRID_PX = 6  # pixels per 0.1 m cell


def navgrid_png(nav: dict, cameras_xz: np.ndarray, path: Path) -> None:
    """Top-down picture of the grid (x right, z down, as stored): walkable / blocked / unknown, the camera
    path, the spawn and a 1 m scale bar. Written to quality/navgrid.png for a quick look after a run."""
    import cv2

    w, h, c, px = nav["width"], nav["height"], nav["cellSize"], NAVGRID_PX
    img = np.empty((h * w, 3), np.uint8)
    img[:] = NAVGRID_COLOURS["walkable"]
    img[nav["unknown"]] = NAVGRID_COLOURS["unknown"]
    img[nav["blocked"]] = NAVGRID_COLOURS["blocked"]
    img = np.repeat(np.repeat(img.reshape(h, w, 3), px, 0), px, 1)
    margin = 40
    canvas = np.full((h * px + 2 * margin, w * px + 2 * margin, 3), 255, np.uint8)
    canvas[margin:margin + h * px, margin:margin + w * px] = img
    o = np.array([nav["origin"]["x"], nav["origin"]["z"]])
    to_px = lambda xz: np.round((np.asarray(xz) - o) / c * px + margin).astype(np.int32)
    if len(cameras_xz) > 1:
        cv2.polylines(canvas, [to_px(cameras_xz).reshape(-1, 1, 2)], False, (230, 120, 20), 2, cv2.LINE_AA)
    sp = (np.array([nav["start"]["x"], nav["start"]["z"]]) + 0.5) * px + margin
    sp = tuple(int(v) for v in sp)
    cv2.circle(canvas, sp, 9, (220, 30, 30), -1, cv2.LINE_AA)
    cv2.circle(canvas, sp, 9, (255, 255, 255), 2, cv2.LINE_AA)
    y = canvas.shape[0] - 14
    cv2.line(canvas, (margin, y), (margin + int(round(1 / c * px)), y), (0, 0, 0), 3)
    cv2.putText(canvas, "1 m", (margin + int(round(1 / c * px)) + 8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    legend = [("walkable", NAVGRID_COLOURS["walkable"]), ("blocked", NAVGRID_COLOURS["blocked"]),
              ("unknown", NAVGRID_COLOURS["unknown"]), ("camera path", (230, 120, 20)), ("spawn", (220, 30, 30))]
    x = margin
    for name, col in legend:
        cv2.rectangle(canvas, (x, 12), (x + 14, 26), col, -1)
        cv2.rectangle(canvas, (x, 12), (x + 14, 26), (0, 0, 0), 1)
        cv2.putText(canvas, name, (x + 18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        x += 24 + 8 * len(name) + 10
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def spawn_position(nav: dict, sp: dict) -> np.ndarray:
    """World position of a spawn cell's centre, eye at the height the views were filmed from (sharpest there)."""
    c = nav["cellSize"]
    return np.array([nav["origin"]["x"] + (sp["x"] + 0.5) * c, ASSUMED_CAMERA_HEIGHT_M,
                     nav["origin"]["z"] + (sp["z"] + 0.5) * c])


def respawn(bundle: Path) -> dict:
    """Re-pick the spawn of an existing bundle from its grid alone (no SfM model needed): updates the grid's
    start, the manifest's startPosition/startYawDegrees, the "start" POI, the quality numbers and navgrid.png
    (redrawn without the camera path, which the bundle does not store)."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    nav_path = bundle / manifest["navigationUri"]
    nav = json.loads(nav_path.read_text())
    before = grid_stats(nav)
    sp = choose_spawn(nav)
    nav["start"] = {"x": sp["x"], "z": sp["z"]}
    start = spawn_position(nav, sp).round(3).tolist()
    nav_path.write_text(json.dumps(nav), encoding="utf-8")
    manifest["startPosition"], manifest["startYawDegrees"] = start, round(math.degrees(sp["yaw"]), 2)
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    poi_path = bundle / manifest["semanticUri"]
    pois = json.loads(poi_path.read_text())
    for p in pois:
        if p.get("id") == "start":
            p["position"] = p["approachPoint"] = start
    poi_path.write_text(json.dumps(pois, indent=1), encoding="utf-8")
    after = grid_stats(nav)
    q_path = bundle / manifest["qualityUri"]
    q = json.loads(q_path.read_text())
    q.setdefault("measured", {}).update(after)
    q_path.write_text(json.dumps(q, indent=1), encoding="utf-8")
    navgrid_png(nav, np.zeros((0, 2)), bundle / "quality" / "navgrid.png")
    print("before:", summary_line(before))
    print("after: ", summary_line(after))
    return {"before": before, "after": after, "start": start, "yaw_degrees": manifest["startYawDegrees"]}


def summary_line(stats: dict) -> str:
    return (f"nav grid: walkable {stats['walkable_m2']} m2 in {stats['components']} component(s), "
            f"spawn component {stats['spawn_component_m2']} m2, unknown {stats['unknown_m2']} m2")


def world_alignment(sparse: Path, scratch: Path, metres_per_unit: float | None = None) -> dict:
    """COLMAP frame -> bundle world frame: world = s * R @ x + t (levelled, metres, floor at y=0, walked path
    centred on x=z=0). Also returns what build() needs from the model, in the levelled frame."""
    pts, tracks, centres, axes = load_model(sparse, scratch)
    R = level_rotation(centres, axes[:, 1])
    P, C = (pts @ R.T)[tracks >= 3], centres @ R.T
    keep = np.all((P >= np.percentile(P, 1, axis=0)) & (P <= np.percentile(P, 99, axis=0)), axis=1)
    P = P[keep]
    s, floor, ceiling = estimate_scale(P, C)
    if metres_per_unit:
        s = metres_per_unit  # the floor and ceiling still come from the points
    mid = C.mean(0)
    t = -s * np.array([mid[0], floor, mid[2]])
    return {"s": float(s), "R": R, "t": t, "P": P, "C": C, "centres": centres, "axes": axes,
            "floor": floor, "ceiling": ceiling}


def world_matrix(a: dict) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = a["s"] * a["R"], a["t"]
    return M


# ---------------------------------------------------------------- baking the transform into a splat
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
         1.445305721320277, -0.5900435899266435)


def sh_basis(d: np.ndarray, degree: int) -> np.ndarray:
    """Real SH basis without the DC term, (N, (degree+1)^2 - 1), as 3DGS/gsplat/PlayCanvas evaluate it."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    cols = []
    if degree >= 1:
        cols += [-SH_C1 * y, SH_C1 * z, -SH_C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        cols += [SH_C2[0] * x * y, SH_C2[1] * y * z, SH_C2[2] * (2 * zz - xx - yy), SH_C2[3] * x * z,
                 SH_C2[4] * (xx - yy)]
    if degree >= 3:
        cols += [SH_C3[0] * y * (3 * xx - yy), SH_C3[1] * x * y * z, SH_C3[2] * y * (4 * zz - xx - yy),
                 SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy), SH_C3[4] * x * (4 * zz - xx - yy),
                 SH_C3[5] * z * (xx - yy), SH_C3[6] * x * (xx - 3 * yy)]
    return np.stack(cols, 1)


def sh_rotation(R: np.ndarray, degree: int) -> np.ndarray:
    """X with new_coeffs = X @ old_coeffs, so the rotated splat shows at direction d what the original showed
    at R^T d. Each band maps onto itself, so a least-squares fit on random directions is exact."""
    d = np.random.default_rng(0).normal(size=(256, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    A, B = sh_basis(d, degree), sh_basis(d @ R, degree)  # (d @ R)[i] = R^T d_i
    return np.linalg.lstsq(A, B, rcond=None)[0]  # Y(d) @ X = Y(R^T d), so f_new(e) = Y(e) @ X @ c = f(R^T e)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions (broadcasts)."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def bake_ply(src: Path, dst: Path, s: float, R: np.ndarray, t: np.ndarray) -> int:
    """Write `src` (float 3DGS PLY, COLMAP frame) as `dst` in the world frame world = s * R @ x + t:
    positions, log-scales (+log s), rotations (R * q) and SH bands 1-3 (rotated). Returns the splat count."""
    from clean_splat import read_ply, write_ply

    if abs(np.linalg.det(R) - 1) > 1e-6:
        raise ValueError("world alignment must be a proper rotation")
    header, props, d = read_ply(src)
    col = {k: i for i, k in enumerate(props)}
    d = d.astype(np.float64)
    xyz = [col[k] for k in ("x", "y", "z")]
    d[:, xyz] = s * d[:, xyz] @ R.T + t
    for k in ("scale_0", "scale_1", "scale_2"):
        d[:, col[k]] += math.log(s)
    rot = [col[f"rot_{i}"] for i in range(4)]
    q = quat_multiply(matrix_to_quat(R), d[:, rot])
    d[:, rot] = q / np.linalg.norm(q, axis=1, keepdims=True)
    rest = sorted((k for k in props if k.startswith("f_rest_")), key=lambda k: int(k[7:]))
    if rest:
        k = len(rest) // 3  # coefficients per channel (channel-major: all R, then G, then B)
        degree = {3: 1, 8: 2, 15: 3}[k]
        X = sh_rotation(R, degree)
        for c in range(3):
            cols = [col[rest[c * k + i]] for i in range(k)]
            d[:, cols] = d[:, cols] @ X.T
    for k in ("nx", "ny", "nz"):
        if k in col:
            d[:, col[k]] = 0.0
    write_ply(dst, header, d.astype(np.float32))
    return len(d)


def build(sparse: Path, splat: Path, out: Path, title: str, quality: dict | None = None,
          metres_per_unit: float | None = None, dense_splat: Path | None = None,
          filmed_radius: float = FILMED_RADIUS_M, reach_radius: float = REACH_RADIUS_M) -> dict:
    """`splat` must already be in the world frame (bake_ply with world_alignment's transform). `dense_splat`
    (default: `splat` itself when it is a PLY) is the world-frame splat read for dense nav evidence."""
    a = world_alignment(sparse, out / "_sparse_txt", metres_per_unit)
    R, s, t, P, C, centres, axes = a["R"], a["s"], a["t"], a["P"], a["C"], a["centres"], a["axes"]
    floor, ceiling = a["floor"], a["ceiling"]
    to_world = lambda X: s * X + t  # levelled model units -> metres, floor at y=0, path centred on 0
    W, Cw = to_world(P), to_world(C)

    lo = np.percentile(W, 1, axis=0)
    hi = np.percentile(W, 99, axis=0)
    lo[1], hi[1] = 0.0, max(float(hi[1]), 2.2)
    # The grid must hold every camera position (the start is one): on a partial reconstruction the walked path
    # can stick out past the 1st/99th percentile of the points.
    lo[[0, 2]] = np.minimum(lo[[0, 2]], Cw[:, [0, 2]].min(0) - CELL_M)
    hi[[0, 2]] = np.maximum(hi[[0, 2]], Cw[:, [0, 2]].max(0) + CELL_M)
    # Spawn in the largest walkable component, on a cell the (non-outlier) cameras stood in, facing open space.
    # (room_coverage.viewer_alignment's spawn ignores the grid: on the hostel capture it picked a mis-posed first
    # frame in a 0.86 m2 island.)
    outlier, _ = pose_outliers(Cw)
    samples = splat_samples(dense_splat or splat)
    nav = navigation_grid(W, Cw, lo, hi, None, axes[:, 0] @ R.T, samples, filmed_radius, reach_radius)
    floor_m2 = round(nav.pop("_floorEvidenceCells") * CELL_M ** 2, 2)
    sp = choose_spawn(nav, Cw[~outlier][:, [0, 2]])
    start = spawn_position(nav, sp)

    out.mkdir(parents=True, exist_ok=True)
    for d in ("visual", "navigation", "semantic", "quality"):
        (out / d).mkdir(exist_ok=True)
    shutil.copy2(splat, out / "visual" / splat.name)
    (out / "navigation" / "grid.json").write_text(json.dumps(nav), encoding="utf-8")
    pois = [{"id": "start", "label": "Start", "category": "entrance", "position": start.round(3).tolist(),
             "approachPoint": start.round(3).tolist(), "aliases": ["spawn", "entrance"],
             "description": "Where the walk-through begins (a real camera position from the capture).",
             "confidence": 1, "source": "operator"}]
    (out / "semantic" / "pois.json").write_text(json.dumps(pois, indent=1), encoding="utf-8")
    room_h = (ceiling - floor) * s
    warnings = [] if metres_per_unit else [
        f"Scale is estimated from an assumed {ASSUMED_CAMERA_HEIGHT_M} m camera height "
        f"(room height comes out at {room_h:.2f} m)."]
    if not 2.2 <= room_h <= 4.5:
        warnings.append(f"Estimated room height {room_h:.2f} m is implausible: the scale guess is probably off.")
    nav_stats = grid_stats(nav)
    navgrid_png(nav, Cw[:, [0, 2]], out / "quality" / "navgrid.png")
    q = quality or {
        "status": "needs-review", "readiness": "navigation_ready", "warnings": warnings,
        "checks": [{"id": "scale", "label": "Scale", "status": "passed" if metres_per_unit else "warning",
                    "detail": f"{s:.3f} m per model unit, {'known' if metres_per_unit else 'estimated'}"}]}
    q.setdefault("measured", {}).update(nav_stats)  # the studio schema's free-form `measured` record
    (out / "quality" / "report.json").write_text(json.dumps(q, indent=1), encoding="utf-8")

    notes = [*warnings, f"Reconstruction-to-world transform (scale {s:.4f}) baked into the splat by "
                        "scene_bundle.bake_ply, so worldFromReconstruction is identity."]
    manifest = {
        "schemaVersion": 1, "sceneId": out.name, "displayName": title,
        "sceneKind": "reconstructed-capture",
        "description": "Gaussian splat reconstructed from a smartphone video (COLMAP + Gaussian-splat training).",
        "coordinateSystem": "right-handed-y-up", "units": "metre",
        "worldFromReconstruction": np.eye(4, dtype=int).tolist(),  # the splat is baked (see bake_ply)
        "bounds": {"min": lo.round(3).tolist(), "max": hi.round(3).tolist()},
        "startPosition": start.round(3).tolist(),
        "startYawDegrees": round(math.degrees(sp["yaw"]), 2),
        "visualAssets": [{"kind": splat.suffix.lstrip("."), "uri": f"visual/{splat.name}", "label": title,
                          "sha256": hashlib.sha256(splat.read_bytes()).hexdigest()}],
        "collisionAsset": None, "navigationUri": "navigation/grid.json",
        "semanticUri": "semantic/pois.json", "qualityUri": "quality/report.json",
        "scaleStatus": "known" if metres_per_unit else "estimated",
        "provenance": {"pipeline": "pipeline/scene_bundle.py", "pipelineVersion": "0.1.0",
                       "capturedAt": "", "source": "Smartphone video capture", "notes": notes},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    shutil.rmtree(out / "_sparse_txt", ignore_errors=True)
    stats = {"metres_per_unit": round(s, 4), "pose_outliers": int(outlier.sum()),
             "dense_evidence": samples is not None, "floor_evidence_m2": floor_m2, "room_height_m": round(room_h, 2),
             "room_size_m": (hi - lo).round(2).tolist(), "grid": [nav["width"], nav["height"]],
             "blocked_cells": len(nav["blocked"]), "unknown_cells": len(nav["unknown"]), **nav_stats}
    print("scene_bundle:", json.dumps(stats))
    print(summary_line(nav_stats))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a scene bundle; or, with --bake, move a float PLY into the "
                                             "bundle's world frame first.")
    ap.add_argument("sparse", type=Path)
    ap.add_argument("splat", type=Path, help="world-frame web splat (or, with --bake, the COLMAP-frame float PLY)")
    ap.add_argument("out", type=Path, help="bundle folder (or, with --bake, the world-frame PLY to write)")
    ap.add_argument("--bake", action="store_true")
    ap.add_argument("--respawn", action="store_true", help="only re-pick the spawn of the existing bundle at <out>")
    ap.add_argument("--title", default="Captured room")
    ap.add_argument("--metres-per-unit", type=float, help="known model scale; skips the camera-height guess")
    ap.add_argument("--dense-splat", type=Path, help="world-frame float PLY for dense nav evidence "
                                                     "(default: the web splat when it is a PLY)")
    ap.add_argument("--filmed-radius", type=float, default=FILMED_RADIUS_M,
                    help=f"metres from the walked path that count as filmed (default {FILMED_RADIUS_M})")
    ap.add_argument("--reach-radius", type=float, default=REACH_RADIUS_M,
                    help=f"seen floor this close to the path is walkable (default {REACH_RADIUS_M}; 0 turns it off)")
    a = ap.parse_args()
    if a.respawn:
        respawn(a.out)
    elif a.bake:
        al = world_alignment(a.sparse, a.out.parent / "_sparse_txt", a.metres_per_unit)
        shutil.rmtree(a.out.parent / "_sparse_txt", ignore_errors=True)
        print(f"baked {bake_ply(a.splat, a.out, al['s'], al['R'], al['t'])} splats -> {a.out}")
    else:
        build(a.sparse, a.splat, a.out, a.title, metres_per_unit=a.metres_per_unit, dense_splat=a.dense_splat,
              filmed_radius=a.filmed_radius, reach_radius=a.reach_radius)
