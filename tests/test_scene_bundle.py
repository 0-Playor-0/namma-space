"""Scene bundle for the studio viewer: estimated scale and navigation grid on a synthetic room."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")
import scene_bundle  # noqa: E402  (conftest puts pipeline/ on sys.path)
from test_coverage import write_model  # noqa: E402


def test_bundle_scale_walls_and_start(tmp_path):
    # 4 x 2.6 x 6 m box, cameras at 1.4 m: the guess (1.45 m camera height) is within 4% of the truth.
    model = write_model(tmp_path / "m", 0.3, n_cams=96)  # dense enough for MIN_VIEW_FRAMES per direction
    splat = tmp_path / "scene.sog"
    splat.write_bytes(b"not a real splat")
    stats = scene_bundle.build(model, splat, tmp_path / "bundle", "Synthetic")
    assert stats["room_height_m"] == pytest.approx(2.6 * 1.45 / 1.4, rel=0.05)

    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    grid = json.loads((tmp_path / "bundle" / manifest["navigationUri"]).read_text())
    assert manifest["scaleStatus"] == "estimated" and manifest["units"] == "metre"
    assert (tmp_path / "bundle" / manifest["visualAssets"][0]["uri"]).is_file()

    blocked = np.zeros(grid["width"] * grid["height"], bool)
    blocked[grid["blocked"]] = True
    blocked[grid["unknown"]] = True
    assert not set(grid["blocked"]) & set(grid["unknown"])        # the viewer rejects overlapping lists
    occ = blocked.reshape(grid["height"], grid["width"])          # [z, x]
    s = grid["start"]
    assert not occ[s["z"], s["x"]]                                # the visitor starts on free floor
    assert occ[:, 0].all() and occ[:, -1].all()                   # walls along both x edges block walking
    # The camera ring (radius 1-1.5 m) is walkable all the way round; the far corners, never filmed, are not.
    assert (~occ).sum() * grid["cellSize"] ** 2 > 2.0
    assert occ[0, 0] and occ[-1, -1]
    start_y = manifest["startPosition"][1]
    assert start_y == pytest.approx(scene_bundle.ASSUMED_CAMERA_HEIGHT_M)
    # Every walkable cell near the camera ring knows which way the phone faced; occupied cells carry none.
    views = np.array(grid["views"]).reshape(grid["height"], grid["width"])
    assert grid["viewSectors"] == scene_bundle.VIEW_SECTORS
    assert views[s["z"], s["x"]] > 0 and not views[occ].any()
    assert views.max() < 1 << grid["viewSectors"]


def test_bundle_known_scale(tmp_path):
    # A model already in metres: the given scale is used as-is and reported as known.
    model = write_model(tmp_path / "m", 0.3)
    splat = tmp_path / "scene.sog"
    splat.write_bytes(b"not a real splat")
    stats = scene_bundle.build(model, splat, tmp_path / "bundle", "Synthetic", metres_per_unit=1.0)
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    assert stats["metres_per_unit"] == 1.0 and manifest["scaleStatus"] == "known"
    assert stats["room_height_m"] == pytest.approx(2.6, rel=0.05)


# ------------------------------------------------------------------ world frame baked into the splat
def _rotmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _random_rotation(seed):
    q = np.random.default_rng(seed).normal(size=4)
    return _rotmat(q)


def test_sh_rotation_is_exact_for_every_band():
    R = _random_rotation(3)
    rng = np.random.default_rng(1)
    for degree in (1, 2, 3):
        X = scene_bundle.sh_rotation(R, degree)
        c = rng.normal(size=X.shape[0])
        d = rng.normal(size=(50, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        # baked splat seen from R d == original seen from d
        assert np.allclose(scene_bundle.sh_basis(d @ R.T, degree) @ (X @ c), scene_bundle.sh_basis(d, degree) @ c,
                           atol=1e-9)


def test_bake_ply_moves_positions_scales_rotations_and_colour(tmp_path):
    import clean_splat
    import train_gsplat
    from test_kaggle_pipeline import _fake_splats
    src, dst = tmp_path / "in.ply", tmp_path / "out.ply"
    f = _fake_splats(200)
    train_gsplat.write_ply(src, **f)
    R, s, t = _random_rotation(5), 0.37, np.array([1.0, -2.0, 0.5])
    assert scene_bundle.bake_ply(src, dst, s, R, t) == 200
    _, props, a = clean_splat.read_ply(src)
    _, _, b = clean_splat.read_ply(dst)
    col = {k: i for i, k in enumerate(props)}
    xyz = [col[k] for k in "xyz"]
    assert np.allclose(b[:, xyz], s * a[:, xyz] @ R.T + t, atol=1e-4)
    sc = [col[f"scale_{i}"] for i in range(3)]
    assert np.allclose(b[:, sc], a[:, sc] + np.log(s), atol=1e-5)
    rot = [col[f"rot_{i}"] for i in range(4)]
    for i in range(5):
        assert np.allclose(_rotmat(b[i, rot]), R @ _rotmat(a[i, rot]), atol=1e-5)
    dc = [col[f"f_dc_{i}"] for i in range(3)]
    assert np.allclose(b[:, dc], a[:, dc])
    # view-dependent colour (red channel) seen from R d after baking equals the original seen from d
    rest = [col[f"f_rest_{i}"] for i in range(15)]
    d = np.random.default_rng(2).normal(size=(10, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    before = scene_bundle.sh_basis(d, 3) @ a[:, rest].T
    after = scene_bundle.sh_basis(d @ R.T, 3) @ b[:, rest].T
    assert np.allclose(before, after, atol=1e-4)


def test_pipeline_bundle_passes_the_studio_validator(tmp_path):
    """scene_bundle.py on a small fixture model -> baked, compressed splat -> validate_scene.py exits 0."""
    import subprocess
    import compressed_ply
    import train_gsplat
    from test_kaggle_pipeline import _fake_splats
    model = write_model(tmp_path / "m", 0.3, n_cams=96)
    raw, world, web = tmp_path / "splat.ply", tmp_path / "world.ply", tmp_path / "scene.compressed.ply"
    train_gsplat.write_ply(raw, **_fake_splats(600))
    al = scene_bundle.world_alignment(model, tmp_path / "txt")
    scene_bundle.bake_ply(raw, world, al["s"], al["R"], al["t"])
    compressed_ply.convert(world, web)
    scene_bundle.build(model, web, tmp_path / "bundle", "Fixture")
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    assert manifest["worldFromReconstruction"] == np.eye(4, dtype=int).tolist()
    validator = Path(__file__).resolve().parent.parent / "apps" / "studio" / "scripts" / "validate_scene.py"
    proc = subprocess.run([sys.executable, str(validator), str(tmp_path / "bundle")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ------------------------------------------------------------------ nav-grid diagnostic
def _grid(rows: list[str], start=(0, 0)) -> dict:
    """'.' walkable, '#' blocked, '?' unknown; 0.1 m cells."""
    w, h = len(rows[0]), len(rows)
    flat = "".join(rows)
    return {"width": w, "height": h, "cellSize": 0.1, "origin": {"x": 0.0, "z": 0.0},
            "blocked": [i for i, ch in enumerate(flat) if ch == "#"],
            "unknown": [i for i, ch in enumerate(flat) if ch == "?"],
            "start": {"x": start[0], "z": start[1]}}


def test_grid_stats_counts_components_like_the_route_planner():
    # the two walkable areas touch only diagonally: the planner cannot cut that corner, so 2 components
    nav = _grid(["..#?",
                 "..#?",
                 "##.?",
                 "??.?"], start=(1, 1))
    st = scene_bundle.grid_stats(nav)
    assert st == {"walkable_m2": 0.06, "components": 2, "spawn_component_m2": 0.04, "unknown_m2": 0.06}
    assert scene_bundle.grid_stats({**nav, "start": {"x": 3, "z": 0}})["spawn_component_m2"] == 0.0  # spawn off-grid
    assert "walkable 0.06 m2 in 2 component(s)" in scene_bundle.summary_line(st)


def test_navgrid_png_and_quality_numbers(tmp_path):
    import cv2
    import train_gsplat
    from test_kaggle_pipeline import _fake_splats
    model = write_model(tmp_path / "m", 0.3, n_cams=96)
    splat = tmp_path / "scene.ply"
    train_gsplat.write_ply(splat, **_fake_splats(100))
    stats = scene_bundle.build(model, splat, tmp_path / "bundle", "Synthetic")
    nav = json.loads((tmp_path / "bundle" / "navigation" / "grid.json").read_text())
    img = cv2.imread(str(tmp_path / "bundle" / "quality" / "navgrid.png"))
    assert img.shape[0] > nav["height"] * scene_bundle.NAVGRID_PX and img.shape[1] > nav["width"] * scene_bundle.NAVGRID_PX
    rgb = img[..., ::-1].reshape(-1, 3)
    for colour in scene_bundle.NAVGRID_COLOURS.values():  # all three cell kinds are drawn
        assert (rgb == colour).all(1).any()
    q = json.loads((tmp_path / "bundle" / "quality" / "report.json").read_text())
    keys = {"walkable_m2", "components", "spawn_component_m2", "unknown_m2"}
    assert keys <= set(q["measured"]) and keys <= set(stats)
    assert 0 < q["measured"]["spawn_component_m2"] <= q["measured"]["walkable_m2"]


# ------------------------------------------------------------------ spawn and pose outliers
def _walk(n, start, step=(0.1, 0.0)):
    return np.array([[start[0] + i * step[0], 1.4, start[1] + i * step[1]] for i in range(n)])


def test_spawn_goes_to_largest_component_near_the_path_facing_open_space():
    # A 0.3 m island (where a mis-posed first frame would put the spawn) and a 3 m x 1 m room; one camera
    # stood in the island and one in the room's west end. The room is the largest component.
    rows = ["???????????????????????????????????",
            "?...?#..............................",
            "?...?#..............................",
            "?...?#..............................",
            "?????#..............................",
            "?????#..............................",
            "?????#..............................",
            "?????#..............................",
            "?????#..............................",
            "?????#..............................",
            "?????#..............................",
            "???????????????????????????????????"]
    nav = _grid([r.ljust(36, "?") for r in rows], start=(2, 2))
    cams = np.array([[0.25, 0.25], [0.75, 0.55]])  # (x, z) metres: island cell (2, 2), room cell (7, 5)
    sp = scene_bundle.choose_spawn(nav, cams)
    assert (sp["x"], sp["z"]) == (7, 5)  # the only camera cell in the big room
    nav["start"] = {"x": sp["x"], "z": sp["z"]}
    st = scene_bundle.grid_stats(nav)
    assert st["components"] == 2 and st["spawn_component_m2"] > 2.5
    # Facing the long axis of the room (+x): forward = (-sin yaw, -cos yaw) points along +x.
    assert -math.sin(sp["yaw"]) > 0.9
    # Without camera positions it still lands in the big room, on its roomiest cell.
    sp2 = scene_bundle.choose_spawn(nav)
    assert sp2["x"] >= 6 and 3 <= sp2["z"] <= 8


def test_pose_jumps_are_outliers_and_seed_no_walkable_cells():
    # 3 mis-posed frames 6 m away, then a 60-frame walk with one 4 m jump in the middle (a teleport between two
    # real stretches). The 3 frames are outliers; the jump is not walked, so the wall it crosses stays blocked.
    bad = _walk(3, (-6.0, 0.0))
    a = _walk(30, (0.0, 0.0))                      # x 0 .. 2.9, z 0
    b = _walk(30, (0.0, 2.0))                      # x 0 .. 2.9, z 2
    C = np.concatenate([bad, a, b])                # (2.9, 0) -> (0, 2): a 3.5 m jump through the wall at z = 1
    outlier, jump = scene_bundle.pose_outliers(C)
    assert outlier[:3].all() and not outlier[3:].any()
    assert jump[2] and jump[32] and jump.sum() == 2

    wall = np.array([[x, 1.0, z] for x in np.arange(-7, 4, 0.03) for z in (0.95, 1.0, 1.05) for _ in range(2)])
    lo, hi = np.array([-7.0, 0, -0.6]), np.array([4.0, 2.2, 2.6])
    nav = scene_bundle.navigation_grid(wall, C, lo, hi)
    occ = ~scene_bundle.walkable_mask(nav)
    cell = lambda x, z: (int((z - lo[2]) / 0.1), int((x - lo[0]) / 0.1))
    assert occ[cell(1.45, 1.0)]                    # the jump's midpoint is still wall
    assert occ[cell(-5.9, 0.0)]                    # the outlier frames seed nothing
    assert not occ[cell(1.0, 0.0)] and not occ[cell(1.0, 2.0)]
    s = nav["start"]
    assert not occ[s["z"], s["x"]] and s["x"] >= cell(0.0, 0.0)[1]


def test_respawn_moves_an_island_spawn_into_the_main_area(tmp_path):
    model = write_model(tmp_path / "m", 0.3, n_cams=96)
    splat = tmp_path / "scene.sog"
    splat.write_bytes(b"not a real splat")
    scene_bundle.build(model, splat, tmp_path / "bundle", "Synthetic")
    grid_path = tmp_path / "bundle" / "navigation" / "grid.json"
    nav = json.loads(grid_path.read_text())
    walk = scene_bundle.walkable_mask(nav)
    # carve an island: free a cell surrounded by unknown in a corner and put the spawn there
    unknown = set(nav["unknown"])
    idx = 1 * nav["width"] + 1
    unknown.discard(idx)
    nav["blocked"] = [i for i in nav["blocked"] if i != idx]
    nav["unknown"] = sorted(unknown)
    nav["start"] = {"x": 1, "z": 1}
    grid_path.write_text(json.dumps(nav))
    assert not walk[1, 1]
    r = scene_bundle.respawn(tmp_path / "bundle")
    assert r["before"]["spawn_component_m2"] == 0.01
    assert r["after"]["spawn_component_m2"] == r["after"]["walkable_m2"] - 0.01  # all but the island
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    pois = json.loads((tmp_path / "bundle" / "semantic" / "pois.json").read_text())
    assert manifest["startPosition"] == pois[0]["position"]
