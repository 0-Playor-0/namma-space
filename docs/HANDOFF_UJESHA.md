# Handoff: Ujesha — finish and deploy Namma Space

Written 18 Nov 2026. You own everything from here to a **completed, deployed project**, except the submission
material (screen recordings, demo video, PDFs, abstract): another team member does those. Your job ends when
production serves the finished app and you have given that teammate the URL and the facts in step 8.

Kaggle is dropped: it was too fiddly to keep working. Reconstruction now runs on our own GPU machines (step 2).

---

## 0. Where things stand

| Thing | State |
|---|---|
| Code | Branch `claude/modest-dirac-fm1mih` is 3+ commits ahead of `main`: hostel scene, spawn fix, pose-outlier filter, dense nav evidence, `--metres-per-unit`. Python 165 tests and vitest 828 tests pass; lint and typecheck are clean. |
| Production | <https://namma-space-pearl.vercel.app> still serves **Round 1**. Vercel is connected to GitHub: a merge to `main` deploys production, any other branch gets a preview (behind Vercel login). |
| Scenes | `classroom` (default, keep it default unless the team decides otherwise), `real-room`, `apartment`, fixtures, and new `hostel`. |
| Hostel | Installed at `apps/studio/public/scenes/hostel/` from a Kaggle **preview** run. Opens with `?scene=hostel`; it is **not** in the scene switcher. It is very blurry and its nav grid is poor: walkable 16.24 m² in 18 pieces, 202 m² "unknown". The spawn was moved into the main area (spawn area 0.86 → 15.11 m²). Scale is a guess (0.4969 m per unit). |
| Why the hostel is bad | (1) Preview training: 7k steps at 720p; the loss had not converged and 331k of 500k splats were near-invisible. (2) Coverage: only about half the floor and 24–36 % of each wall was filmed. (3) The nav grid closes off anything over 0.5 m from the walked path, and stray SfM points become obstacles. The camera poses were excellent (862/864 frames, 0.70 px), so the recording technique is fine; it just needs to cover more. |
| Open issues | `docs/OPEN_ISSUES.md`: read it once now, and update it as you close things. |

---

## 1. Set up (about 30 min)

```bash
git clone https://github.com/0-Playor-0/namma-space && cd namma-space
git checkout claude/modest-dirac-fm1mih
cd apps/studio && npm ci && npm test && npm run lint && npm run build && cd ../..
```
Python tests (from the repo root, in a venv with `pipeline/requirements.txt` + pytest): `python -m pytest -q tests/`.

**Done when** all of those pass on your machine.

## 2. Reconstruction machine (replaces Kaggle)

Use the lab PC (Windows, A4000, 16 GB) if you can; otherwise the team laptop (RTX 5060, 8 GB). On Windows:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_tools.ps1
.venv\Scripts\python pipeline\run_pipeline.py --doctor
```
Windows uses COLMAP + Brush (no CUDA toolkit needed). Linux uses `scripts/setup_tools.sh` (pycolmap + gsplat).
On the 8 GB laptop add `--max-splats 1000000` if it runs out of GPU memory. Details: `docs/TRAINING.md`.

**Done when** `--doctor` reports every tool OK.

## 3. Re-film the hostel (about 1 h on site)

Follow `docs/SOP.md`. Specifically for the hostel:
- **Two laps**: one slowly along the walls looking into the room, one through the middle looking out. Cover the
  gaps listed in the last run's coverage report. In world terms, the far side and the floor/ceiling were least
  seen. Every 30 cm of floor you want people to walk on must be filmed from close by.
- Landscape, 1080p 30 fps, **lock exposure and focus**, no fast turns, good light, ~1–2 min per room. Split a big
  space into several clips/scenes rather than one 5-minute walk.
- **No people in frame.** If anyone gets in, run with `--mask-people`. A splat with visible faces must not be
  committed or deployed.
- **Measure one real distance** with a tape (e.g. a door's width, wall to wall across a corridor) and write it down.
- Start the clip in an open spot facing into the space; the first seconds are where poses most often go wrong.

## 4. Reconstruct (final quality)

```powershell
.venv\Scripts\python pipeline\run_pipeline.py C:\captures\hostel.mp4 --out work\hostel --profile final --title Hostel
```
`final` = 1080p, 30k steps, up to 1.5M splats. Expect a few hours; the stage times land in
`work\hostel\report.json`. The run prints the nav-grid summary and writes `work\hostel\bundle\`.

**Fix the scale with your measurement:**
1. Open the bundle in the studio (step 5), stand at one end of the measured distance and note X/Z from the
   position readout (top right), then at the other end. The distance between the two points is `measured`.
2. `new = report.json → world_frame.metres_per_unit × real / measured`.
3. Re-run the same command with `--metres-per-unit <new>` (finished stages are reused; only world/compress/publish
   redo). The manifest then says `scaleStatus: known`.

## 5. Inspect, tune, approve

Copy `work\hostel\bundle\*` over `apps/studio/public/scenes/hostel/`, then `npm run dev` in `apps/studio` and open
`http://localhost:5173/?scene=hostel`. Check:
- **Look**: walk the filmed path; it should be sharp near where you walked. Note any blurry areas honestly.
- **Nav picture** `quality/navgrid.png`: the spawn is in the big walkable area; no long straight orange lines (pose
  jumps); no walkable (light green) cells inside beds, desks or walls.
- **Numbers** in `quality/report.json → measured`: aim for most of the real walkable floor, `components` small,
  `spawn_component_m2` ≈ `walkable_m2`.

If too much real floor is still closed off, re-run only the bundle with looser limits (the sparse model and the
world splat are in the work folder):
```bash
python pipeline/scene_bundle.py work/hostel/dataset/sparse work/hostel/web/<scene file> work/hostel/bundle --title Hostel --dense-splat work/hostel/world/splat_world.ply --filmed-radius 0.75 --reach-radius 2.0
```
Add `--metres-per-unit <new>` there too if you set it. Only keep looser values if no walkable cells appear inside
furniture. Then record the inspection:
```bash
python pipeline/run_pipeline.py --approve work/hostel --note "walked it: <what you checked>"
```
Finally `cd apps/studio && npm run validate:scenes`.

**Done when** the hostel looks acceptable, the nav grid is sensible, it is approved, and validation passes.

## 6. Finish the app

In priority order (each also listed in `docs/OPEN_ISSUES.md`):
1. **POIs.** The classroom and hostel have only "Start". Tag the real places (door, desks, board, beds, sinks…)
   with the POI editor (`?scene=<name>&edit=1`), export, commit each `semantic/pois.json`.
2. **Scene switcher.** Add the hostel to `SCENES` in `apps/studio/src/components/viewer/sceneOptions.ts` once it
   is good enough to show. Do **not** change `DEFAULT_SCENE_NAME` unless the team agrees. Update
   `apps/studio/tests/scene-options.test.ts` to match.
3. **Real phones.** Test on one real Android and one iPhone (open the scene, look, Follow route); fix what breaks.
4. **Docs.** Update `README.md` (Kaggle is no longer the main path; mention the hostel) and `docs/OPEN_ISSUES.md`.

## 7. Merge and deploy to production

1. Everything green: `npm test`, `npm run lint`, `npm run build`, `npm run predeploy` (in `apps/studio`), and
   `python -m pytest -q tests/`.
2. Push the branch; check its Vercel preview (log in to Vercel to open it).
3. **Round 1 links.** Before merging to `main`, confirm with the team that Round 1 judging no longer needs the old
   links (see the Deployment section of `docs/OPEN_ISSUES.md`).
4. Open a pull request into `main`, merge it. Vercel deploys production automatically.
5. Open <https://namma-space-pearl.vercel.app> in a private window: default scene loads, `?scene=hostel` loads,
   search, a route and Follow route work, no console errors.

**Done when** production serves this build and the checks in 5 pass.

## 8. Hand to the submission teammate

Send them: the production URL; which scenes to show and their deep links (e.g. `?scene=hostel&poi=<id>&route=1`);
each scene's numbers from `quality/report.json` (walkable m², scale known/estimated) and training stage times from
`report.json`; and the honest limitations (blur in unfilmed areas, scale accuracy, one floor only).

---

## Rules

- No faces or identifiable people in any committed or deployed splat.
- Keep `classroom` as the default scene unless the team decides otherwise.
- Commits are authored by the team member who made them, with no AI co-author or attribution lines.
- `.claude/` (local editor/agent settings) is git-ignored; do not commit it.
- Large raw outputs (`work/`, `tools/`, float `.ply` files) stay out of git; only `bundle/` contents go into
  `apps/studio/public/scenes/<name>/`.
