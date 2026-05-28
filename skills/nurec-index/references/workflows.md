# Common NuRec workflows

Each workflow lists the sibling skills to read **in order**, with a
one-line summary of what to do inside each one. Open the named skill
for the full recipe — do not try to follow these steps from this
page alone.

## A. Make a NuRec scene from your own recording

Use this when the user has a fresh sensor log and wants a renderable
3D scene at the end.

1. `ncore` — convert the recording into NCore V4. The skill has a
   built-in converter for common formats (PAI, Waymo, NuScenes,
   PandaSet, COLMAP, ScanNet++); for anything else, it walks the
   agent through writing a new converter.
2. `nre` — generate the extra inputs (depth, segmentation, ego
   mask), then train and validate. Out comes a USDZ. Render it three
   ways: with the local `nre render` CLI; with a warm `serve-grpc`
   server driven by the bundled thin Python gRPC client
   (`batch_render_rgb` for repeated / multi-camera renders); or by
   handing the USDZ to a simulator over the public gRPC API.

## B. Use a NuRec scene NVIDIA has already trained

Use this when the user just wants to see NuRec working without
training anything themselves.

1. `physical-ai-datasets` — accept the gated license on Hugging
   Face, then download **one** scene (~1.5–2 GB) from
   `PhysicalAI-Autonomous-Vehicles-NuRec`. The full dataset is
   ~1.5 TB, so do not pull all of it.
2. `nre` — render the USDZ. The "highest quality" preset in that
   skill renders at the original resolution along the original
   camera positions; new camera positions are available via the
   gRPC server.

## C. Use NuRec for indoor robot simulation

1. `physical-ai-datasets` — download `PhysicalAI-Robotics-NuRec`
   (62.9 GB of indoor scenes: cafés, offices, hand-held captures).
2. `nre` — optional: re-train the scene to tweak it, or open it in
   the viewer to inspect.
3. Hand the USDZ to **Isaac Sim 5.1** for AMR (autonomous mobile
   robot) simulation. There is no skill for this step in this index;
   use the Isaac Sim docs directly.

## D. Add, remove, or replace 3D objects in a scene

1. `ncore` — confirm the original NCore clip is still on disk;
   Asset Harvester needs it to crop the object views.
2. `asset-harvester` — point it at the object IDs of interest. For
   each one, it produces a `.ply` file (the 3D Gaussian model) plus
   a `metadata.yaml` (size, position, label).
3. `nre` — package those `.ply` files back into the USDZ and edit
   the scene with `serve-grpc --enable-editing-actors` plus
   `render-grpc --edit-assets`. The skill has a JSON schema for the
   add / remove / replace operations.

## E. Clean up rendered frames

NuRec sometimes leaves visible artifacts (floating dots, ghosting,
flickering between frames) or object-insertion mismatches (lighting,
shadows, color). Two ways to fix this — pick one:

- **Quick path** — turn on `--enable-difix` when starting the gRPC
  server in `nre`. NRE owns this inline rendering integration.
  Default for users who are already rendering through NRE.
- **Standalone path** — render frames first with `nre`, then run
  `nurec-fixer` on the folder of frames. Use this for the public
  DiffusionHarmonizer code/model card, paired evaluation,
  fine-tuning, or fixes for frames rendered earlier without
  re-running NRE.

## F. Benchmark reconstruction quality

1. `physical-ai-datasets` — download `PhysicalAI-NuRec-PPISP`
   (15 GB of outdoor scenes shot at three exposure levels for fair
   comparisons).
2. `ncore` — only needed for re-building the NCore shards. The
   dataset already ships with both COLMAP and NCore V4 versions,
   so usually this step is skipped.
3. `nre` — train, then run `eval-rendering-metrics` against the
   ground-truth frames the dataset includes.

## G. Connect NuRec to a simulator

CARLA, Isaac Sim, AlpaSim, or a custom simulator can ask NRE for
frames over a network API.

1. `physical-ai-datasets` — pick a USDZ if one is not already on
   disk.
2. `nre` — start the server with `serve-grpc`. The simulator sends
   it a camera position and a timestamp; NRE sends back an image
   (or a LiDAR sweep). The server also supports adding / removing
   actors and the built-in Fixer.
3. For a Python driver loop without a full simulator, `nre` ships a
   thin host-side gRPC client
   (`references/NRE_RenderClient/SKILL.md`,
   `scripts/session_warm_server.sh`, `thin_client.py`,
   `batch_render_rgb`) that keeps one warm `serve-grpc` container
   up for the session and avoids the per-call Docker / Python /
   CUDA cold start.
4. For map-coordinate-to-NuRec coordinate conversion, `nre` has a
   recipe in its `physical-ai-render` reference.
