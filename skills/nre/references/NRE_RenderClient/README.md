---
name: nre-render-client
description: >-
  Use when an agent needs to render frames from a long-running NRE
  (Neural Reconstruction Engine) `serve-grpc` server using a thin
  host-side Python gRPC client — no per-call Docker boot, no per-call
  Python+CUDA cold-start in a client container, and (for multi-camera
  turns) a single `batch_render_rgb` RPC instead of N sequential CLI
  invocations.
  This is the warm-server companion to the per-turn `nre render` CLI
  path: keep one server up for the chat session, hit it from Python
  for each render. Trigger keywords: nre thin client, nre render
  client, warm nre server, nre serve-grpc, nre grpc render, nre
  render-grpc python, render_rgb, batch_render_rgb, low-latency nre
  render, nre warm server demo path, render multiple cameras nre.
metadata:
  author: nvidia
  version: "1.0.0"
---

# NRE Render Client (warm-server gRPC, host-side Python)

A small host-side Python gRPC client (`scripts/thin_client.py`) for
talking to an `nre serve-grpc` server directly, without re-paying the
client-container cold-start (Docker boot + Python interpreter + CUDA
context init) on every render. Companion to (not a replacement for)
the bundled `nre render` and `nre render-grpc` CLIs.

Use this when:

- A render-heavy session renders the same scene multiple times. Boot
  one `serve-grpc` for the session, then fire the thin client per
  turn.
- A multi-camera turn needs to render N cameras as one server-side
  scheduling unit. The bundled `render-grpc` CLI is single-camera, so
  the only way to reach the proto's `batch_render_rgb` RPC is from a
  Python client.
- A turn asks for a **novel rig view** ("shift the rig 0.5m left",
  "yaw the cameras 5 degrees right"). The thin client supports
  `--rig-translation-offset X Y Z` and `--rig-rotation-offset YAW NROLL NPITCH`
  with the same axis convention as `nre render`, so the rendered
  output is directly comparable to the per-turn-CLI baseline.
- A turn asks to **render with a different rig file** (e.g. an
  `augmented_rig.json` that adds bivariate-windshield distortion on
  top of the recorded base rig for the front-wide camera). Pass
  `--rig-file <path>` and the thin client overrides the per-camera
  intrinsics on the gRPC request: F-theta lens
  (`pixeldist_to_angle_poly`, cx/cy, optional linear-cde) and, when
  all four windshield polys are present,
  `bivariate_windshield_model_param` (forward + inverse 2D polys).
  Cameras absent from the rig file fall through to the recorded
  intrinsics, so partial overrides are fine. Combine with
  `--rig-translation-offset` / `--rig-rotation-offset` to render a
  novel viewpoint *through* a different rig.
- A turn asks to **switch to a new scene** ("download clip XYZ and
  render it") without restarting the warm server. Stage the USDZ
  anywhere under the host-side bind-mounted root (whatever path you
  passed to `NRE_GRPC_USDZ_HOST_DIR` when booting the warm server)
  and pass `--scene-url file:///inputs/sdg/<sub-path>.usdz` (or
  `http(s)://...usdz`) plus a unique `--scene-id`; the server's
  `SceneDownloadInterceptor` registers / downloads the scene on the
  first RPC of the run. Subsequent runs against the same scene-id
  pay nothing — the scene is in the server's `--cache-size` LRU.
Do **not** use this for:

- One-off renders where Docker cold-start is irrelevant. Use
  `nre render` (covered by the main `nre` skill and `../local-render.md`).
- Workflows that need rolling-shutter modeling, dynamic obstacle
  edits, or asset edits — the thin client uses instant-shutter framing
  and doesn't model dynamic actors / asset edits. The bundled
  `render-grpc` CLI covers all of that.

---

## Quick start

```bash
# 1. Prerequisites: NRE Docker image cached + scene/clip available on disk.
# Pick the highest-versioned cached NRE renderer (any path matching
# `*/nre[-_:]*`, excluding the nre-tools / nre-isaac siblings).
# Tag-based version sort means a pinned X.Y.Z tag beats :latest. The
# warm-server fast path requires NRE 26.04+; export NRE_IMAGE to force
# a specific tag. (For a script that does discovery + version-check +
# warm-server boot orchestration in one shot, see
# `../../scripts/session_warm_server.sh`.)
#
# The bash regex (rather than `grep -oE | head -1`) is on purpose:
# `grep` exits 1 on no-match, which combines badly with `set -euo
# pipefail` in any caller that copies this snippet into a script.
NRE_IMAGE="${NRE_IMAGE:-$(
    docker images --format '{{.Repository}}:{{.Tag}}' \
        | grep -E '/nre[-_:]' | grep -v -E 'tools|isaac|<none>' \
        | while read -r line; do
            tag="${line##*:}"
            if [[ "$tag" =~ [0-9]+\.[0-9]+(\.[0-9]+)? ]]; then
                v="${BASH_REMATCH[0]}"
            else
                v="0.0.0"
            fi
            printf '%s\t%s\n' "$v" "$line"
          done \
        | sort -V -r | head -1 | cut -f2
)}"

# 2. Extract proto stubs from the image (ONCE per image SHA; idempotent).
bash scripts/setup_protos.sh
# -> ~/.cache/nre-render-client/proto/nre/grpc/protos/{common,sensorsim}_pb2{,_grpc}.py

# 3. Boot a serve-grpc server in the background (one per session).
#    Mount your USDZ data root and let the recursive glob find the
#    .usdz wherever you staged it. The root can carry multiple kinds
#    of staged content (USDZ scenes, raw training features,
#    user-supplied subtrees, etc.); the recursive glob only matches
#    USDZs and ignores the rest.
USDZ_HOST=/path/to/your/usdz/root
docker run -d --name nre-grpc-server --gpus all --shm-size 64g --network host \
    -v "$USDZ_HOST":/inputs/sdg:ro \
    "$NRE_IMAGE" \
    serve-grpc \
    --artifact-glob '/inputs/sdg/**/*.usdz' \
    --renderer default \
    --host 0.0.0.0 --port 8080
# Wait for "Serving on" in `docker logs nre-grpc-server`. The cold
# boot loads CUDA + the model once per session; subsequent renders
# reuse it.

# 4. Render one camera, all trajectory poses, JPEG out (recorded view).
#    NOTE: use a Python that has grpcio/protobuf/numpy/scipy installed.
#    Override the default interpreter via NRE_RENDER_CLIENT_PYTHON if
#    your venv lives somewhere specific (e.g. a launchable's frozen
#    /opt venv).
python3 scripts/thin_client.py \
    --cameras camera_front_wide_120fov \
    --output-dir "$USDZ_HOST/outputs/<run-id>/"

# 5. Render four cameras in one batched RPC per pose (recorded view):
python3 scripts/thin_client.py \
    --cameras camera_cross_left_120fov,camera_cross_right_120fov,camera_front_tele_30fov,camera_rear_right_70fov \
    --mode batch \
    --output-dir "$USDZ_HOST/outputs/<run-id>-multicam/"

# 6. Novel rig view. Shift rig 0.5m along Y and yaw 5 degrees right
#    (axis convention matches `nre render --rig-translation-offset` /
#    `--rig-rotation-offset`):
python3 scripts/thin_client.py \
    --cameras camera_front_wide_120fov \
    --rig-translation-offset 0 0.5 0 \
    --rig-rotation-offset 5 0 0 \
    --output-dir "$USDZ_HOST/outputs/<run-id>-novel/"

# 7. Render a NEW scene that wasn't in `serve-grpc --artifact-glob`
#    at boot, without restarting the warm server. The path is
#    container-side (must be inside a bind mount the server can read,
#    e.g. anywhere under /inputs/sdg/ when the warm server bind-mounts
#    $USDZ_HOST); --scene-id is the cache key + filename, and must
#    match the server-side regex [A-Za-z0-9_-]+.
python3 scripts/thin_client.py \
    --cameras camera_front_wide_120fov \
    --scene-url file:///inputs/sdg/<sub-path-to>/<new-clip>.usdz \
    --scene-id <new-clip> \
    --output-dir "$USDZ_HOST/outputs/<run-id>-new-scene/"

# 8. Render with a different rig file (e.g. add windshield optics on
#    top of the recorded base rig). The rig JSON's per-camera
#    intrinsics overlay the server's recorded values; cameras absent
#    from the file fall through.
python3 scripts/thin_client.py \
    --cameras camera_front_wide_120fov \
    --rig-file <rig-host-dir>/augmented_rig.json \
    --output-dir "$USDZ_HOST/outputs/<run-id>-augmented/"
```

The client writes:

- `<output-dir>/000000.jpeg ...` (single-cam) or `<output-dir>/<cam>/000000.jpeg ...` (batch).
- `<output-dir>/timestamps.json` — per-frame timing array.
- `<output-dir>/results.json` — `{setup_s, render_s, wall_s, frame_stats_excl_frame_0, ...}`.

Encode the JPEGs to MP4 with the host's `ffmpeg` — see
[`../mp4-encoding.md`](../mp4-encoding.md) for the canonical recipe
and filename conventions.

---

## Why this skill exists

The **demo-path latency floor problem.** A naive per-turn `docker
run … nre render` repays the same fixed overhead every call:

- `docker run` container startup (cgroups, namespaces, GPU device
  passthrough)
- Python interpreter startup + heavy imports (torch, NRE modules)
- CUDA context init + cuDNN load
- Model load (USDZ deserialization, checkpoint to GPU, any TRT JIT
  warm-up the artifact triggers on first use)
- Then the actual render

The first four are pure overhead — they don't depend on trajectory
length, camera count, or anything else the caller controls. On a
representative ~20 s NuRec clip that overhead is roughly a minute
per turn on its own, independent of clip length. The architectural
fact this skill leans on is just "everything except 'the actual
render' is cold-start tax". Parent skills that care about absolute
numbers should keep their own measured tables (see e.g.
[`../local-render.md`](../local-render.md)); this skill deliberately
stays clip- and hardware-agnostic.

Standing up one `serve-grpc` for the whole session and hitting it via
the Docker-bundled `render-grpc` cuts the **server-side** cold-start
to once per session, but **the Docker client itself still pays its
own cold-start every call** — `render-grpc` is an `nre`-image-internal
binary, so each invocation is a fresh container with the full Python
import + CUDA context init even though the client never loads a
model.

The thin client bypasses the client-side container entirely:

- Talk to the server's gRPC port from the host's Python directly.
- Per-call wall = server-side render time + gRPC round-trip overhead
  + (one-time-per-process) Python startup. The architectural change
  is that the **client-side Docker + Python + CUDA tax is gone every
  call after the first**. The server-side render time itself is
  unchanged (it's the same `serve-grpc` doing the same work).
- For multi-camera turns, dispatch all cameras in one
  `batch_render_rgb` RPC. Server gets to schedule the work itself
  (single CUDA context, shared scene state), instead of the host
  sequentially booting N Docker clients.

---

## How the thin client works

The gRPC server already exposes everything the client needs — there
is no NRE-side change required:

| Setup RPC (once at start) | Purpose |
|---|---|
| `get_available_scenes` | Pick `scene_id` (or pass `--scene-id`). |
| `get_available_cameras` | Per-camera intrinsics + `rig_to_camera` (misnamed proto field; actually `T_camera_rig`). |
| `get_available_trajectories` | Recorded ego trajectory: `Trajectory.poses` = list of `(timestamp_us, T_rig_world)`. |

| Render RPC (per pose) | Purpose |
|---|---|
| `render_rgb` | One camera, one pose, one image_bytes blob back. Used by `--mode single`. |
| `batch_render_rgb` | N cameras, one pose, list of image_bytes blobs back (one per camera). Each `BatchRGBRenderRequestItem` carries a full per-camera `RGBRenderRequest`, so per-camera intrinsics / poses / formats / qualities can all differ. Used by `--mode batch`. |

The script iterates `Trajectory.poses` verbatim — no client-side
interpolation, no rolling-shutter modeling. For each pose `(t, T_rig_world)`:

1. Compose `T_camera_world = T_rig_world @ rig_offset_se3 @ T_camera_rig`
   (per camera). Composition + axis convention match
   `nre.utils.io.export.render_grpc.py`, so output is bytewise comparable
   to the per-turn-CLI baseline.
2. Build `RGBRenderRequest` (or N items wrapped in `BatchRGBRenderRequest`).
3. Send. Write `image_bytes` to disk. Record per-call wall time.

`rig_offset_se3` is built once per invocation from
`--rig-translation-offset` / `--rig-rotation-offset` via a port of
`nre.utils.geometry.pose_offsets_to_se3` (NCore's `euler_2_so3` is
replaced with scipy's `Rotation`; the val-mode axis-permutation hack
from NRE is replicated client-side so the input tuple semantics —
`(yaw, -roll, -pitch)` degrees — match the CLI exactly). Default
`(0, 0, 0)` for both gives identity, i.e. the recorded view.

The thin client renders at the trajectory's natural sample points. A
sim-driven caller that wants to render at arbitrary timestamps would
need to interpolate poses between recorded samples — that's a small
addition on top of the cached trajectory but not something this skill
implements out of the box.

---

## Loading new scenes at runtime (no warm-server restart)

`nre serve-grpc` ships a `SceneDownloadInterceptor`
(`nre/grpc/serve.py`) that runs on every gRPC request and looks for
two metadata headers:

- `x-nre-scene-url` — `file:///container-side/path.usdz` (registers
  the local file) or `http(s)://...usdz` (downloads via the server's
  cache directory).
- `x-nre-scene-id` — the cache key + filename. Must match the
  server-side regex `^[A-Za-z0-9_-]+$`. Validated client-side too.

If the scene under that id isn't already in the server's backend
cache, the interceptor registers/downloads + loads it synchronously
*before* the actual RPC continues. Subsequent calls with the same id
short-circuit. The server holds up to `--cache-size N` loaded scenes
in an LRU; new scenes evict the least-recently-used when full. The
upstream default is 10, but each loaded USDZ holds gaussian-splat
tensors plus any windshield-model state on the GPU, so parent skills
should pin a value sized to their hardware (e.g.
[`../../scripts/session_warm_server.sh`](../../scripts/session_warm_server.sh)
defaults to `--cache-size 3` for A100-40GB, raisable via
`NRE_GRPC_CACHE_SIZE`).

The thin client exposes this via `--scene-url` + `--scene-id`. When
both are set, every RPC of the run carries the metadata, so:

- The first RPC pays the load cost (USDZ deserialization, model
  upload to GPU). Comparable to a `serve-grpc` cold-start for that
  one scene.
- Every subsequent RPC of the run, and every subsequent
  `thin_client.py` invocation against the same scene-id, is back to
  the warm-render fast path.

Practical usage from a parent skill:

1. User: "render clip XYZ from HuggingFace".
2. Parent skill downloads the USDZ to a path inside the warm
   server's bind mount (anywhere under the host-side root you
   bind-mounted into `/inputs/sdg` is fine).
3. Parent skill invokes `thin_client.py --scene-url
   file:///inputs/sdg/<sub-path>/<clip>.usdz --scene-id <clip>
   --cameras ...`.
4. First RPC of run = USDZ register + load (~serve-grpc cold-start
   for that one artifact). Per-pose renders thereafter run at warm
   speed.
5. Next turn against `<clip>` ("now shift left 0.5m") = pure-warm
   path; parent skill reuses `--scene-url` + `--scene-id` and the
   server's interceptor sees "already loaded", short-circuits.

If the user rotates through more than `--cache-size` distinct
scenes in a session, the server evicts the least-recently-used one
and the next turn against that scene re-pays the load cost. Bump
`--cache-size` on the server boot only if you have GPU memory
headroom AND expect lots of scene churn — the cost is GPU-resident
splat tensors per scene, not CPU/disk.

---

## Proto stubs: extracted from the Docker image, not bundled

The gRPC `*_pb2.py` / `*_pb2_grpc.py` stubs are **extracted at runtime
from the cached NRE Docker image** — never bundled in this skill, and
never recompiled host-side. The image already ships the exact stubs
its server uses; extracting them guarantees binary compatibility and
avoids any toolchain skew between host protobuf and image protobuf.

```bash
bash scripts/setup_protos.sh                    # default image
bash scripts/setup_protos.sh --image <tag>      # override
NRE_IMAGE=<tag> bash scripts/setup_protos.sh    # env override
```

What it does:

1. Resolve the NRE Docker image SHA via `docker image inspect`.
2. Check the cache: if `~/.cache/nre-render-client/proto/.image_sha`
   matches and stubs exist, exit 0 (idempotent — safe to call on
   every boot, every render).
3. If not cached, locate `sensorsim_pb2.py` inside the image (probes
   a few known package paths first, falls back to `find /` for
   robustness), `docker cp` it + the three sibling stubs out to
   `~/.cache/nre-render-client/proto/nre/grpc/protos/`.
4. Touch `__init__.py` at every package level so `import nre.grpc.protos.*`
   resolves cleanly.
5. Run an import sanity check (`Pose`, `RGBRenderRequest`,
   `BatchRGBRenderRequest` all importable) before declaring success.
6. Write the new image SHA to the cache.

`thin_client.py` then resolves the proto stubs in this order:

1. `NRE_RENDER_CLIENT_PROTO_DIR` env var (set this when integrating
   into a parent skill or when running multi-instance against
   different image tags).
2. `~/.cache/nre-render-client/proto/` (the default that
   `setup_protos.sh` writes to).
3. Hard error with instructions to run `setup_protos.sh`.

If the NRE image SHA changes (image rotation, version bump), re-run
`setup_protos.sh` — it'll detect the SHA delta and regenerate.

---

## Dependencies

The thin client needs a Python interpreter with `grpcio`,
`protobuf`, `numpy`, and `scipy` installed. The convention is to
keep these in a dedicated venv rather than the system Python, so
the agent (or any other unprivileged caller) cannot `pip install`
arbitrary packages on the runtime image.

| Dep | Where it comes from |
|---|---|
| `grpcio` >= 1.71, `protobuf` 5.x | host venv (typically `python3 -m venv` + `pip install`) |
| `numpy`, `scipy` | same venv |
| `docker` CLI + GPU runtime | system |
| NRE Docker image (`nre serve-grpc`) | system, pulled from `nvcr.io/nvidia/nre/`; see [`../ngc-and-registry.md`](../ngc-and-registry.md) |

Point both `setup_protos.sh` and `thin_client.py` at that
interpreter via `NRE_RENDER_CLIENT_PYTHON`:

```bash
export NRE_RENDER_CLIENT_PYTHON=/path/to/your/venv/bin/python3
bash scripts/setup_protos.sh
"$NRE_RENDER_CLIENT_PYTHON" scripts/thin_client.py --cameras ...
```

Both scripts fall back to `python3` on PATH if
`NRE_RENDER_CLIENT_PYTHON` is unset and the launchable's frozen
venv at `/opt/launchable-grpc-py/venv/bin/python3` (the historical
default) isn't present — handy on developer workstations.

`setup_protos.sh` doesn't compile anything — it just `docker cp`s
the already-generated `*_pb2.py` files out of the NRE Docker
image — so there's no `grpcio-tools` / `protoc` host dependency to
install or pin.

---

## Limitations (what the thin client deliberately doesn't do)

These are punted to the bundled `render-grpc` CLI (or future
iterations of this skill if a chat user actually needs them):

1. **Rolling shutter modeling.** `start_pose == end_pose` per frame;
   the server gets a `[t, t+1)` instant-shutter window. Fine for the
   demo path; not faithful to the recorded camera physics.
2. **Dynamic obstacle / actor edits, asset edits.** Out of scope.
3. **Lidar rendering.** RGB only. Lidar via `render-grpc --lidar`.
4. **Egocar hood overlay.** Not supported on either path. The
   bundled `render-grpc` CLI has `--egocar-hood-dir` plumbing but
   the thin client deliberately does not invoke it; the standalone
   `nre render` CLI has no hood plumbing at all.
5. **Stale-server detection.** If the server dies between calls,
   the script raises a gRPC error on the next RPC. Caller
   (the parent skill) is responsible for restarting if needed.
6. **Sim-driven timestamps / interpolated poses.** Renders at the
   trajectory's natural sample points. If a parent skill needs
   `get_pose_at_timestamp`-style interpolation between recorded
   samples, port pacsim's helper (it's a few lines on top of the
   cached trajectory). For the typical "novel rig view" use case,
   the rig translation/rotation offsets are sufficient — the
   trajectory itself doesn't need resampling.

---

## Integrating from another skill

A parent skill that wants the warm-server flow should:

1. Run [`scripts/setup_protos.sh`](scripts/setup_protos.sh) once per
   session (idempotent; no-op when the cache matches the current
   image SHA).
2. Boot the warm server via
   [`../../scripts/session_warm_server.sh`](../../scripts/session_warm_server.sh)
   (or its own equivalent `docker run … serve-grpc …` if it needs
   custom boot flags).
3. Invoke [`scripts/thin_client.py`](scripts/thin_client.py) per
   turn through the venv interpreter
   (`$NRE_RENDER_CLIENT_PYTHON`, or `python3` on PATH).

```bash
# Once per session:
NRE_SKILL_DIR=/path/to/nre-skill
bash "$NRE_SKILL_DIR/references/NRE_RenderClient/scripts/setup_protos.sh"

# Boot warm server (once per session). Either use the bundled script:
NRE_GRPC_USDZ_HOST_DIR=/path/to/usdz/root \
    bash "$NRE_SKILL_DIR/scripts/session_warm_server.sh"
# ...or roll your own docker run:
docker run -d --name nre-grpc-server --gpus all --shm-size 64g --network host \
    -v /path/to/usdz/root:/inputs/sdg:ro \
    "$NRE_IMAGE" serve-grpc \
    --artifact-glob '/inputs/sdg/**/*.usdz' \
    --renderer default --host 0.0.0.0 --port 8080
# Poll: docker logs nre-grpc-server | grep -q "Serving on .*:8080"

# Per-turn render. The parent skill is expected to translate user
# intent ("shift left 0.5m, yaw 5deg right") into the X Y Z and
# YAW NROLL NPITCH tuples here; pass 0 0 0 for both to render the
# recorded view.
#
# When the user asks for a NEW scene (HuggingFace clip, S3 USDZ,
# etc.), the parent skill downloads it into a path that is inside
# the warm server's bind mount, then sets $SCENE_URL + $SCENE_ID;
# the warm server loads on first RPC and caches in LRU. For the
# default sample clip (already in --artifact-glob at boot), leave
# $SCENE_URL / $SCENE_ID unset.
#
# When the parent skill wants to render with a non-recorded rig
# (e.g. an augmented_rig with windshield optics on top of the
# recorded base rig), point $RIG_FILE at the rig JSON; the thin
# client overrides the per-camera intrinsics on each request.
# Leave $RIG_FILE unset to render with the rig baked into the USDZ.
"${NRE_RENDER_CLIENT_PYTHON:-python3}" \
    "$NRE_SKILL_DIR/references/NRE_RenderClient/scripts/thin_client.py" \
    --cameras "$CAMERAS_CSV" \
    --rig-translation-offset "$TX" "$TY" "$TZ" \
    --rig-rotation-offset "$RYAW" "$RNROLL" "$RNPITCH" \
    ${RIG_FILE:+--rig-file "$RIG_FILE"} \
    ${SCENE_URL:+--scene-url "$SCENE_URL"} \
    ${SCENE_ID:+--scene-id "$SCENE_ID"} \
    --output-dir "$TURN_OUT_DIR" \
    --image-quality 95 --height 1080
# results.json is at $TURN_OUT_DIR/results.json
# (also records `rig_translation_offset_m`, `rig_rotation_offset_deg`,
# `is_novel_view`, `rig_file`, `rig_file_overrides`, `scene_url` so
# post-hoc debugging / video naming knows what was rendered against
# which scene + rig). `rig_file_overrides` is a per-camera dict:
#   {"<cam>": {"ftheta": true, "windshield": <bool>,
#              "rig_resolution": [<w>, <h>] | null}}
# `windshield: false` when the rig file was missing one or more of
# the four bivariate-windshield polys; `rig_resolution: null` when
# the rig file didn't declare a calibration resolution. Compare
# `rig_resolution` against the recorded camera resolution to see
# whether F-theta polys / cx / cy were rescaled.
# JPEGs at $TURN_OUT_DIR/000000.jpeg ... (single) or $TURN_OUT_DIR/<cam>/... (batch)

# In the parent skill's session teardown:
docker stop nre-grpc-server && docker rm nre-grpc-server
```

The thin client is stateless across invocations (each call opens a
fresh gRPC channel). Boot/teardown is the parent skill's responsibility.

---

## File layout

In this repo (committed) — nested under the `nre` skill as one of
its references:

```
<nre-skill-dir>/references/NRE_RenderClient/
  README.md                         # this file
  scripts/
    thin_client.py                  # the host-side gRPC client
    setup_protos.sh                 # extract gRPC stubs from the NRE Docker image
```

Cached at runtime (per-user, regenerated when the NRE image SHA changes):

```
~/.cache/nre-render-client/
  proto/
    nre/grpc/protos/
      common_pb2.py
      common_pb2_grpc.py
      sensorsim_pb2.py
      sensorsim_pb2_grpc.py
    .image_sha                       # the image SHA these stubs were extracted from
```

Some hosting environments (e.g. launchables) pre-bake a dedicated
`grpcio` + `protobuf` + `numpy` + `scipy` venv at a canonical path
like `/opt/launchable-grpc-py/venv/bin/python3`. If yours does,
point `NRE_RENDER_CLIENT_PYTHON` at that interpreter so the
unprivileged caller doesn't have to `pip install` on a read-only
runtime image. Otherwise `python3 -m venv` + `pip install grpcio
protobuf numpy scipy` works fine.

---

## Related

- [`../local-render.md`](../local-render.md) — per-turn `nre render`
  CLI flow. Use it when the thin client doesn't expose what you
  need (custom trajectory, lidar, rolling shutter, in-container
  video encode, actor edits).
- [`../../scripts/session_warm_server.sh`](../../scripts/session_warm_server.sh)
  + [`../../scripts/session_teardown.sh`](../../scripts/session_teardown.sh)
  — bundled session-scoped warm-server boot / teardown that wraps
  the `docker run … serve-grpc …` recipe above with image
  discovery, version checks, and idempotency.
- The canonical [`nre`
  SKILL.md](https://github.com/NVIDIA/nurec-skills/blob/main/.agents/skills/nre/SKILL.md)
  in the `NVIDIA/nurec-skills` repo — the bundled `nre render` and
  `nre serve-grpc` / `nre render-grpc` CLI reference (covers both
  26.02 legacy and 26.04+ current Docker images). Use the CLIs for
  one-off renders, scenes that need rolling-shutter or actor edits,
  or any non-RGB workload. See
  [`../nurec-skill-catalog.md`](../nurec-skill-catalog.md) for the
  full routing block.
- The canonical [`nurec-fixer`
  SKILL.md](https://github.com/NVIDIA/nurec-skills/blob/main/.agents/skills/nurec-fixer/SKILL.md)
  — Difix / Harmonizer post-processor. The thin client returns raw
  `image_bytes` blobs; if the parent skill wants harmonized frames,
  it runs the fixer separately on the JPEGs.
