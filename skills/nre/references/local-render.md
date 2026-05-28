# NRE Local Docker Render Reference

Read this file when you need to render a NuRec USDZ scene on the
local host's GPU via `docker run`. For OSMO, see
[`example-workflows/osmo/pai-nurec.yaml`](example-workflows/osmo/pai-nurec.yaml).

Shared details that are not repeated here:

- NRE image variants, version-specific flags, rig-rotation sign
  convention → [`nre-image-notes.md`](nre-image-notes.md)
- `docker login nvcr.io` / NGC key resolution →
  [`ngc-and-registry.md`](ngc-and-registry.md)
- ffmpeg encoding + required MP4 filenames →
  [`mp4-encoding.md`](mp4-encoding.md)

---

## Environment expectations

- Shell execution on the host.
- `docker` with GPU support (`--gpus all`).
- `apt-get` (or equivalent) available for a one-time ffmpeg install
  (if missing).
- NRE image: a renderer image cached locally. The discovery snippet
  in [`nre-image-notes.md`](nre-image-notes.md) matches any cached
  path of the form `*/nre[-_:]*` (excluding the `nre-tools` sibling,
  which doesn't speak `render`) and picks the highest-versioned one.
  Pick flags by the version of whatever image was discovered.
  Private registry — host must be `docker login`'d to `nvcr.io`; see
  [`ngc-and-registry.md`](ngc-and-registry.md).
- Host mount points (conventional):
  - `INPUT_USDZ_HOST` → `/inputs/usdz`
  - `INPUT_RIG_HOST`  → `/inputs/augmented_rig`
  - `OUTPUT_HOST`     → `/outputs`

`INPUT_USDZ_HOST` should point at a directory tree containing at
least one `.usdz` file. The canonical
`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` layout is
`<root>/<clip-uuid>/<clip-uuid>.usdz` alongside the clip's
`labels.json` and `camera_front_wide_120fov.mp4` reference video;
scope a single invocation to one clip by setting
`INPUT_USDZ_HOST=<root>/<clip-uuid>`.

`INPUT_RIG_HOST` should point at a directory containing
`*_augmented.json`. The bundled example rig JSONs at
[`rig-json/`](rig-json/) can be used as a fallback when no
host-side rig dir is provided:

```bash
INPUT_RIG_HOST="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")/skills/nre/references/rig-json"
```

and pass `augmented_rig.json` explicitly to `--rig-json`
(`*_augmented.json` globs match it, but its prefix is not
session-specific — be explicit). Surface the fallback to the caller;
do not use it silently.

Fail fast if a required input, path, or the image is missing — name
exactly what is missing.

---

## Two render modes

| Mode | When to use | Needs rig JSON? |
|------|-------------|-----------------|
| **Rig offset (simple)** | Target view is a translation/rotation offset from the original rig | No |
| **Custom rig trajectory (full)** | User provides an augmented rig JSON | Yes |

For the rig-rotation sign convention (`(yaw, -roll, -pitch)` in
degrees), see [`nre-image-notes.md`](nre-image-notes.md). If the user
says "rotate 5° yaw", pass `5.0 0.0 0.0` — not `0 0 5`.

Default sequence id when unspecified: `manual-test`.

---

## Per-turn wall-time (measured, A100 40GB)

Reference numbers from a measured pass on A100 40GB hardware:
clips of ~600 poses @ 30 fps (~20 s of recorded driving), single
front-wide camera, 1080p JPEG, augmented rig overlay applied.
Numbers are across two clips run back-to-back in one warm-server
session.

| Path | First-ever turn in session | Subsequent new-scene turn | Repeat turn, same scene |
|---|---|---|---|
| Per-turn CLI fallback (Option 2 with `export-custom-rig-trajectory`) | ~3 min | ~3 min | ~3 min (no warm reuse) |
| Per-turn CLI fallback (Option 2, **pre-baked trajectory cache hit**) | ~2 min | ~2 min | ~2 min (no warm reuse) |
| Warm gRPC + thin client | ~2 min | ~35 s | ~25 s |

The "pre-baked trajectory cache hit" row applies on Option 2 turns
where the active rig IS the bundled `augmented_rig.json` AND the
USDZ matches a `<uuid-prefix>_custom_rig_trajectories.json` shipped
under [`custom-rig-trajectories/`](custom-rig-trajectories/).
On a hit, the `export-custom-rig-trajectory` `docker run` is
skipped — one cold-start tax instead of two — and only the
`render --custom-rig-trajectory …` `docker run` actually executes.
See the Option 2 recipe below.

Where the time goes (warm-thin path has **two** distinct one-time
costs that are easy to conflate):

- The **CLI cold-start tax** is paid once per `docker run`
  invocation: Docker boot + Python imports + CUDA context init +
  USDZ deserialization + GPU model upload, roughly ~50 s/container
  on this clip. Option 2 (rig-file swap) needs **two** invocations
  per turn — `nre export-custom-rig-trajectory` to bake the
  augmented rig into a custom trajectory, then `nre render
  --custom-rig-trajectory <that-json>` to render — so it pays the
  tax twice (~85 s/turn cold-start total, before any pixels are
  rendered). When the **pre-baked trajectory cache** (see
  [`custom-rig-trajectories/`](custom-rig-trajectories/))
  fires on a turn — bundled `augmented_rig.json` + a cached clip
  UUID prefix — the `export-custom-rig-trajectory` `docker run` is
  elided and Option 2 pays the cold-start tax **once**, matching
  Option 1 on cold-start cost. Option 1 (rig offset only,
  `--rig-translation-offset` / `--rig-rotation-offset`) needs one
  container and pays it once. Either way, every turn is fresh —
  there's no warm-reuse on the CLI path.
- The **server-process warmup tax** (~80–90 s) is paid once per
  `serve-grpc` process, on the first render call regardless of
  scene. It's the GPU side of cold-start: torch / CUDA / cuDNN
  context init + any TRT JIT compilation the artifact triggers on
  first use. Every subsequent render in that server's lifetime
  skips this tax entirely, including renders against new scenes.
- The **per-scene JIT-load tax** (~7–10 s, split across two
  phases) is paid the first time the `SceneDownloadInterceptor`
  sees a new `scene_id`. It's split between two RPCs in the
  thin-client flow:
    - The first metadata RPC (`get_available_*`) registers the
      USDZ with the renderer + deserialises it into the server's
      scene cache. Cost surfaces in `results.json.setup_s` (~7–9 s
      on this clip class — vs ~0.05 s when the scene is already
      cached).
    - The first render RPC pays the rest (~2–3 s) as a frame-0
      spike: model upload to GPU. Cost surfaces in
      `results.json.frame_stats_all.max_ms` for frame 0; the
      `frame_stats_excl_frame_0` view drops it.
  Every subsequent render against that same `scene_id`
  short-circuits the whole thing — `setup_s` returns to ~0.05 s
  and frame 0 looks like every other frame. Effectively cheap once
  the scene is warm — orders of magnitude smaller than the process
  warmup.
- The **warm-thin steady-state** (~22–42 ms/frame, ~25–40 s/turn
  on a 600-pose 20 s clip) is essentially the render itself plus a
  small per-pose gRPC round-trip — no per-turn container, Python,
  CUDA, or model-load tax. Rig-file swaps are free here too: the
  thin client overlays per-camera intrinsics on each gRPC request,
  no separate `export-custom-rig-trajectory` step needed.

So the realistic shape of a chat session: the first render of the
session pays both the warmup tax and the scene-load tax (~2 min
total on this clip class); every later new-scene render pays just
the scene-load tax on top of steady-state (~35 s); every repeat
render against an already-loaded scene is pure steady-state (~25 s).

**Per-frame cost varies with scene complexity.** Steady-state
frame time on the warm-thin path is dominated by visible splat
count, not USDZ size on disk. Measured medians on this hardware:

- Highway / open-road clips (sparse near-field, sky+road
  dominant): ~28–32 ms/frame, p95 ~32 ms, very tight spread.
- Urban / construction-zone clips (dense vehicles, signs,
  buildings, traffic): ~40–45 ms/frame, p95 ~50 ms, wider spread
  (~±15 % around median because frame-to-frame splat count swings
  with traffic / intersections).

Multi-camera batched runs (`thin-batch` mode) median around
~95–110 ms/pose for 3 cameras — ~33 ms per camera amortised, ~20 %
cheaper per camera than N sequential single-cam runs because pose
interpolation, gRPC serialisation, and JPEG IO amortise across
cameras in the batched RPC.

**Scaling.** Wall time is roughly linear in `frames × cameras` for
all three columns; multi-camera runs that exercise
`batch_render_rgb` (the thin-client batch mode) scale better than N
sequential single-cam invocations because the server schedules all
cameras in one shared CUDA context per pose. Different GPUs shift
the absolute numbers; the ratios between paths are what's
architectural.

`results.json` always carries the real measured `wall_s` /
`render_s` / `frame_stats_excl_frame_0` for the actual run — those
fields are the source of truth for any specific turn, and the
numbers above are just the rough expectation to set against them.

---

## Option 1: Rig offset (no rig JSON)

Pick `IMAGE` via the discovery snippet below — highest-versioned
cached NRE renderer image (see
[`nre-image-notes.md`](nre-image-notes.md) for the full pattern),
with `NRE_IMAGE` as an override.

`--rig-translation-offset` / `--rig-rotation-offset` work on **both** 26.04
and pre-`26.03` images. For non-rigid rig changes on `26.03+`, switch to
the `export-custom-rig-trajectory` flow (Option 2 /
[`nre-image-notes.md`](nre-image-notes.md)).

`--renderer default` is added explicitly on 26.04+ — this loads the
artifact's natively-trained renderer (the `model.renderer.name` saved
alongside the splat) rather than overriding it. Portable across clips,
and avoids the visible flicker that `--renderer gsplat` can introduce
on artifacts not natively trained for it. Full rationale in
[`nre-image-notes.md`](nre-image-notes.md). On pre-`26.03` images,
**omit** the `--renderer` flag — it does not exist there.

`--image-format jpeg` is the new default — typical end-to-end speedup
over `--image-format png` is ~2-3× on long 1080p sequences. PIL still
encodes on CPU for both, but JPEG encode is ~10× cheaper and the file
is ~6× smaller at 1080p (~125 KB vs ~764 KB), so the per-frame disk
write is much faster too. PNG remains user-overridable.

`--height 1080` is used here in place of `--image-scale 1` (the CLI
requires exactly one of the two). NuRec USDZs in the PAI sample set
ship at 4K (3840×2160), so `--image-scale 1` would render every
frame at native 4K — ~4× the per-frame work and disk write of the
warm-thin path's 1080p output, with no visible benefit in the
side-by-side comparison viewer (which also pre-scales to 1080p for
parity). 1080p also matches the warm-thin client's default so a
CLI-fallback turn drops in cleanly next to a previous warm-thin
render in the same viewer set. Bump to native 4K with
`--image-scale 1` (and drop `--height`) only if a user explicitly
asks for "full-resolution" / "native" output.

```bash
set -euxo pipefail

# Image discovery — NRE_IMAGE env var overrides; otherwise pick the
# highest-versioned cached NRE renderer image, excluding the
# nre-tools sibling. Tag-based version sort: tags without a
# parseable X.Y[.Z] (e.g. ":latest") sort last, so a pinned X.Y.Z
# tag wins over ":latest". Full pattern + flag-selection rules in
# nre-image-notes.md.
IMAGE="${NRE_IMAGE:-}"
if [ -z "$IMAGE" ]; then
    IMAGE=$(docker images --format '{{.Repository}}:{{.Tag}}' \
        | grep -E '/nre[-_:]' | grep -v -E 'tools|<none>' \
        | while read -r line; do
            tag="${line##*:}"
            v=$(printf '%s' "$tag" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
            printf '%s\t%s\n' "${v:-0.0.0}" "$line"
          done \
        | sort -V -r | head -1 | cut -f2)
fi
[ -n "$IMAGE" ] || { echo "No NRE renderer image cached locally; docker pull one from nvcr.io/nvidia/nre/ first." >&2; exit 1; }

INPUT_USDZ_HOST=/host/path/to/usdz
OUTPUT_HOST=/host/path/to/outputs

SEQUENCE_ID=manual-test
CAMERA_ID="${CAMERA_ID:-camera_front_wide_120fov}"
IMAGE_FORMAT="${IMAGE_FORMAT:-jpeg}"
output_dir=/outputs/$SEQUENCE_ID
gt_dir="/outputs/${SEQUENCE_ID}/gt"

command -v ffmpeg >/dev/null || { apt-get update && apt-get install -y ffmpeg; }

# Pick a .usdz under INPUT_USDZ_HOST. Prefer the NuRec canonical layout
# `<uuid>/<uuid>.usdz`; fall back to any *.usdz the recursive search
# finds (covers the historic `<uuid>/usd_out/last.usdz` and any
# user-supplied layout).
found_usdz_path=$(find "$INPUT_USDZ_HOST" -type f -name "*.usdz" -print -quit || true)
[ -n "${found_usdz_path:-}" ] || { echo "No .usdz under $INPUT_USDZ_HOST"; exit 1; }
container_usdz_path="/inputs/usdz${found_usdz_path#$INPUT_USDZ_HOST}"

mkdir -p "$OUTPUT_HOST/$SEQUENCE_ID"

# Ground truth — skip if the GT MP4 is already present
if [ ! -f "$OUTPUT_HOST/$SEQUENCE_ID/gt_${CAMERA_ID}.mp4" ]; then
  mkdir -p "$OUTPUT_HOST/${SEQUENCE_ID}/gt"
  docker run --rm --gpus all --shm-size "64g" \
    -v "$INPUT_USDZ_HOST":/inputs/usdz \
    -v "$OUTPUT_HOST":/outputs \
    "$IMAGE" render \
      --artifact-path="$container_usdz_path" \
      --output-dir="$gt_dir" \
      --renderer default \
      --image-format "$IMAGE_FORMAT" --frame-step 1 --height 1080 \
      --camera-id "$CAMERA_ID"
fi

# Adapted — replace the offset values with whatever the user asked for.
# Translation is (x,y,z) meters, rig frame.
# Rotation is (yaw, -roll, -pitch) degrees — see nre-image-notes.md.
docker run --rm --gpus all --shm-size "64g" \
  -v "$INPUT_USDZ_HOST":/inputs/usdz \
  -v "$OUTPUT_HOST":/outputs \
  "$IMAGE" render \
    --artifact-path="$container_usdz_path" \
    --output-dir="$output_dir" \
    --rig-translation-offset 0.0 0.5 0.0 \
    --rig-rotation-offset 0.0 0.0 5.0 \
    --no-replicate-training-views \
    --renderer default \
    --image-format "$IMAGE_FORMAT" --frame-step 1 --height 1080 \
    --camera-id "$CAMERA_ID"
```

> **Pre-`26.03` fallback**: drop `--renderer default` (the flag does not
> exist on those images). The `--image-format jpeg` default is still a
> clear win — the encoder is the same PIL CPU path on both image
> versions, the speedup comes from JPEG's cheaper encode + smaller file
> regardless.

Then encode MP4s with the host's ffmpeg — see
[`mp4-encoding.md`](mp4-encoding.md). If a downstream comparison
viewer is in use (e.g. `nurec-compare`-style side-by-side), follow
its `generate-manifest.sh` to refresh the manifest after the new
MP4 pair lands; the filename rules in `mp4-encoding.md` are what
those manifest scans look for.

---

## Option 2: Custom rig trajectory (full flow)

Requires `26.03+` (provides `export-custom-rig-trajectory`). Verify the
cached image tag is `26.03` or newer before running; on `26.02` or
earlier this flow fails with `unknown command`. See
[`nre-image-notes.md`](nre-image-notes.md).

`--renderer default` and `--image-format jpeg` defaults are picked up here
for the same reasons documented in Option 1 above.

```bash
set -euxo pipefail

# Image discovery — same pattern as Option 1 (highest-versioned cached
# NRE renderer wins; ":latest" sorts last so a pinned X.Y.Z tag beats
# it). NRE_IMAGE overrides.
IMAGE="${NRE_IMAGE:-}"
if [ -z "$IMAGE" ]; then
    IMAGE=$(docker images --format '{{.Repository}}:{{.Tag}}' \
        | grep -E '/nre[-_:]' | grep -v -E 'tools|<none>' \
        | while read -r line; do
            tag="${line##*:}"
            v=$(printf '%s' "$tag" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
            printf '%s\t%s\n' "${v:-0.0.0}" "$line"
          done \
        | sort -V -r | head -1 | cut -f2)
fi
[ -n "$IMAGE" ] || { echo "No NRE renderer image cached locally; docker pull one from nvcr.io/nvidia/nre/ first." >&2; exit 1; }

INPUT_USDZ_HOST=/host/path/to/usdz
INPUT_RIG_HOST=/host/path/to/augmented_rig
OUTPUT_HOST=/host/path/to/outputs

SEQUENCE_ID=manual-test
CAMERA_ID="${CAMERA_ID:-camera_front_wide_120fov}"
IMAGE_FORMAT="${IMAGE_FORMAT:-jpeg}"
output_dir=/outputs/$SEQUENCE_ID
gt_dir="/outputs/${SEQUENCE_ID}/gt"

command -v ffmpeg >/dev/null || { apt-get update && apt-get install -y ffmpeg; }

found_usdz_path=$(find "$INPUT_USDZ_HOST" -type f -name "*.usdz" -print -quit || true)
[ -n "${found_usdz_path:-}" ] || { echo "No .usdz under $INPUT_USDZ_HOST"; exit 1; }
container_usdz_path="/inputs/usdz${found_usdz_path#$INPUT_USDZ_HOST}"

AUGMENTED_RIG=$(find "$INPUT_RIG_HOST" -name "*_augmented.json" -type f | head -1)
if [ -z "$AUGMENTED_RIG" ]; then
  # Fallback to the bundled example rig that ships with this skill.
  REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
  FALLBACK_RIG_DIR="$REPO_ROOT/skills/nre/references/rig-json"
  if [ -f "$FALLBACK_RIG_DIR/augmented_rig.json" ]; then
    echo "No *_augmented.json under $INPUT_RIG_HOST; falling back to bundled example rig at $FALLBACK_RIG_DIR/augmented_rig.json"
    INPUT_RIG_HOST="$FALLBACK_RIG_DIR"
    AUGMENTED_RIG="$FALLBACK_RIG_DIR/augmented_rig.json"
  else
    echo "No *_augmented.json under $INPUT_RIG_HOST and no bundled fallback at $FALLBACK_RIG_DIR/augmented_rig.json"; exit 1
  fi
fi
container_rig_path="/inputs/augmented_rig/$(basename "$AUGMENTED_RIG")"

mkdir -p "$OUTPUT_HOST/$SEQUENCE_ID"

# Ground truth — skip if already present
if [ ! -f "$OUTPUT_HOST/$SEQUENCE_ID/gt_${CAMERA_ID}.mp4" ]; then
  mkdir -p "$OUTPUT_HOST/${SEQUENCE_ID}/gt"
  docker run --rm --gpus all --shm-size "64g" \
    -v "$INPUT_USDZ_HOST":/inputs/usdz \
    -v "$OUTPUT_HOST":/outputs \
    "$IMAGE" render \
      --artifact-path="$container_usdz_path" \
      --output-dir="$gt_dir" \
      --renderer default \
      --image-format "$IMAGE_FORMAT" --frame-step 1 --height 1080 \
      --camera-id "$CAMERA_ID"
fi

# Map the rig onto the scene trajectory.
#
# Fast path: a pre-baked trajectory may ship in
# skills/nre/references/custom-rig-trajectories/ for
# the bundled augmented_rig.json + canonical sample clips. Filename
# convention: <clip-uuid-prefix>_custom_rig_trajectories.json (8-char
# prefix of the USDZ UUID). Only valid when the active rig IS the
# bundled augmented rig and the USDZ IS one of the cached clips —
# the cache encodes (USDZ × rig). Skip the bake on a hit; saves one
# full docker run (~2-3 min on A100-class hardware).
mkdir -p "$output_dir"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "")"
CACHE_DIR="$REPO_ROOT/skills/nre/references/custom-rig-trajectories"
CLIP_UUID="$(basename "${found_usdz_path%.usdz}")"
CLIP_PREFIX="${CLIP_UUID:0:8}"
CACHED_TRAJ="$CACHE_DIR/${CLIP_PREFIX}_custom_rig_trajectories.json"
USING_BUNDLED_RIG=0
case "$AUGMENTED_RIG" in
  "$REPO_ROOT/skills/nre/references/rig-json/augmented_rig.json")
    USING_BUNDLED_RIG=1
    ;;
esac

if [ -n "$REPO_ROOT" ] && [ "$USING_BUNDLED_RIG" = "1" ] && [ -f "$CACHED_TRAJ" ]; then
  echo "[nre] using pre-baked custom_rig_trajectories.json for clip ${CLIP_PREFIX} — skipping export-custom-rig-trajectory bake"
  cp "$CACHED_TRAJ" "$output_dir/custom_rig_trajectories.json"
else
  docker run --rm --gpus all --shm-size "64g" \
    -v "$INPUT_USDZ_HOST":/inputs/usdz \
    -v "$INPUT_RIG_HOST":/inputs/augmented_rig \
    -v "$OUTPUT_HOST":/outputs \
    "$IMAGE" export-custom-rig-trajectory \
      --artifact-path="$container_usdz_path" \
      --rig-json "$container_rig_path" \
      --output "$output_dir/custom_rig_trajectories.json"
fi

# Render the adapted view using that trajectory
docker run --rm --gpus all --shm-size "64g" \
  -v "$INPUT_USDZ_HOST":/inputs/usdz \
  -v "$OUTPUT_HOST":/outputs \
  "$IMAGE" render \
    --artifact-path="$container_usdz_path" \
    --output-dir="$output_dir" \
    --custom-rig-trajectory "$output_dir/custom_rig_trajectories.json" \
    --no-replicate-training-views \
    --renderer default \
    --image-format "$IMAGE_FORMAT" --frame-step 1 --height 1080 \
    --camera-id "$CAMERA_ID"

# Copy rig JSONs next to the render for provenance
ORIGINAL_RIG=$(find "$INPUT_RIG_HOST" -name "*.json" ! -name "*_augmented.json" -type f | head -1)
[ -n "$ORIGINAL_RIG" ] && cp "$ORIGINAL_RIG" "$OUTPUT_HOST/$SEQUENCE_ID/$(basename "$ORIGINAL_RIG")"
cp "$AUGMENTED_RIG" "$OUTPUT_HOST/$SEQUENCE_ID/$(basename "$AUGMENTED_RIG")"
```

Then encode MP4s with the host's ffmpeg — see
[`mp4-encoding.md`](mp4-encoding.md). If a downstream comparison
viewer is in use (e.g. `nurec-compare`-style side-by-side), follow
its `generate-manifest.sh` to refresh the manifest after the new
MP4 pair lands.

---

## Output reporting (local specifics)

Outputs are written directly to the host under `OUTPUT_HOST`. Report
the host path (e.g.
`$OUTPUT_HOST/manual-test/camera_front_wide_120fov.mp4`) as the final
MP4 location.

## Failure handling (local specifics)

Watch for:

- `docker` missing, not on PATH, or no GPU support configured.
- `docker pull` fails with "unauthorized" — host was never
  `docker login`'d to `nvcr.io`. See
  [`ngc-and-registry.md`](ngc-and-registry.md).
- `apt-get install -y ffmpeg` fails — host has no network or no
  `apt`. Install ffmpeg through the host's native package manager.

State exactly what is missing, include paths/identifiers, and suggest
the smallest next action.
