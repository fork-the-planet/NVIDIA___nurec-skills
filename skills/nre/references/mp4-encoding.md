# MP4 Encoding (Shared)

Read this file after an `nre render` produces per-frame images
(JPEG by default on `26.04+`, PNG if the caller overrode
`--image-format png`, or anywhere the legacy pre-`26.03` path was
taken) and you need MP4s — regardless of backend (local docker or
OSMO). Side-by-side comparison viewers (e.g. `nurec-compare`)
require precise filenames, so follow the naming rules below
exactly.

---

## Two ways to produce MP4s

1. **Host-side `ffmpeg`** — encode from the per-frame images (JPEG by
   default, PNG if the user overrode the format) after the render step
   exits. Works on any NRE image, predictable memory, streams frames
   from disk. **Default.** Used by both the local flow and the canonical
   OSMO workflow.
2. **In-container `render --export-video`** — on `release/26.04` and
   newer, `render` can emit one H.264 MP4 per camera alongside the
   frames (`--export-video --video-fps 30 --video-crf 20`). It loads
   all frames per camera into memory during encoding, so only use it
   for short / low-resolution clips. Skip it entirely on older images.

Prefer (1) unless the user explicitly asks for `--export-video`.

---

## File naming (REQUIRED — viewer depends on it)

Side-by-side comparison viewers (e.g. `nurec-compare`) require
exact names in the set directory:

- `gt_camera_<id>.mp4` — ground truth
- `camera_<id>.mp4` — adapted render

**Never** add suffixes like `_adapted`, `_render`, or `_new` — any
deviation silently breaks the viewer's `generate-manifest.sh` scan
and the viewer dropdown stays empty. When using `--export-video`,
NRE writes `<output-dir>/<camera-id>.mp4`; rename it to the
required pattern **before** regenerating the viewer manifest.

---

## Host-side ffmpeg

```bash
CAMERA_ID=camera_front_wide_120fov
SEQ_DIR=/path/to/sequence/output
FRAME_DIR=$(find "$SEQ_DIR" -type d -name "$CAMERA_ID" | head -1)

# Auto-detect the per-frame extension produced by the render
# (jpeg by default; png if the user overrode --image-format png).
FRAME_EXT=$(ls "$FRAME_DIR" 2>/dev/null \
  | grep -oE '\.(jpeg|jpg|png)$' \
  | sort -u | head -1 | tr -d '.')
[ -n "$FRAME_EXT" ] || { echo "No jpeg/jpg/png frames found in $FRAME_DIR"; exit 1; }

ffmpeg -y -framerate 30 -pattern_type glob -i "${FRAME_DIR}/*.${FRAME_EXT}" \
  -c:v libx264 -pix_fmt yuv420p \
  "$SEQ_DIR/${CAMERA_ID}.mp4"
```

Defaults (do not change unless the user asks):

- framerate: 30 fps
- codec: `libx264`
- pixel format: `yuv420p`
- per-frame extension: auto-detected (`jpeg` matches the new render default)

If ffmpeg is not installed:

```bash
apt-get update && apt-get install -y ffmpeg
```

---

## Where this runs per backend

- **Local backend:** on the host after `docker run` exits, against the
  mounted `OUTPUT_HOST` directory.
- **OSMO backend:** on the host after `osmo dataset download
  <output-dataset> <local_path>` completes, against the downloaded
  directory.

After encoding, regenerate any viewer / dashboard manifest that
consumes these MP4s. Filenames must follow the `gt_camera_<id>.mp4` /
`camera_<id>.mp4` convention above for the side-by-side viewer used
by `nurec-compare`-style tools to pick up the new pair.
