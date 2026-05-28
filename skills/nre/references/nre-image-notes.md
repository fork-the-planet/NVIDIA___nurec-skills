# NRE Image Notes (Shared)

Read this file when you need to pick a `render` / `export-custom-rig-
trajectory` invocation style for a specific NRE container image, or when
you need to understand the rig-rotation sign convention. Shared between
local docker and OSMO workflows.

---

## Default

The canonical default is a 26.04+ NRE image, run with `--renderer default`
and `--image-format jpeg`. Every code path in this skill assumes this
unless explicitly noted otherwise. Pre-`26.03` images are documented
below as a legacy fallback for instances that still have an old tag
cached.

`--image-format jpeg|png` is **universal** — every NRE image family
in the matrix below (including the pre-`26.03` legacy fallback)
accepts `--image-format=jpeg`. Don't fall back to `png` just because
the image is older. The version-gated flags are `--renderer` (26.04+)
and `export-custom-rig-trajectory` (26.03+).

### Image discovery

`NRE_IMAGE` env var overrides everything. Otherwise pick the
highest-versioned cached NRE renderer image. The grep matches any
locally-cached path of the form `*/nre[-_:]*` (covers floating-tag
channels like `nre-ga:latest` and `nre:latest`, version-pinned tags
like `nre:<X.Y>`, and alternate-org channels like `nre_run:<X.Y.Z-sha>`)
while excluding the `nre-tools` sibling, which doesn't speak
`render`. The version sort extracts the first `X.Y[.Z]` from each
tag and picks the highest; tags without a parseable version (e.g.
`:latest`) sort last so a pinned `:X.Y.Z` tag wins over a floating
`:latest`:

```bash
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
```

When the user explicitly wants a different cached image — for
example to test the legacy fallback against an older version — pass
it via `NRE_IMAGE=<tag>` to short-circuit discovery. If the chosen
tag is `<26.04`, the warm-server boot will exit 3 on its version
check and the agent's CLI fallback handles it per the table below.

---

## Image variants — quick decision table

Keyed by NRE version (run `<image> --version` to confirm), regardless
of which canonical path the image was pulled from:

| NRE version | Entry style | Rig-change mechanism | `--renderer` flag |
|-------------|-------------|----------------------|-------------------|
| `26.04+` (canonical default; warm-server fast path requires this) | default entrypoint (`command: ["render"]`) | `--rig-translation-offset` / `--rig-rotation-offset` (rigid) **or** `export-custom-rig-trajectory` + `--custom-rig-trajectory` (arbitrary) | available — use `--renderer default` (loads the artifact's own renderer) |
| `26.03` | default entrypoint | same as above | available |
| `<26.03` (legacy fallback) | default entrypoint | `--rig-translation-offset` + `--rig-rotation-offset` on `render` only (no `export-custom-rig-trajectory`) | not available — omit |

---

## Official tagged images (`26.04` default — `26.03+` family)

- Pass the subcommand directly as `command:` (OSMO) or as arguments to
  `docker run` (local). For OSMO:
  ```yaml
  image: nvcr.io/nvidia/nre/nre:26.04
  command: ["render"]
  args:
    - --artifact-path={{input:0}}/...usdz
    - --output-dir={{output}}
    - --custom-rig-trajectory={{output}}/custom_rig_trajectories.json
    - --no-replicate-training-views
    - --renderer=default
    - --image-format=jpeg
    - --frame-step=1
    - --height=1080
    - --camera-id=camera_front_wide_120fov
  ```
- For a simple rigid-offset render (no rig JSON), substitute
  `--rig-translation-offset=<tx>,<ty>,<tz>` + `--rig-rotation-offset=<yaw>,<-roll>,<-pitch>`
  for the `--custom-rig-trajectory=...` line. Both flag styles are
  supported on `26.04`; pick `export-custom-rig-trajectory` only if you
  need a non-rigid rig change. See [`local-render.md`](local-render.md)
  Option 1 vs Option 2.
- If a preparatory step (`export-custom-rig-trajectory`) is needed,
  make it a separate task with the same image and a different
  `command:` — do not wrap it in a `bash -c` script.

### Why `--renderer default` is the default on `26.04+`

`--renderer default` loads each artifact's natively-trained renderer —
the value of `model.renderer.name` saved alongside the splat. That makes
the skill portable across clips: an artifact trained as `3dgut-gsplat`
gets the GSplatRenderer; one trained as `3dgut-nrend` gets the PyTorch
forward pass through `3dgut-nrend`. Forcing `--renderer gsplat` on a
clip whose native renderer is something else still produces a valid
render (gsplat can rasterize any 3dgut artifact), but introduces visible
frame-to-frame variation (= flicker) and noticeably larger MP4s at the
same x264 CRF.

Cost: on `nrend`-trained artifacts the first render call may pay a
sizeable one-time TRT JIT compile (tens of seconds, sometimes longer
for larger models). For warm-server workflows (gRPC `serve-grpc` once
per session) this amortises immediately; for one-shot CLI renders the
cold-start is real but the quality win is worth it.

`--renderer nrend` is a separate code path (`enable_nrend=True`, loads
via `NRendWrapper` C++/CUDA) which can fail on artifacts whose
spherical-harmonics tensor shape doesn't match the wrapper's compiled
SH degree (typical error: `NRESHGaussianModel : input albedo data has
wrong size [...]`). Don't use `--renderer nrend` unless you've confirmed
the artifact is compatible with `NRendWrapper` — `default` is the right
knob for "let the artifact decide".

### Why `--image-format jpeg` is the default

Both `png` and `jpeg` go through `PIL.Image.save()` on CPU in the
standalone `render` CLI (the gRPC path additionally uses nvJPEG GPU
encode for `jpeg` only — see `nre/grpc/serve.py::encode_image`). Even on
the CPU PIL path, JPEG is ~10× cheaper to encode than PNG and produces
files ~6× smaller at 1080p (~125 KB vs ~764 KB), which makes per-frame
disk write much faster. Typical end-to-end speedup over `png` on long
1080p sequences is ~2-3×.

## Pre-`26.03` images (legacy fallback)

`export-custom-rig-trajectory` **does not exist** on these. If you try
to invoke it, the workflow fails with "unknown command". Instead, pass
the rig offset directly to `render`. `--renderer` does not exist on
these either — omit the flag.

```yaml
command: ["render"]
args:
  - --artifact-path=...
  - --output-dir={{output}}
  - --rig-translation-offset=<tx>,<ty>,<tz>       # meters; rig frame (x-forward, y-left, z-up)
  - --rig-rotation-offset=<yaw>,<-roll>,<-pitch>  # degrees; see rotation convention below
  - --no-replicate-training-views
  - --image-format=jpeg
  - --camera-id=camera_front_wide_120fov
```

Derive the offset from the user's augmented rig relative to the
original, or accept `--set` variables. Pre-`26.03` is a fallback only —
prefer the 26.04 defaults documented above for every new run.

## `release/26.04` additions

- `--renderer default|gsplat|nrend` — replaces the deprecated
  `--enable-nrend`. `default` uses the artifact's trained renderer,
  `gsplat` forces the GSplat rasterizer, `nrend` uses the fast
  C++/CUDA `NRendWrapper`. Legacy `--enable-nrend` is still accepted
  but hidden; don't use it unless the image is older than `26.03`.
- `--export-video` + `--video-fps` + `--video-crf` — in-container
  MP4 encoding (see [`mp4-encoding.md`](mp4-encoding.md) for why
  host-side ffmpeg is usually preferred).
- `--frame-naming contiguous-output-index|frame-end-timestamp` and
  `--rolling-shutter-duration <us>` — benchmark-friendly output.
- `serve-grpc` gains `--health-port` and the same `--renderer` flag.
- `export-custom-rig-trajectory` gains `--reference-rig-trajectory`
  as an alternative to `--artifact-path` (mutually exclusive; one is
  required).

---

## Rig rotation sign convention

`render --rig-rotation-offset` takes three degrees in the order
**`(yaw, -roll, -pitch)`** (rig frame, x-forward, y-left, z-up). This
is what the NRE CLI docstring says literally, reflecting an axis
permutation inside `pose_offsets_to_se3()` that has not been cleaned
up yet.

- Single-axis offsets are the safest entry point:
  - pure yaw of 5° → `5.0 0.0 0.0`
  - pure pitch down of 5° → `0.0 0.0 5.0` (because input is `-pitch`)
- For anything multi-axis, prefer an augmented rig JSON and
  `export-custom-rig-trajectory` — treat multi-axis offsets as
  experimental.
- When the user hands you a `(roll, pitch, yaw)` triplet, remap it
  to `(yaw, -roll, -pitch)` before passing to `render`.

`--rig-translation-offset X Y Z` is in meters, rig frame
(x-forward, y-left, z-up) — no sign tricks.

---

## Common CLI gotchas

- `Exactly one of --image-scale or --height must be specified` — pass
  only one.
- `--replicate-training-views is set but --rig-translation-offset …`
  — remove the offset or add `--no-replicate-training-views`.
- `unknown option --renderer` / `--export-video` — image is older than
  `release/26.04`. Drop the flag or bump the image tag to a 26.04+ build.
- `NRESHGaussianModel : input albedo data has wrong size […]` /
  `GUTRenderer : cannot get cuda resource on the device 0` — the
  `NRendWrapper` C++/CUDA path is failing because the artifact's
  spherical-harmonics tensor shape doesn't match the wrapper's compiled
  SH degree. This only triggers if `--renderer nrend` (or, on legacy
  images, `--enable-nrend`) was passed explicitly. On `26.04+` switch to
  `--renderer default` (loads the artifact's native renderer through the
  PyTorch forward pass instead of `NRendWrapper`); falling back to
  `--renderer gsplat` is also valid as long as you accept the flicker
  trade-off documented in "Why `--renderer default`" above. On pre-`26.03`
  images the only fallback is `--no-enable-nrend` since `--renderer`
  doesn't exist there.
