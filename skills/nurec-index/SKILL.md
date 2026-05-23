---
name: nurec-index
description: >-
  Use when starting any NVIDIA NuRec task. Helps the agent pick the
  right sibling skill — `nre` (train and render reconstructions),
  `ncore-data-conversion` (prepare input data), `asset-harvester`
  (extract 3D objects from driving clips), `nurec-fixer` (clean up
  rendered frames), or `physical-ai-datasets` (find a dataset on
  Hugging Face). Read this skill first whenever a user mentions
  NuRec, NRE, neural reconstruction, USDZ rendering, 3DGUT, gaussian
  splatting from driving logs, sensor sim, or asks "which skill
  should I use for X?". Also points to the cross-skill teardown
  inventory. Trigger keywords: nurec, nurec index, nurec router,
  neural reconstruction engine, NRE, 3DGUT, 3DGRT, USDZ, sensorsim,
  novel view synthesis, PhysicalAI-Autonomous-Vehicles-NuRec, NuRec
  end-to-end, NuRec pipeline, NuRec workflow, where do I start with
  NuRec, getting started with NuRec, nurec teardown, cleanup.
version: "0.2.0"
author: NVIDIA NRS
tags:
  - nurec
  - index
  - router
  - table-of-contents
tools:
  - Read
license: CC-BY-4.0
metadata:
  canonical_repo: https://github.com/NVIDIA/nurec-skills
  canonical_skills_dir: .agents/skills/
  agentskills_io_compatible: true
---

# NuRec Skills Index

This is the starting point. Read it first, decide which sibling
skill answers the user's question, then open that skill. Every link
below uses the skill's **name** (e.g. `nre`) instead of a file path,
so this index works the same when it lives in a different repo from
the skills it points to. If you ever need to fetch a sibling skill
that isn't on the local disk, jump to [Find a skill that isn't
already on disk](#find-a-skill-that-isnt-already-on-disk).

> **Thin-local-skill policy.** The repo this index ships in
> (<https://github.com/NVIDIA/nurec-skills>) treats these skills as
> coordination layers. The canonical sources live upstream — NCore
> (<https://github.com/NVIDIA/ncore>), the NRE NGC containers
> (`nvcr.io/nvidia/nre/nre`, `nvcr.io/nvidia/nre/nre-tools`), Asset
> Harvester (<https://github.com/NVIDIA/asset-harvester>), the unified
> Fixer & Harmonizer
> (<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nre/models/nurec-fixer>),
> and the `PhysicalAI-*` Hugging Face datasets under
> <https://huggingface.co/nvidia>. Each
> sibling skill links its upstream in its frontmatter
> `metadata.upstream` field.

## What is NuRec?

**NuRec** (NVIDIA Omniverse Neural Reconstruction) takes a recording
from cameras and LiDAR — usually from a self-driving car or a robot —
and turns it into a 3D scene that you can re-render from any angle.

A few names you'll see a lot:

- **NRE** — "Neural Reconstruction Engine", the program that does
  the actual training and rendering. NuRec is the product name; NRE
  is the engine inside it.
- **USDZ** — the file format the trained scene is saved in. It's a
  zip archive that Omniverse, Isaac Sim, and CARLA know how to open.
- **NCore V4** — the input format. Raw recordings have to be
  converted into NCore V4 before NRE can use them.
- **3DGUT / 3DGRT** — two flavours of 3D Gaussian Splatting that NRE
  uses internally. Most users never need to pick between them; the
  default Hydra recipe handles it.

A typical NuRec project has three stages:

1. **Get the input** — either convert your own recording into
   NCore V4, or download a pre-converted dataset from Hugging Face.
2. **Train the reconstruction** — feed NCore V4 to NRE; out comes a
   USDZ file.
3. **Render new views** — point NRE at the USDZ to render images,
   videos, or LiDAR sweeps from any camera angle.

Some projects skip step 2 entirely by downloading a USDZ that NVIDIA
has already trained.

## Pick a skill

Find your goal in the left column, then open the skill on the
right. Arrows mean "do these in order".

| I want to… | Read this skill |
|------------|-----------------|
| Find or download a NuRec dataset NVIDIA has published | `physical-ai-datasets` |
| Convert my own camera / LiDAR / radar / depth / stereo recording into NCore V4 | `ncore-data-conversion` |
| Write a new converter for a sensor setup that isn't already supported (drone, RGB-D camera, ROS 2 bag, COLMAP, …) | `ncore-data-conversion` |
| Train a 3D reconstruction from an NCore clip | `ncore-data-conversion` → `nre` |
| Generate the extra inputs NRE needs (segmentation masks, depth, ego mask, …) | `nre` (uses the `nre-tools` container) |
| Render a USDZ along the **original** camera positions | `nre` |
| Render at full resolution / highest quality | `nre` (see "Quality presets" inside that skill) |
| Render along a **shifted** trajectory (e.g. car moved 3 m to the left) | `nre` |
| Render through a server so a simulator (CARLA, Isaac Sim, your own) can ask for frames | `nre` (`serve-grpc`) |
| Render LiDAR sweeps (point clouds) from a USDZ | `nre` (`render-grpc --lidar`) |
| Skip training and just render a NuRec scene that NVIDIA already built (driving logs) | `physical-ai-datasets` → `nre` |
| Skip training and use a pre-built indoor robotics scene | `physical-ai-datasets` → `nre` (then Isaac Sim 5.1) |
| Extract individual 3D objects (cars, pedestrians) from a driving clip | `asset-harvester` |
| Add, remove, or replace cars / pedestrians in a NuRec scene | `asset-harvester` → `nre` |
| Clean up rendered frames (remove ghosting, floaters, flickering) | `nurec-fixer`, **or** turn on `--enable-difix` inside `nre` |
| Export the scene as a PLY, mesh, depth maps, ego mask, etc. | `nre` |
| Upgrade an old USDZ so newer NRE versions load it faster | `nre` (`upgrade-artifact`) |
| Open a USDZ or PLY in a browser viewer | `nre` (`viewer` / `ply_viewer`) |
| Measure rendering quality (PSNR, SSIM, LPIPS) against ground-truth frames | `nre` (`eval-rendering-metrics`) |
| Benchmark different reconstruction methods on the same scenes | `physical-ai-datasets` (`PhysicalAI-NuRec-PPISP`) → `nre` |
| Train on multiple GPUs or on SLURM | `nre` (Workflow D) |

## Common workflows

Each workflow lists the skills to read in order, with a one-line
summary of what to do inside each one. Open the named skill for the
full recipe — don't try to follow the steps from this page alone.

### A. Make a NuRec scene from your own recording

Use this when the user has a fresh sensor log and wants a renderable
3D scene at the end.

1. `ncore-data-conversion` — convert the recording into NCore V4.
   The skill has a built-in converter for common formats (PAI,
   Waymo, NuScenes, PandaSet, COLMAP, ScanNet++); for anything else,
   it walks you through writing a new converter.
2. `nre` — generate the extra inputs (depth, segmentation, ego
   mask), then train and validate. Out comes a USDZ. Render it
   locally, or hand it to a simulator over the gRPC API.

### B. Use a NuRec scene NVIDIA has already trained

Use this when the user just wants to see NuRec working without
training anything themselves.

1. `physical-ai-datasets` — accept the gated license on Hugging
   Face, then download **one** scene (~1.5–2 GB) from
   `PhysicalAI-Autonomous-Vehicles-NuRec`. The full dataset is
   ~1.5 TB, so don't pull all of it.
2. `nre` — render the USDZ. The "highest quality" preset in that
   skill renders at the original resolution along the original
   camera positions; you can also ask for new camera positions via
   the gRPC server.

### C. Use NuRec for indoor robot simulation

1. `physical-ai-datasets` — download `PhysicalAI-Robotics-NuRec`
   (62.9 GB of indoor scenes: cafés, offices, hand-held captures).
2. `nre` — optional: re-train the scene if you want to tweak it,
   or just open it in the viewer to inspect.
3. Hand the USDZ to **Isaac Sim 5.1** for AMR (autonomous mobile
   robot) simulation. There's no skill for this step in this index —
   use the Isaac Sim docs directly.

### D. Add, remove, or replace 3D objects in a scene

1. `ncore-data-conversion` — make sure you still have the original
   NCore clip on disk; Asset Harvester needs it to crop the object
   views.
2. `asset-harvester` — point it at the object IDs you care about.
   For each one, it produces a `.ply` file (the 3D Gaussian model)
   plus a `metadata.yaml` (size, position, label).
3. `nre` — package those `.ply` files back into the USDZ and edit
   the scene with `serve-grpc --enable-editing-actors` plus
   `render-grpc --edit-assets`. The skill has a JSON schema for the
   add / remove / replace operations.

### E. Clean up rendered frames

NuRec sometimes leaves visible artifacts (floating dots, ghosting,
flickering between frames). Two ways to fix this — pick one:

- **Quick path** — turn on `--enable-difix` when you start the gRPC
  server in `nre`. NRE has a small "Fixer" model built in and runs
  it on every frame as it renders. Default for most users.
- **Standalone path** — render frames first with `nre`, then run
  `nurec-fixer` on the folder of frames. Use this when you want to
  try different Fixer model variants, or fix frames that were
  rendered earlier without re-running NRE.

### F. Benchmark reconstruction quality

1. `physical-ai-datasets` — download `PhysicalAI-NuRec-PPISP`
   (15 GB of outdoor scenes shot at three exposure levels for fair
   comparisons).
2. `ncore-data-conversion` — only needed if you want to re-build
   the NCore shards. The dataset already ships with both COLMAP and
   NCore V4 versions, so usually you can skip this.
3. `nre` — train, then run `eval-rendering-metrics` against the
   ground-truth frames the dataset includes.

### G. Connect NuRec to a simulator

CARLA, Isaac Sim, AlpaSim, or your own simulator can ask NRE for
frames over a network API.

1. `physical-ai-datasets` — pick a USDZ if you don't have one.
2. `nre` — start the server with `serve-grpc`. The simulator sends
   it a camera position and a timestamp; NRE sends back an image (or
   a LiDAR sweep). The server also supports adding / removing
   actors and the built-in Fixer.
3. If you're writing a new client and need to convert between map
   coordinates and NuRec's coordinate system, `nre` has a recipe
   for it in its `physical-ai-render` reference.

## The skills in this folder

Five sibling skills cover everything above. Open them by their
**Name** — that's the canonical identifier. The **Folder** column is
just where the skill lives in this repo if you've cloned it locally.

| Name | Folder | What it does | When to read it |
|------|--------|--------------|-----------------|
| `physical-ai-datasets` | `physical-ai-datasets/` | Catalog and download recipes for every NVIDIA Physical AI dataset on Hugging Face (driving, robotics, manipulation, NuRec scenes, benchmarks). | First step of almost every NuRec workflow — "I need data". |
| `ncore-data-conversion` | `ncore/` | Converts any sensor recording into NCore V4, the format NRE needs. Also covers writing a new converter. | NRE refuses to load anything that isn't valid NCore V4, so this comes before training. |
| `nre` | `nre/` | The Neural Reconstruction Engine itself. Trains reconstructions, renders frames, exports meshes / point clouds / depth, edits actors, runs the gRPC server, browses the result in a viewer, evaluates quality. | The "actually do something with NuRec" skill. Read after the data is ready. |
| `asset-harvester` | `asset-harvester/` | Open-source pipeline (Apache-2.0) that extracts individual 3D objects from sparse views in a driving clip and saves them as `.ply` Gaussian splats with metadata. | Only needed for the "add / remove / replace objects" workflow (D above). |
| `nurec-fixer` | `nurec-fixer/` | Standalone "Fixer" model (built on Difix3D+) that cleans up rendered frames in a separate step from rendering. | Only needed when the built-in `--enable-difix` inside `nre` isn't enough or when you're fixing frames that were rendered earlier. |

## Easy mix-ups

These pairs sound similar but are different things. When in doubt,
come back here.

- **NuRec vs NRE.** NuRec is the product name; NRE is the engine
  inside it. Both map to the same skill: `nre`.
- **NRE's built-in Fixer vs the standalone Fixer.** Both are based
  on the **Difix3D+** model family (paper:
  <https://arxiv.org/abs/2503.01774>), but they're packaged
  differently. `nre`'s `--enable-difix` flag runs a built-in Fixer
  variant (`cosmos_difix` by default since 25.09, or `sd_difix` for
  the legacy Stable-Diffusion build) inside the NRE container as
  it renders. The `nurec-fixer` skill is the standalone unified
  Fixer & Harmonizer release on NGC
  (`nvidia/nre/nurec-fixer:cosmos_3dgut_fixer_harmonizer`) that runs
  inside the public `pytorch:24.10-py3` base image on a folder of
  already-rendered frames. Default to the built-in one; reach for
  `nurec-fixer` when you want to swap model variants or fix frames
  that were rendered earlier.
- **`ncore-data-conversion` vs `nre`.** They run **in order**, never
  as alternatives. `ncore-data-conversion` produces the input
  format; `nre` reads it.
- **`asset-harvester` vs `nre`'s `export-external-assets`.** Asset
  Harvester **produces** the per-object `.ply` files. `nre`'s
  `export-external-assets` **packages** them into a USDZ. Always
  Asset Harvester first.
- **Cosmos-Drive-Dreams vs PhysicalAI-Autonomous-Vehicles-NuRec.**
  Both are AV datasets on Hugging Face, both managed by
  `physical-ai-datasets`, but they're different things.
  Cosmos-Drive-Dreams is **synthetic** weather-augmented video
  (CC-BY-4.0). The NuRec dataset is **real** driving scenes turned
  into renderable USDZs (gated AV License).

## Cross-skill teardown

A complete NuRec workflow can leave **150 GB+ on disk** between
container images, model weights, code clones, conda envs, and
output directories. Each sibling skill has its own dedicated
"Teardown" section — use them in this order when you no longer
need the workflow:

| Sibling skill | What to read | Approximate footprint |
|---------------|--------------|------------------------|
| `nre` | [Teardown](nre/SKILL.md#teardown) + [`references/teardown.md`](nre/_versions/release_26.04/85ba2e2/references/teardown.md) | ~120 GB images + caches + per-run outputs |
| `nurec-fixer` | [Teardown](nurec-fixer/SKILL.md#teardown) + [`references/teardown.md`](nurec-fixer/_versions/main/617a990/references/teardown.md) | ~30 GB image + harmonizer artifact + outputs |
| `asset-harvester` | [Teardown](asset-harvester/SKILL.md#teardown) | ~30 GB conda envs + checkpoints + outputs |
| `ncore-data-conversion` | NCore shards live under `<dataset_dir>/`; delete after you're done with NRE training | clip-dependent |
| `physical-ai-datasets` | HF caches live under `${HF_HOME:-$HOME/.cache/huggingface}/hub/`; remove per-dataset directories | dataset-dependent |

Two practical tips that apply across every container-based skill in
this set:

1. **Pin `-u $(id -u):$(id -g)` on every `docker run`** so outputs
   land owned by you, not by `root`. Every documented `docker run`
   command in `nre` and `nurec-fixer` already does this. If outputs
   end up `root`-owned anyway, recover with
   `sudo chown -R "$(id -u):$(id -g)" <output_dir>` before deleting.
2. **Do not revoke `NGC_API_KEY` / `HF_TOKEN` as part of teardown**
   unless you have reason to believe they were leaked — they are
   per-user and shared across every NVIDIA workflow on the host.

## Secrets handling across every sibling skill

Every sibling skill on this index includes a `Verifying secrets
safely` block in its Prerequisites section. **Always verify
prerequisites by running `scripts/validate_setup.py` (where it
exists) or, for skills without one (`ncore-data-conversion`,
`physical-ai-datasets`, this index), use `hf auth whoami` or a
length-only shell check. Never write ad-hoc bash that interpolates
secret values.**

In particular, do not use the bash anti-pattern

```bash
# BAD — leaks the secret to the terminal when the variable is set
echo "HF_TOKEN: ${HF_TOKEN:+yes}${HF_TOKEN:-no}"
```

— this prints `yes<token-value>` because `${VAR:-no}` only falls back
to "no" when the variable is empty. If you suspect a token was
echoed, rotate it (`huggingface.co/settings/tokens`,
`org.ngc.nvidia.com/setup/api-key`) before continuing.

## Find a skill that isn't already on disk

If you're reading this index from a repo that doesn't have the
sibling skills next to it, fetch them by **name**. Don't assume
they're at any particular relative path.

1. **Source of truth.** All sibling skills live in this public
   repo:

   <https://github.com/NVIDIA/nurec-skills>

   Each one is at `.agents/skills/<folder>/SKILL.md`. Skills that
   pin a specific upstream commit ship the actual file under
   `.agents/skills/<folder>/_versions/<branch>/<commit>/SKILL.md`,
   with a top-level `<folder>/SKILL.md` symlink that points at the
   currently-selected version.

2. **The name is portable; the folder isn't.** Refer to a sibling
   skill by its `name:` (e.g. `nre`). Any agent runtime that
   supports the `agentskills.io` standard can resolve a skill by
   name without knowing where the file lives.

3. **Where to look on the local disk** (try these in order):

   1. `.agents/skills/<name>/SKILL.md` (Cursor, Codex, NemoClaw)
   2. `.claude/skills/<name>/SKILL.md` (Claude Code)
   3. `.cursor/skills/<name>/SKILL.md` (project-scoped)
   4. `~/.cursor/skills/<name>/SKILL.md` (your personal skills)

4. **If none of those exist, clone the upstream:**

   ```bash
   git clone --depth 1 \
     https://github.com/NVIDIA/nurec-skills.git /tmp/nurec-skills
   cat /tmp/nurec-skills/.agents/skills/<folder>/SKILL.md
   ```

   The folder usually matches the name. The one exception is
   `ncore-data-conversion`, which lives under `ncore/`.

> **Heads-up:** any `references/`, `scripts/`, or `assets/` files
> mentioned by a sibling skill live next to **that** skill's
> `SKILL.md`, not next to this index. Open the target `SKILL.md`
> first — its own "References" section will tell you exactly which
> companion files to read.

## Keeping this index up to date

This index is **hand-curated** — it groups skills by what users
want to do, not by alphabetical name. There is no generator script
in this repo; edit this `SKILL.md` by hand whenever the sibling set
changes.

When a new sibling skill is added or a use case shifts:

1. Add a row to [Pick a skill](#pick-a-skill) for the new use case.
2. Add a row to [The skills in this folder](#the-skills-in-this-folder).
3. If the new skill changes a multi-step pipeline, update [Common
   workflows](#common-workflows).
4. Confirm the sibling's `metadata.upstream` field still points at
   the canonical upstream repo or container — the thin-local-skill
   policy at the top of this file relies on that link being live.

Otherwise the index will quietly drift and beginners will end up
reading the wrong skill.
