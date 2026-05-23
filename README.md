# NuRec Skills

Agent skills for **NVIDIA Omniverse NuRec** and the surrounding
neural-reconstruction stack — installable, version-pinned `SKILL.md`
bundles that teach a coding agent how to ingest sensor data, train
reconstructions, render novel views, harvest 3D objects, clean up
artifacts, and find the right NVIDIA dataset for the job.

The canonical home is <https://github.com/NVIDIA/nurec-skills>.

## What's a skill?

A **skill** is a single Markdown file (plus a few companion files)
that an agent reads on demand to gain task-specific knowledge. Each
skill in this repo follows the [agentskills.io](https://agentskills.io)
convention: a YAML frontmatter block (name, description, trigger
keywords, compatibility, upstream pointer) followed by a hand-curated
recipe. Agents that support the standard — Cursor, Claude Code, Codex,
and others — can resolve a skill by **name**, regardless of where the
file is on disk.

These skills are **thin coordination layers**. They don't redistribute
NVIDIA source; instead, they teach an agent how to drive the public
NGC containers, GitHub repos, and HuggingFace artifacts that make up
the NuRec stack.

## Skills in this repo

Start with [`nurec-index`](./.agents/skills/SKILL.md) — it routes any
NuRec task to the right sibling skill below.

| Name | Folder | Pinned version | Purpose |
|------|--------|----------------|---------|
| [`nurec-index`](./.agents/skills/SKILL.md) | `.agents/skills/` | hand-curated | Router. Read first. Picks the right skill for any NuRec task. |
| [`physical-ai-datasets`](./.agents/skills/physical-ai-datasets/SKILL.md) | `.agents/skills/physical-ai-datasets/` | hand-curated | Catalog of every NVIDIA `PhysicalAI-*` dataset on Hugging Face — AV, robotics, NuRec scenes, benchmarks. |
| [`ncore-data-conversion`](./.agents/skills/ncore/SKILL.md) | `.agents/skills/ncore/` | upstream `2026.04` | Convert any sensor recording (cameras, LiDAR, radar, IMU, depth, stereo) into [NCore V4](https://github.com/NVIDIA/ncore) — the format NRE consumes. Includes a converter template. |
| [`nre`](./.agents/skills/nre/SKILL.md) | `.agents/skills/nre/_versions/release_26.04/85ba2e2/` | NRE `release_26.04` | Train 3DGUT/3DGRT Gaussian reconstructions, render novel views (local or via gRPC), export PLY/mesh/depth, edit actors, evaluate quality. Drives the public `nvcr.io/nvidia/nre/{nre,nre-tools}` containers. |
| [`asset-harvester`](./.agents/skills/asset-harvester/SKILL.md) | `.agents/skills/asset-harvester/_versions/main/e08b1b2/` | upstream `main` @ `e08b1b2` | Extract per-object 3D Gaussian Splat assets from sparse AV-clip views via SparseViewDiT + TokenGS. Open-source ([NVIDIA/asset-harvester](https://github.com/NVIDIA/asset-harvester), Apache-2.0). |
| [`nurec-fixer`](./.agents/skills/nurec-fixer/SKILL.md) | `.agents/skills/nurec-fixer/_versions/main/617a990/` | upstream `main` @ `617a990` | Post-process novel-view renders with the NVIDIA Fixer (Difix3D+) diffusion model — removes ghosting, floaters, and temporal flicker. |

## Repo layout

```text
.agents/
└── skills/
    ├── SKILL.md                      # The nurec-index router
    ├── physical-ai-datasets/
    │   └── SKILL.md
    ├── ncore/
    │   ├── SKILL.md
    │   └── ncore_template/           # Converter scaffold for new sensor formats
    ├── nre/
    │   ├── SKILL.md            ──►  _versions/release_26.04/85ba2e2/SKILL.md
    │   ├── references/         ──►  _versions/release_26.04/85ba2e2/references/
    │   ├── scripts/            ──►  _versions/release_26.04/85ba2e2/scripts/
    │   └── _versions/release_26.04/85ba2e2/
    │       ├── SKILL.md
    │       ├── references/
    │       └── scripts/
    ├── asset-harvester/
    │   └── _versions/main/e08b1b2/   # Same shape; symlinks at the parent
    └── nurec-fixer/
        └── _versions/main/617a990/   # Same shape; symlinks at the parent
```

Skills that wrap a specific upstream commit pin that commit under
`<skill>/_versions/<branch>/<commit>/`. Symlinks at the skill root
(`<skill>/SKILL.md`, `references/`, `scripts/`, `tests.yaml`) point at
the currently-selected version, so cross-skill markdown links like
`../nre/SKILL.md` keep resolving when an agent navigates the symlinked
paths.

## Using these skills

Most modern agent runtimes already auto-discover skills under
`.agents/skills/`, `.claude/skills/`, `.cursor/skills/`, or
`~/.cursor/skills/`. The two common ways to consume this repo:

### 1. Drop the repo next to your project

Clone into your project (or a parent directory the agent indexes):

```bash
git clone https://github.com/NVIDIA/nurec-skills.git
```

Then ask your agent to do anything in the trigger surface — e.g. "use
NuRec to render this clip", "convert my ROS 2 bag to NCore", "harvest
3D assets from this driving log". The agent picks the right skill via
`nurec-index` and follows the recipe.

### 2. Install a skill into your user-space

Symlink (or copy) one or more skills into your runtime's user-space
skills directory. For Cursor:

```bash
mkdir -p ~/.cursor/skills
ln -s "$(pwd)/.agents/skills/nre"               ~/.cursor/skills/nre
ln -s "$(pwd)/.agents/skills/ncore"             ~/.cursor/skills/ncore-data-conversion
ln -s "$(pwd)/.agents/skills/asset-harvester"   ~/.cursor/skills/asset-harvester
ln -s "$(pwd)/.agents/skills/nurec-fixer"       ~/.cursor/skills/nurec-fixer
ln -s "$(pwd)/.agents/skills/physical-ai-datasets" ~/.cursor/skills/physical-ai-datasets
ln -s "$(pwd)/.agents/skills/SKILL.md"          ~/.cursor/skills/nurec-index
```

Adjust the destination directory (`~/.claude/skills`, etc.) for other
runtimes.

## Prerequisites

These skills drive external NVIDIA infrastructure. Each skill lists
its own prerequisites in detail; the headline ones:

- **OS / arch:** Linux x86_64 with NVIDIA drivers (CUDA 12.x). aarch64
  is not supported by the NRE containers.
- **GPU:** Ampere or newer (compute capability ≥ 8.0). 16 GB VRAM is
  the practical floor for the Fixer; 24–48 GB+ is recommended for NRE
  training.
- **Containers:** Docker 23+ and the NVIDIA Container Toolkit are
  required for `nre`, `nre-tools`, and `nurec-fixer`. NGC API key is
  required to pull `nvcr.io/nvidia/nre/*`.
- **Hugging Face:** an `HF_TOKEN` is required for any gated dataset
  (`nvidia/PhysicalAI-Autonomous-Vehicles*`, `nvidia/Fixer`,
  `nvidia/asset-harvester`, …).
- **Python / conda:** required for the Asset Harvester install path
  and for the NCore in-process API (`pip install nvidia-ncore`).

## Upstream sources of truth

Each skill is thin; the canonical artifacts live upstream:

- **NCore** — <https://github.com/NVIDIA/ncore>
  (spec: <https://nvidia.github.io/ncore/data/conventions.html>)
- **NRE / NuRec containers** — `nvcr.io/nvidia/nre/nre`,
  `nvcr.io/nvidia/nre/nre-tools` (NGC); product page
  <https://www.nvidia.com/en-us/omniverse/nurec/>
- **Asset Harvester** — <https://github.com/NVIDIA/asset-harvester>
  (paper: <https://arxiv.org/abs/2604.18468>; demo:
  <https://huggingface.co/spaces/nvidia/asset-harvester>)
- **Fixer (Difix3D+)** — <https://huggingface.co/nvidia/Fixer>
  (open-source code: <https://github.com/nv-tlabs/Difix3D>; paper:
  <https://arxiv.org/abs/2503.01774>)
- **Physical AI datasets** — <https://huggingface.co/nvidia> (filter
  `PhysicalAI-`); curated collection
  <https://huggingface.co/collections/nvidia/physical-ai>

When upstream releases shift, bump the pinned `_versions/<branch>/<commit>/`
folder and update the symlinks at the skill root.

## Contributing

- Frontmatter follows the [agentskills.io](https://agentskills.io)
  schema (`name`, `description`, `version`, `license`, `metadata`).
  Trigger keywords belong inside `description:` so the runtime indexes
  them.
- New skills go under `.agents/skills/<folder>/`. Use the
  `_versions/<branch>/<commit>/` layout when the skill wraps a
  specific upstream commit; otherwise keep the SKILL flat (like
  `physical-ai-datasets`).
- After adding or renaming a skill, update the [`nurec-index`](./.agents/skills/SKILL.md)
  router so it knows how to route to it.

## License

This repo is dual-licensed:

- **Skills** (every `SKILL.md` and its accompanying prose / references)
  are licensed under [**CC-BY-4.0**](https://creativecommons.org/licenses/by/4.0/).
  Each skill restates this in its frontmatter (`license: CC-BY-4.0`).
- **Scripts** (everything under `scripts/` and the `ncore_template/`
  Python package) are licensed under
  [**Apache-2.0**](https://www.apache.org/licenses/LICENSE-2.0). Every
  source file carries an `SPDX-License-Identifier: Apache-2.0` header.

The skills only *drive* upstream NVIDIA artifacts (NGC containers,
GitHub repos, Hugging Face models and datasets). Those upstream
artifacts retain their own licenses — see each skill's `metadata:`
block for the upstream pointer.
