# Finding sibling skills that are not on disk

If this index is being read from a repo that does not have the
sibling skills next to it, fetch them by **name**. Do not assume
they live at any particular relative path.

## 1. Source of truth

All sibling skills live in this public repo:

<https://github.com/NVIDIA/nurec-skills>

Each one is at `.agents/skills/<folder>/SKILL.md`, plus its
companion `references/`, `scripts/`, and (where present)
`tests.yaml`. Upstream commits / container tags / model release
branches are pinned inside the skill's frontmatter `metadata:`
block, not in the folder path.

## 2. The name is portable; the folder is not

Refer to a sibling skill by its `name:` (e.g. `nre`). Any agent
runtime that supports the `agentskills.io` standard can resolve a
skill by name without knowing where the file lives.

## 3. Where to look on the local disk (in order)

1. `.agents/skills/<name>/SKILL.md` (Cursor, Codex, NemoClaw)
2. `.claude/skills/<name>/SKILL.md` (Claude Code)
3. `.cursor/skills/<name>/SKILL.md` (project-scoped)
4. `~/.cursor/skills/<name>/SKILL.md` (personal skills)

## 4. If none of those exist, ask the user before cloning the upstream

> **Security note — DO NOT auto-clone.** The next step performs a
> **network fetch** of an external GitHub repository **and writes to
> the local filesystem**. That side effect can violate organizational
> network/security policies and exposes the user to a supply-chain
> risk if the upstream is ever tampered with. The agent **MUST**
> show this notice to the user and get explicit consent (e.g. "OK
> to `git clone https://github.com/NVIDIA/nurec-skills` to a local
> directory?") **before** running any `git clone`. Prefer pinning a
> known-good commit or tag (`--branch <tag>` or an explicit SHA)
> over `HEAD`, and prefer fetching the single needed `SKILL.md` over
> a full clone whenever the upstream layout allows it.

Only after the user confirms, run (replacing `<DIR>` with a path the
user agreed to — do not silently default to `/tmp`):

```bash
git clone --depth 1 --branch <pinned-tag-or-sha> \
  https://github.com/NVIDIA/nurec-skills.git <DIR>/nurec-skills
cat <DIR>/nurec-skills/.agents/skills/<folder>/SKILL.md
```

The folder name always matches the skill `name:` frontmatter
field; e.g. the `ncore` skill lives at `.agents/skills/ncore/`.

If the user declines, stop here and report which sibling skill is
missing; do **not** fall back to silent network access.

> **Heads-up:** any `references/`, `scripts/`, or `assets/` files
> mentioned by a sibling skill live next to **that** skill's
> `SKILL.md`, not next to this index. Open the target `SKILL.md`
> first — its own "References" section will tell the agent exactly
> which companion files to read.

## Thin-local-skill policy

The repo this index ships in
(<https://github.com/NVIDIA/nurec-skills>) treats these skills as
coordination layers. The canonical sources live upstream:

- NCore — <https://github.com/NVIDIA/ncore>
- NRE — NGC containers `nvcr.io/nvidia/nre/nre` and
  `nvcr.io/nvidia/nre/nre-tools`
- Asset Harvester — <https://github.com/NVIDIA/asset-harvester>
- DiffusionHarmonizer — <https://github.com/NVIDIA/harmonizer> and
  <https://huggingface.co/nvidia/DiffusionHarmonizer>
- Physical AI datasets — `PhysicalAI-*` under
  <https://huggingface.co/nvidia>

Each sibling skill links its upstream in its frontmatter
`metadata.upstream` field.
