# Canonical NuRec Skill Catalog (External)

A copy of the canonical-NuRec routing block from
the canonical NuRec skill repo, restated here so the `nre`
skill (and any skill that nests it) is self-contained. The `nre`
flow frequently hands work off to these sibling NuRec-stack skills
(`ncore` for dataset conversion, `nurec-fixer` for known render
failure modes, `physical-ai-datasets` for PAI clip discovery,
`asset-harvester` for non-NuRec assets staged alongside a render),
so the routing rule is part of the skill's surface area, not just a
skill-tree convention.

---

## Catalog

The following NuRec-stack skills may live in this repo but if not, get
from the canonical catalog at <https://github.com/NVIDIA/nurec-skills>:

| Slug                   | Source path in canonical repo                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| `asset-harvester`      | <https://github.com/NVIDIA/nurec-skills/blob/main/.agents/skills/asset-harvester/SKILL.md>      |
| `ncore`                | <https://github.com/NVIDIA/nurec-skills/blob/main/.agents/skills/ncore/SKILL.md>                |
| `nre`                  | <https://github.com/NVIDIA/nurec-skills/blob/main/.agents/skills/nre/SKILL.md>                  |
| `nurec-fixer`          | <https://github.com/NVIDIA/nurec-skills/blob/main/.agents/skills/nurec-fixer/SKILL.md>          |
| `physical-ai-datasets` | <https://github.com/NVIDIA/nurec-skills/blob/main/.agents/skills/physical-ai-datasets/SKILL.md> |

The canonical `nre` skill covers both the 26.02 (legacy) and 26.04
(current) Docker CLI images — there is no separate `nre-26.02`
directory there.

---

## Routing rules

1. When a user request matches a trigger from one of these skills, open
   the local sibling skill if it exists; otherwise fetch the SKILL.md
   from the canonical URL above.
2. Cross-references from versioned skills should prefer the sibling
   skill name or the canonical URL over fragile local paths such as
   `skills/<deleted-slug>/SKILL.md`.
3. Keep this table in sync with the top-level `nurec-index` skill when
   sibling skills are added, renamed, or removed.

---

## Common cross-references

For agents tracing the rule end-to-end:

- **`nurec-fixer`** (NuRec post-train fix-up) — failed
  reconstructions, bad sky / floater triage, and re-training
  prompts all route there.
- **`nre`** is the canonical home for the underlying `nre render` /
  `nre serve-grpc` CLI semantics that this skill drives via Docker.
  Use it when a caller asks about a flag this skill doesn't
  document (the `26.04+` CLI surface is large; the local
  references cherry-pick a common subset).
- **`ncore`** is the canonical home for NCore dataset conversion
  (PAI → NCore v4 in `pai-nurec.yaml`'s `convert-pai-to-ncore`
  task, rosbag → NCore conversions, etc.).
- **`physical-ai-datasets`** is the canonical home for the PAI
  Hugging Face dataset reference itself (clip discovery, schema,
  what each release contains).
- **`asset-harvester`** shows up when a caller asks for non-NuRec
  assets staged alongside a render.
