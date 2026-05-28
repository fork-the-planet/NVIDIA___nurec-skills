# Evaluation Report

Evaluation of the `nre` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `nre`
- Evaluation date: 2026-05-28
- NVSkills-Eval profile: `external`
- Overall verdict: FAIL
- Tier 3 live agent evaluation: not available in this report

## Agents Used

- Tier 3 agent details were not available in this report.

## Metrics Used

Reported benchmark dimensions:

- Security: checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access.
- Correctness: checks whether the agent follows the expected workflow and produces the correct final output.
- Discoverability: checks whether the agent loads the skill when relevant and avoids using it when irrelevant.
- Effectiveness: checks whether the agent performs measurably better with the skill than without it.
- Efficiency: checks whether the agent uses fewer tokens and avoids redundant work.

Underlying evaluation signals used in this run:

- No Tier 3 evaluation signal details were available in this report.

## Test Tasks

Tier 3 evaluation task details were not available in this report.

## Results

Tier 3 dimension rollup was not available in this report.

## Tier 1: Static Validation Summary

Tier 1 validation passed with observations. NVSkills-Eval ran 9 checks and found 15630 total findings.

Top findings:

- MEDIUM PII/gps_coordinates: GPS coordinates (location information) (`references/nre-image-notes.md:187`)
- MEDIUM PII/gps_coordinates: GPS coordinates (location information) (`references/nre-image-notes.md:188`)
- MEDIUM PII/gps_coordinates: GPS coordinates (location information) (`references/local-render.md:72`)
- MEDIUM PII/gps_coordinates: GPS coordinates (location information) (`references/asset-editing.md:61`)
- MEDIUM PII/gps_coordinates: GPS coordinates (location information) (`references/asset-editing.md:111`)

## Tier 2: Deduplication Summary

Tier 2 validation reported findings. NVSkills-Eval ran 2 checks and found 4 total findings.

Top findings:

- HIGH DUPLICATE/duplicate: Duplicate content found within references/local-render.md:
  "# Ground truth — uses render_ground_truth() from the shared helper above." in references/local-render.md (lines 300-302)
  vs "# Ground truth — uses render_ground_truth() from the shared helper above." in references/local-render.md (lines 397-399) (`references/local-render.md:300`)
- HIGH DUPLICATE/duplicate: Duplicate content found across references/NRE_RenderClient/README.md and references/asset-editing.md and references/cli-reference.md and references/cookbook.md and references/grpc-api.md and references/physical-ai-render.md and references/workflows.md:
  "#    USDZs and ignores the rest." in references/NRE_RenderClient/README.md (lines 94-102)
  vs "# ...or roll your own docker run:" in references/NRE_RenderClient/README.md (lines 442-447)
  vs "## Pipeline overview" in references/asset-editing.md (lines 9-40)
  vs "## Step 3 — Launch the gRPC server with editing enabled" in references/asset-editing.md (lines 263-291)
  vs "## Step 4 — Render with `--edit-assets`" in references/asset-editing.md (lines 292-326)
  vs "### `render` (in-container, no gRPC)" in references/cli-reference.md (lines 167-179)
  vs "### `render` (in-container, no gRPC)" in references/cli-reference.md (lines 183-204)
  vs "### `serve-grpc`" in references/cli-reference.md (lines 252-300)
  vs "### `render-grpc`" in references/cli-reference.md (lines 305-314)
  vs "### `render-grpc`" in references/cli-reference.md (lines 318-335)
  vs "## Render frames locally (no gRPC) at quarter resolution" in references/cookbook.md (lines 43-60)
  vs "## Render at native resolution along original sensor poses" in references/cookbook.md (lines 61-87)
  vs "## Launch sensorsim gRPC server with editing enabled" in references/cookbook.md (lines 115-131)
  vs "## Render a LiDAR sweep from a running `serve-grpc`" in references/cookbook.md (lines 132-144)
  vs "# NuRec sensorsim gRPC API" in references/grpc-api.md (lines 1-19)
  vs "## 1. Server" in references/grpc-api.md (lines 20-65)
  vs "## 6. `render-grpc` CLI helper" in references/grpc-api.md (lines 409-463)
  vs "## 4. Stand up the gRPC server" in references/physical-ai-render.md (lines 183-200)
  vs "## Workflow G — Headless rendering of LiDAR sweeps" in references/workflows.md (lines 106-111)
  vs "# Server (one terminal)." in references/workflows.md (lines 112-124)
  vs "# Client (another terminal / job)." in references/workflows.md (lines 125-142) (`references/NRE_RenderClient/README.md:94`)
- HIGH DUPLICATE/duplicate: Duplicate content found within references/example-workflows/bash/nurec_workflow_pai.md:
  "# End-to-End Workflow: PAI → CDS → NCore → NuRec → Asset Harvester → Alpamayo → AlpaSim" in references/example-workflows/bash/nurec_workflow_pai.md (lines 45-47)
  vs "# End-to-End Workflow: PAI → CDS → NCore → NuRec → Asset Harvester → Alpamayo → AlpaSim" in references/example-workflows/bash/nurec_workflow_pai.md (lines 48-49) (`references/example-workflows/bash/nurec_workflow_pai.md:45`)
- HIGH DUPLICATE/duplicate: Duplicate content found within references/example-workflows/bash/nurec_workflow_pai.md:
  "# (or use --id-field, --video-field, --text-field to match custom names.)" in references/example-workflows/bash/nurec_workflow_pai.md (lines 330-330)
  vs "# (or use --id-field, --video-field, --text-field to match custom names.)" in references/example-workflows/bash/nurec_workflow_pai.md (lines 331-331) (`references/example-workflows/bash/nurec_workflow_pai.md:330`)

## Publication Recommendation

The skill should be reviewed before NVSkills-Eval publication. Skill owners should address the findings above and rerun NVSkills-Eval to refresh this benchmark.
