# Evaluation Report

Evaluation of the `nurec-fixer` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `nurec-fixer`
- Evaluation date: 2026-05-28
- NVSkills-Eval profile: `external`
- Overall verdict: PASS
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

Tier 1 validation passed with observations. NVSkills-Eval ran 9 checks and found 6 total findings.

Top findings:

- MEDIUM QUALITY/quality_efficiency: Deeply nested references in wrapper-image.md (`skills/nurec-fixer/SKILL.md`)
- MEDIUM SECURITY/Unknown (SQP-2): The documentation instructs users to set HF_TOKEN as a plain environment variable (`export HF_TOKEN=<your-hugging-face-t (`references/inference.md:32`)
- MEDIUM SECURITY/Unknown (SQP-2): The documentation instructs users to pass the NGC_API_KEY value via shell command substitution in a docker login invocat (`references/wrapper-image.md:24`)
- LOW QUALITY/quality_discoverability: Description very long (309 chars, recommend 50-150) (`skills/nurec-fixer/SKILL.md`)
- LOW SCRIPT_LINT/magic_numbers: validate_setup.py contains magic numbers (`skills/nurec-fixer/scripts/validate_setup.py`)

## Tier 2: Deduplication Summary

Tier 2 validation passed. NVSkills-Eval ran 2 checks and found 0 total findings.

Notable observations:

- Context Deduplication: Collected 8 file(s)
- Inter-Skill Deduplication: Parsed skill 'nurec-fixer': 309 char description

## Publication Recommendation

The skill is suitable to proceed toward NVSkills-Eval publication based on this benchmark. Skill owners should keep this file with the skill and refresh it when the evaluation dataset, skill behavior, or target agents materially change.
