#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate that the host meets the prerequisites for asset-harvester.

Checks (non-network):
  - conda is on PATH
  - nvidia-smi is on PATH and reports driver >= 570
  - A GCC 10-13 compiler is on PATH
  - Python >= 3.10 is on PATH
  - HF_TOKEN environment variable is set (or prints guidance)

Usage:
    python scripts/validate_setup.py [--strict]

Arguments:
    --strict      Treat warnings (e.g. missing HF_TOKEN) as errors.

Environment variables:
    HF_TOKEN      Optional here, but required at runtime to download
                  the gated `nvidia/asset-harvester` checkpoints.
                  Generate one at: https://huggingface.co/settings/tokens

Exit codes:
    0 - all prerequisites met
    1 - one or more prerequisites missing (details on stderr)
    2 - unexpected error (tool invocation crashed)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys


WARN = "WARN"
FAIL = "FAIL"
OK = "OK"


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: not found"
    except Exception as exc:  # pragma: no cover - defensive
        return 2, "", f"{cmd[0]}: {exc}"


def check_python() -> tuple[str, str]:
    major, minor = sys.version_info.major, sys.version_info.minor
    version = f"{major}.{minor}"
    if (major, minor) < (3, 10):
        return FAIL, f"Python {version} is too old (need >= 3.10)."
    return OK, f"Python {version}"


def check_conda() -> tuple[str, str]:
    if shutil.which("conda") is None:
        return FAIL, "conda not on PATH. Install Miniconda/Miniforge."
    rc, out, _ = _run(["conda", "--version"])
    if rc != 0:
        return FAIL, "conda failed to invoke."
    return OK, out.strip()


def check_driver() -> tuple[str, str]:
    if shutil.which("nvidia-smi") is None:
        return FAIL, "nvidia-smi not on PATH. Install the NVIDIA driver >= 570."
    rc, out, err = _run([
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ])
    if rc != 0:
        return FAIL, f"nvidia-smi failed: {err.strip() or 'unknown'}"
    first = out.strip().splitlines()[0] if out.strip() else ""
    match = re.match(r"^(\d+)\.(\d+)", first)
    if not match:
        return WARN, f"Could not parse driver version from '{first}'."
    major = int(match.group(1))
    if major < 570:
        return FAIL, (
            f"Driver {first} is too old. CUDA 12.8 needs driver >= 570."
        )
    return OK, f"NVIDIA driver {first}"


def check_gcc() -> tuple[str, str]:
    if shutil.which("gcc") is None:
        return WARN, "gcc not on PATH; setup.sh may pick a conda fallback."
    rc, out, _ = _run(["gcc", "-dumpversion"])
    if rc != 0:
        return WARN, "Unable to query gcc version."
    version = out.strip().split(".")[0]
    try:
        major = int(version)
    except ValueError:
        return WARN, f"Unexpected gcc version '{out.strip()}'."
    if major < 10 or major > 13:
        return WARN, (
            f"gcc {out.strip()} is outside the tested 10-13 range; "
            "setup.sh will probe for an alternative."
        )
    return OK, f"gcc {out.strip()}"


def check_hf_token() -> tuple[str, str]:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        return WARN, (
            "HF_TOKEN not set. Required at runtime for "
            "`hf download nvidia/asset-harvester`. "
            "Generate one at https://huggingface.co/settings/tokens"
        )
    return OK, "HF_TOKEN is set"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors.",
    )
    args = parser.parse_args()

    checks = [
        ("Python", check_python),
        ("conda", check_conda),
        ("NVIDIA driver", check_driver),
        ("GCC", check_gcc),
        ("HF_TOKEN", check_hf_token),
    ]

    failures = 0
    warnings = 0
    for name, fn in checks:
        try:
            status, detail = fn()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[{FAIL}] {name}: unexpected error: {exc}", file=sys.stderr)
            return 2

        stream = sys.stdout if status == OK else sys.stderr
        print(f"[{status}] {name}: {detail}", file=stream)
        if status == FAIL:
            failures += 1
        elif status == WARN:
            warnings += 1

    print(file=sys.stderr)
    if failures:
        print(
            f"{failures} prerequisite(s) missing; fix before running setup.sh.",
            file=sys.stderr,
        )
        return 1
    if args.strict and warnings:
        print(
            f"{warnings} warning(s) in --strict mode — treating as failure.",
            file=sys.stderr,
        )
        return 1
    if warnings:
        print(
            f"{warnings} warning(s); host is usable but review the messages.",
            file=sys.stderr,
        )
    else:
        print("All prerequisites look good.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
