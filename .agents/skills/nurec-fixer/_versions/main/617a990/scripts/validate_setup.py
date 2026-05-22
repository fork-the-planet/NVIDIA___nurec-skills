#!/usr/bin/env python3
"""Validate that the host meets the prerequisites for nurec-fixer.

Checks performed (no network calls):
  1. `docker` is on PATH.
  2. NVIDIA Container Toolkit is available (`nvidia-container-cli` on PATH
     OR `docker info` reports an `nvidia` runtime).
  3. `nvidia-smi` reports at least one GPU and a compute capability >= 8.0
     (Ampere or newer).
  4. `NGC_API_KEY` environment variable is set.
  5. `ngc` CLI is on PATH (used to download the harmonizer artifact).
  6. At least 30 GB of free disk space in the current working directory.

Usage:
    python scripts/validate_setup.py [--strict]

Arguments:
    --strict    Treat warnings (e.g. NGC_API_KEY missing, ngc CLI absent)
                as errors.

Environment variables:
    NGC_API_KEY    Required (warning only without --strict). NGC personal
                   API key. Used for `docker login nvcr.io` and for
                   `ngc registry model download-version`.
                   Create at: https://ngc.nvidia.com/setup/personal-keys

Exit codes:
    0 - all prerequisites met
    1 - one or more prerequisites failed (details on stderr)
    2 - unexpected error (e.g. subprocess crash)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_FREE_DISK_GB = 30


def _err(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def check_docker() -> bool:
    if shutil.which("docker") is None:
        _err("docker not on PATH — install Docker from https://docs.docker.com/engine/install/")
        return False
    _ok("docker found")
    return True


def check_nvidia_runtime() -> bool:
    if shutil.which("nvidia-container-cli") is not None:
        _ok("nvidia-container-cli found")
        return True
    try:
        out = subprocess.run(
            ["docker", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        _err(f"could not run `docker info`: {exc}")
        return False
    if "nvidia" in (out.stdout or "").lower():
        _ok("nvidia runtime reported by `docker info`")
        return True
    _err(
        "NVIDIA Container Toolkit not detected — install from "
        "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    )
    return False


def check_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        _err("nvidia-smi not on PATH — is an NVIDIA driver installed?")
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        _err(f"nvidia-smi failed: {exc.stderr.strip() or exc}")
        return False
    except subprocess.TimeoutExpired:
        _err("nvidia-smi timed out")
        return False

    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if not lines:
        _err("nvidia-smi reported no GPUs")
        return False
    bad = []
    for line in lines:
        name, _, cc_str = line.partition(",")
        cc_str = cc_str.strip()
        try:
            major_s, minor_s = cc_str.split(".")
            cc = (int(major_s), int(minor_s))
        except ValueError:
            _err(f"could not parse compute capability from: {line!r}")
            return False
        if cc < MIN_COMPUTE_CAPABILITY:
            bad.append(f"{name.strip()} (sm_{cc[0]}{cc[1]})")
        else:
            _ok(f"{name.strip()} sm_{cc[0]}{cc[1]} meets minimum sm_80")
    if bad:
        _err(
            "GPU(s) below minimum compute capability 8.0 (Ampere): "
            + ", ".join(bad)
        )
        return False
    return True


def check_ngc_api_key(strict: bool) -> bool:
    if os.environ.get("NGC_API_KEY"):
        _ok("NGC_API_KEY is set")
        return True
    msg = (
        "NGC_API_KEY environment variable not set — required to pull "
        "nvcr.io/nvidia/pytorch:24.10-py3 and to download the harmonizer "
        "artifact nvidia/nre/nurec-fixer:cosmos_3dgut_fixer_harmonizer. "
        "Create at https://ngc.nvidia.com/setup/personal-keys"
    )
    if strict:
        _err(msg)
        return False
    _warn(msg)
    return True


def check_ngc_cli(strict: bool) -> bool:
    if shutil.which("ngc") is not None:
        _ok("ngc CLI found")
        return True
    msg = (
        "ngc CLI not on PATH — install per "
        "https://docs.ngc.nvidia.com/cli/ to download the harmonizer "
        "artifact via `ngc registry model download-version`."
    )
    if strict:
        _err(msg)
        return False
    _warn(msg)
    return True


def check_disk_space() -> bool:
    cwd = Path.cwd()
    try:
        free_bytes = shutil.disk_usage(cwd).free
    except OSError as exc:
        _err(f"could not check free disk: {exc}")
        return False
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < MIN_FREE_DISK_GB:
        _err(f"only {free_gb:.1f} GB free in {cwd}; need >= {MIN_FREE_DISK_GB} GB")
        return False
    _ok(f"{free_gb:.1f} GB free in {cwd}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors.")
    args = parser.parse_args()

    checks = [
        ("docker", check_docker()),
        ("nvidia-container-toolkit", check_nvidia_runtime()),
        ("gpu", check_gpu()),
        ("ngc-api-key", check_ngc_api_key(args.strict)),
        ("ngc-cli", check_ngc_cli(args.strict)),
        ("disk-space", check_disk_space()),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nAll prerequisites met.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] unexpected failure: {exc}", file=sys.stderr)
        raise SystemExit(2)
