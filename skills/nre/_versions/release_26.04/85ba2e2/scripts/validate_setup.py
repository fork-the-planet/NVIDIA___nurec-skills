#!/usr/bin/env python3
"""Validate that the host meets the prerequisites for the NRE skill.

Checks (no network calls):
  - Host is Linux x86_64.
  - `docker` is on PATH and the daemon responds to `docker info`.
  - `nvidia-smi` is on PATH and reports an NVIDIA driver.
        - WARN  when major < 570 (R570 recommended for Ampere/Ada/Hopper).
        - FAIL  when major < 535 (R535 is the absolute floor for Fixer).
  - NVIDIA Container Toolkit is wired into Docker
    (looks for the `nvidia` runtime via `docker info`).
  - GPU has >= 24 GB of memory (best-effort via `nvidia-smi`).
  - `NGC_API_KEY` environment variable is set (required to pull
    nvcr.io/nvidia/nre images).

Usage:
    python scripts/validate_setup.py [--strict]

Arguments:
    --strict      Treat WARN lines as FAIL (non-zero exit code).

Environment variables:
    NGC_API_KEY   Required. Token used to authenticate `docker login
                  nvcr.io` and in-container model downloads.
                  Generate one at: https://org.ngc.nvidia.com/setup/api-key

Exit codes:
    0 - all prerequisites met
    1 - one or more prerequisites missing (details on stderr)
    2 - unexpected error (tool invocation crashed)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"


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


def check_platform() -> tuple[str, str]:
    system = platform.system()
    machine = platform.machine()
    if system != "Linux":
        return FAIL, f"OS is {system!r}; NRE supports Linux only."
    if machine not in ("x86_64", "amd64"):
        return FAIL, (
            f"Architecture is {machine!r}; NRE supports x86_64 only "
            "(aarch64 is not supported)."
        )
    return OK, f"{system} {machine}"


def check_docker() -> tuple[str, str]:
    if shutil.which("docker") is None:
        return FAIL, "docker not on PATH. Install Docker >= 23.0.1."
    rc, out, err = _run(["docker", "--version"])
    if rc != 0:
        return FAIL, f"`docker --version` failed: {(err or out).strip()}"
    version_line = out.strip()
    rc2, out2, err2 = _run(["docker", "info", "--format", "{{json .}}"])
    if rc2 != 0:
        return FAIL, (
            f"`docker info` failed ({(err2 or out2).strip() or 'no output'}). "
            "Is the Docker daemon running and is your user in the docker group?"
        )
    return OK, version_line


def check_nvidia_runtime() -> tuple[str, str]:
    rc, out, err = _run(["docker", "info", "--format", "{{json .}}"])
    if rc != 0:
        return WARN, "Skipping (docker info failed; see docker check)."
    try:
        info = json.loads(out.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return WARN, "Could not parse `docker info` JSON output."
    runtimes = info.get("Runtimes") or {}
    if "nvidia" not in runtimes:
        return FAIL, (
            "NVIDIA Container Toolkit not wired into Docker (no 'nvidia' "
            "runtime). See https://docs.nvidia.com/datacenter/cloud-native/"
            "container-toolkit/install-guide.html"
        )
    return OK, "Docker runtime 'nvidia' is registered."


def check_driver() -> tuple[str, str]:
    if shutil.which("nvidia-smi") is None:
        return FAIL, "nvidia-smi not on PATH. Install the NVIDIA driver."
    rc, out, err = _run([
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ])
    if rc != 0:
        return FAIL, f"nvidia-smi failed: {(err or out).strip() or 'unknown'}"
    first = out.strip().splitlines()[0] if out.strip() else ""
    match = re.match(r"^(\d+)\.(\d+)", first)
    if not match:
        return WARN, f"Could not parse driver version from {first!r}."
    major = int(match.group(1))
    if major < 535:
        return FAIL, (
            f"Driver {first} is too old. NRE needs R535+ (R570+ recommended)."
        )
    if major < 570:
        return WARN, (
            f"Driver {first} works but R570+ is recommended (R580+ on "
            "Blackwell). Consider upgrading."
        )
    return OK, f"NVIDIA driver {first}"


def check_gpu_memory() -> tuple[str, str]:
    if shutil.which("nvidia-smi") is None:
        return WARN, "nvidia-smi not on PATH; skipping GPU memory check."
    rc, out, err = _run([
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
    ])
    if rc != 0:
        return WARN, "Could not query GPU memory via nvidia-smi."
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines:
        return WARN, "No GPUs reported by nvidia-smi."
    try:
        sizes_mb = [int(line) for line in lines]
    except ValueError:
        return WARN, f"Unexpected nvidia-smi memory output: {out.strip()!r}"
    biggest_gb = max(sizes_mb) / 1024.0
    if biggest_gb < 24.0:
        return FAIL, (
            f"Largest GPU has {biggest_gb:.1f} GB; NRE needs >= 24 GB "
            "(48+ GB recommended)."
        )
    if biggest_gb < 48.0:
        return WARN, (
            f"Largest GPU has {biggest_gb:.1f} GB; usable but 48+ GB is "
            "recommended for AV training."
        )
    return OK, f"Largest GPU has {biggest_gb:.1f} GB VRAM"


def check_ngc_api_key() -> tuple[str, str]:
    if not os.environ.get("NGC_API_KEY", "").strip():
        return FAIL, (
            "NGC_API_KEY environment variable is empty. Generate a key at "
            "https://org.ngc.nvidia.com/setup/api-key and export it before "
            "running the skill."
        )
    return OK, "NGC_API_KEY is set"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (non-zero exit code).",
    )
    args = parser.parse_args()

    checks = [
        ("Platform", check_platform),
        ("Docker", check_docker),
        ("NVIDIA runtime", check_nvidia_runtime),
        ("NVIDIA driver", check_driver),
        ("GPU memory", check_gpu_memory),
        ("NGC_API_KEY", check_ngc_api_key),
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
            f"{failures} prerequisite(s) missing; resolve before pulling "
            "nvcr.io/nvidia/nre images.",
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
