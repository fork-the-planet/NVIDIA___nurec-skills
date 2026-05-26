# Third-Party Notices

This repository does not vendor or redistribute third-party OSS source code or
third-party binary dependencies.

The bundled skills are Markdown documentation plus NVIDIA-authored helper files.
They do not require third-party libraries to be discovered or read. Bundled
validation scripts use only the Python standard library. Workflow instructions
may direct users to install or run external tools, packages, containers,
models, or datasets from upstream distribution channels; those artifacts are
not part of this repository and retain their own license terms.

If third-party OSS code is added to this repository in the future, its copyright
notice, attribution, and license text must be added here before release.

## Optional runtime dependencies referenced by bundled templates

The `skills/ncore/ncore_template/` example is NVIDIA-authored source code, but
when users execute it as a converter template it imports user-installed upstream
packages. Those packages are not distributed with this repository.

- NumPy — BSD-3-Clause — <https://numpy.org/>
- universal-pathlib — MIT — <https://github.com/fsspec/universal_pathlib>
