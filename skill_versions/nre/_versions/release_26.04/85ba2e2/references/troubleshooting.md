# NRE — Troubleshooting

Detail moved out of `SKILL.md`.

| Error | Cause | Solution |
|-------|-------|----------|
| `Unable to find image 'nvcr.io/nvidia/nre/nre:latest'` | Docker not authenticated to NGC. | `export NGC_API_KEY=<key>; docker login nvcr.io` with `Username: $oauthtoken`. |
| `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False` | Container started without `--gpus all` / NVIDIA Container Toolkit missing. | Add `--gpus all` (and `--runtime=nvidia` if needed); install NVIDIA Container Toolkit >= 1.13.5. |
| `OOM Killed` / `CUDA out of memory` during training | Default Hydra recipe targets >= 48 GB VRAM. | Reduce `dataset.camera_ids` to a single ID, lower `trainer.max_epochs`, switch to `trainer.precision=16-mixed`, lower `dataset.n_train_sample_camera_rays`, or run on a 48 GB+ GPU. |
| `wandb` prompt blocks training in non-TTY mode | NRE asks for a wandb mode during validation. | Choose option `3` (skip) interactively, or set `logger=tensorboard` (docs default), `logger.offline=true`, or `WANDB_MODE=disabled` in the container environment. |
| `export-*` complains the artifact path is wrong | Export commands re-read `<output_dir>/<RUN-ID>/config/parsed.yaml`; the dataset and output volume mounts MUST match training. | Re-mount with the same host paths as the training command. |
| `serve-grpc` fails to find the USDZ | `--artifact-glob` must end in `.usdz`. | Use e.g. `--artifact-glob /workdir/output/<RUN-ID>/usd-out/last.usdz`. Quote the glob. |
| `serve-grpc` starts but USDZs aren't there | Training didn't write a USDZ. | Re-train with `checkpoint.artifact.enabled=true`, or re-export an existing checkpoint with `export-usdz-artifact`. |
| `replacement_id <id> not found in external_assets` from `render-grpc --edit-assets` | The repackage step (`export-external-assets`) was skipped or pointed at a USDZ that did not include that AH output. | Re-run `export-external-assets` with the correct `--external-assets-dir` and use the resulting `target-external-assets.usdz` in `serve-grpc`. |
| Edits silently ignored from `render-grpc --edit-assets` | Server was started without `--enable-editing-actors`. | Restart `serve-grpc` with that flag. |
| `--enable-nrend` / `--use-gsplat` deprecation warning | Old renderer flags. | Replace with `--renderer nrend` or `--renderer gsplat`. |
| `--no-enable-nrend` warning in old asset-editing docs | Same deprecation. | Use `--renderer default` (or `--renderer gsplat`). |
| Driver mismatch on Blackwell (RTX Pro 6000D) | R570 is too old. | Upgrade host driver to R580+. |
| Multi-GPU training crashes with `dataset.aux_data=false` | Known issue carried from 25.06–25.07. | Stay single-GPU when debugging this combination, or generate aux data first (use `nre-tools`). |
| `render-grpc` LiDAR responses too large for gRPC | Default gRPC message limit. | The bundled `render-grpc` client already raises send/receive limits to 50 MB; if writing your own client, set `grpc.max_send_message_length` and `grpc.max_receive_message_length` similarly. |
| Old USDZ loads slowly every run | NRE upgrades artifacts in memory each load. | Run `upgrade-artifact --input old.usdz --output upgraded.usdz` once and use the upgraded file thereafter. |
| `compute-metrics` "CUDA not available" warning | Default `--device=cuda`. | Pass `--device=cpu` or run with `--gpus all`. |
