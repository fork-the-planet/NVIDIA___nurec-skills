#!/usr/bin/env bash
# session_warm_server.sh -- idempotent warm boot of `nre serve-grpc`.
#
# Boot once at the start of a render-heavy session (chat agent,
# benchmark loop, batch job, etc.) so each per-turn render can go
# through the thin client (../NRE_RenderClient/scripts/thin_client.py)
# instead of repeating Docker container start + Python interpreter +
# CUDA context init + model load on every turn.
#
# Idempotent: if a session warm server is already up under the canonical
# container name (default: nre-grpc-session), exits 0 immediately.
# Otherwise boots one, polls for readiness, then exits.
#
# Capability fallback: if a caller needs something the thin client
# doesn't expose -- custom rig trajectory (--custom-rig-trajectory),
# lidar, actor / asset edits, rolling shutter, or
# --replicate-training-views semantics -- drop to per-turn `nre render`
# CLI for that turn. The warm server stays up regardless; the next
# thin-client-supported turn re-uses it. Two model copies (server +
# CLI) will co-exist in GPU memory during a fallback turn -- fine on
# H100/80GB for single-clip-at-a-time usage, watch GPU memory on
# smaller cards.
#
# Image selection: this script does not know or hardcode any NRE image
# tags. It iterates locally-cached images, version-probes them, and
# picks the first one that reports 26.04+. Pass NRE_IMAGE=<tag> to
# bypass discovery for dev work against a specific tag.
#
# Environment overrides:
#   NRE_IMAGE              -- skip discovery, use this image (still
#                             version-checked).
#   NRE_GRPC_PORT          -- Server port (default 8080)
#   NRE_GRPC_CONTAINER     -- Container name (default nre-grpc-session)
#   NRE_GRPC_ARTIFACT_GLOB -- Artifact glob inside the container
#                             (default /inputs/sdg/**/*.usdz -- a
#                             recursive search for any .usdz under
#                             the mounted SDG root, so we don't have
#                             to know the caller's exact per-clip
#                             layout. Covers at least:
#                               * <uuid>/usd_out/last.usdz
#                                 (historic clipgt-<UUID> /
#                                 nurec-fixer staging convention)
#                               * <uuid>/<uuid>.usdz
#                                 (PhysicalAI-Autonomous-Vehicles-NuRec
#                                 mirror layout)
#                               * any other ad-hoc per-clip layout.)
#   NRE_GRPC_USDZ_HOST_DIR -- Host dir to bind-mount as /inputs/sdg.
#                             REQUIRED -- no default. Point at the
#                             root of your USDZ tree; the recursive
#                             glob below catches the .usdz wherever
#                             you stage it under this root. The glob
#                             runs against this mount once at server
#                             startup. Clips downloaded after startup
#                             are NOT auto-discovered -- the thin
#                             client picks them up at request time
#                             via --scene-url + --scene-id, which the
#                             SceneDownloadInterceptor turns into a
#                             runtime _register_local_scene call. So
#                             new clips still need to land under this
#                             host root (otherwise they're not
#                             reachable inside the container), but
#                             they don't need a warm-server restart.
#   NRE_GRPC_BOOT_TIMEOUT  -- Boot wait seconds (default 120)
#   NRE_GRPC_CACHE_SIZE    -- Server-side LRU size for loaded scenes
#                             (passed to `serve-grpc --cache-size`,
#                             default 3). Upstream's own default is
#                             10; we pin lower because each loaded
#                             USDZ holds gaussian-splat tensors +
#                             windshield model state on the GPU and
#                             A100-40GB-class hardware can't
#                             comfortably keep 10 scenes resident.
#                             3 covers the typical demo flow
#                             (default clip + 1-2 user-requested
#                             clips); raise it only if you have
#                             headroom and need more churn.
#
# Exit codes:
#   0  -- warm server up (or already up)
#   1  -- generic boot failure (timeout, container died)
#   2  -- precondition unmet (no image cached, USDZ dir missing)
#   3  -- NRE version is < 26.04. Warm server requires --renderer default
#         and SceneDownloadInterceptor, both 26.04+. Caller should fall
#         back to per-turn CLI for the session and warn the user.
#
# State file at /tmp/nre-grpc-session.state records {container_id, port}
# for session_teardown.sh to consume.

set -euo pipefail

CONTAINER_NAME="${NRE_GRPC_CONTAINER:-nre-grpc-session}"
PORT="${NRE_GRPC_PORT:-8080}"
ARTIFACT_GLOB="${NRE_GRPC_ARTIFACT_GLOB:-/inputs/sdg/**/*.usdz}"
USDZ_HOST_DIR="${NRE_GRPC_USDZ_HOST_DIR:-}"
BOOT_TIMEOUT="${NRE_GRPC_BOOT_TIMEOUT:-120}"
CACHE_SIZE="${NRE_GRPC_CACHE_SIZE:-3}"
STATE_FILE="/tmp/nre-grpc-session.state"

log() {
    printf '[warm-server] %s\n' "$*" >&2
}

# Idempotency: if the container is up, do nothing (refresh state file
# in case it was nuked between sessions).
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    cid="$(docker inspect --format '{{.Id}}' "$CONTAINER_NAME")"
    printf '%s %s\n' "$cid" "$PORT" > "$STATE_FILE"
    log "container '$CONTAINER_NAME' already running on port $PORT; nothing to do."
    exit 0
fi

# Boot-on-stale: remove any stopped container under the same name.
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    log "removing stopped container '$CONTAINER_NAME' before re-boot"
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

# Capability-based image selection.
#
# Warm server requires NRE 26.04+ (--renderer default and
# SceneDownloadInterceptor are both 26.04-gated). We do not know or
# hardcode tags -- we iterate locally-cached images and pick the first
# one whose --version reports >=26.04. If $NRE_IMAGE is set, we skip
# discovery and version-check that one.
#
# Iteration order: candidates are sorted by the version extracted from
# the tag (descending), so a pinned ":X.Y.Z" gets probed before a
# floating ":latest" pointing at older bits. Tags without a parseable
# X.Y[.Z] sort last (treated as 0.0.0). On a host with both a 26.04+
# pinned tag and a floating tag pointing at <26.04, the higher tag
# wins on the first probe (~3-5 s) instead of paying for a wasted
# probe on the older one first.
#
# Discovery filter: candidate name must contain "nre" but must not
# contain "tools" or "isaac" -- those NRE-adjacent images don't speak
# `serve-grpc` and the warm server would crash trying to invoke it.
#
# Version check: tag-first, runtime-probe fallback.
#
# If the image tag itself encodes a parseable semver (e.g.
# "26.4.117-f01ec6ed"), we trust it directly. Reasons:
#   - The bake declares the image's nominal version via the tag; if
#     the binary inside disagrees, warm-server boot fails and the
#     caller drops to CLI -- no silent corruption risk.
#   - Skipping the runtime probe avoids depending on every renderer
#     image's entrypoint accepting `--version` cleanly. We've seen at
#     least one EA build (`nvcr.io/nvidian/ct-toronto-ai/nre_run:
#     26.4.117-f01ec6ed`) where the runtime probe didn't yield a
#     parseable version string even though the tag clearly declares
#     26.4 -- a strict probe-only path silently excludes that image.
#   - Skipping the probe also saves ~3-5 s + CUDA init per candidate.
#
# Floating tags (":latest", ":staging", anything without a parseable
# X.Y[.Z]) fall through to a runtime `--version` probe. The probe
# correctly accepts/rejects whatever the tag actually resolves to,
# at the cost of one Docker boot per such tag.
#
# Runtime probe details: `nre --version` is wired via
# click.version_option and prints "<prog>, version X.Y.Z-sha[+dirty]";
# we extract the first semver triple. `--gpus all` is required even
# for `--version` because NRE initialises a CUDA context at click
# load time; without GPU access the entrypoint crashes before
# printing any version string and our regex matches nothing.
# `timeout 60s` is a hard ceiling so a wedged probe (CUDA init hang,
# no GPU available, etc.) can't deadlock the discovery loop.

# Extract the first X.Y[.Z] substring from $1, print to stdout (empty
# if no match). Pure-bash regex on purpose: a `grep -oE ... | head -1`
# pipeline returns exit 1 when no match is found, which combines
# badly with this script's `set -euo pipefail` (the failure
# propagates through the substitution and can kill an enclosing
# pipeline before it produces any output).
extract_semver() {
    local in="$1"
    if [[ "$in" =~ [0-9]+\.[0-9]+(\.[0-9]+)? ]]; then
        printf '%s' "${BASH_REMATCH[0]}"
    fi
}

# Compare an X.Y[.Z] string against 26.04.
#   0 -> parses and is >= 26.04
#   1 -> parses and is <  26.04
#   2 -> empty / unparseable (caller decides next step)
semver_at_least_26_04() {
    local ver_str="$1" major minor rest
    [[ -z "$ver_str" ]] && return 2
    major="${ver_str%%.*}"
    rest="${ver_str#*.}"
    minor="${rest%%.*}"
    [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ ]] || return 2
    (( major > 26 )) && return 0
    (( major == 26 && minor >= 4 )) && return 0
    return 1
}

# Parse X.Y[.Z] out of the tag (text after the last ":") and check.
# Returns 0/1/2 with the same semantics as semver_at_least_26_04.
# Logs each branch outcome to stderr so the agent can read the trace
# back when discovery rejects an unexpected image.
nre_tag_at_least_26_04() {
    local img="$1" tag ver_str rc
    tag="${img##*:}"
    if [[ "$tag" == "$img" ]]; then
        log "  tag-check $img: image has no ':' (untagged) -> defer to runtime probe"
        return 2
    fi
    ver_str="$(extract_semver "$tag")"
    if [[ -z "$ver_str" ]]; then
        log "  tag-check $img: tag '$tag' has no parseable X.Y[.Z] -> defer to runtime probe"
        return 2
    fi
    semver_at_least_26_04 "$ver_str"; rc=$?
    case $rc in
        0) log "  tag-check $img: tag declares $ver_str (>=26.04) -> accept" ;;
        1) log "  tag-check $img: tag declares $ver_str (<26.04)  -> reject" ;;
        2) log "  tag-check $img: tag '$tag' parsed '$ver_str' but not numeric X.Y -> defer to runtime probe" ;;
    esac
    return $rc
}

# Run the image with --version and check the output. Logs the
# captured docker exit code + a snippet of the output on rejection
# so the agent can see *why* a probe failed (timeout, GPU access,
# unexpected entrypoint, no parseable version, etc.) instead of just
# "rejected".
nre_runtime_at_least_26_04() {
    local img="$1" out docker_rc ver_str
    if out=$(timeout 60s docker run --rm --gpus all "$img" --version 2>&1); then
        docker_rc=0
    else
        docker_rc=$?
    fi
    if (( docker_rc != 0 )); then
        log "  runtime-probe $img: docker run --version exited $docker_rc -> reject"
        log "    output (first 5 lines):"
        printf '%s\n' "$out" | head -5 | sed 's/^/      /' >&2
        return 1
    fi
    ver_str="$(extract_semver "$out")"
    if [[ -z "$ver_str" ]]; then
        log "  runtime-probe $img: --version printed no parseable X.Y[.Z] semver -> reject"
        log "    output (first 5 lines):"
        printf '%s\n' "$out" | head -5 | sed 's/^/      /' >&2
        return 1
    fi
    if semver_at_least_26_04 "$ver_str"; then
        log "  runtime-probe $img: reports $ver_str (>=26.04) -> accept"
        return 0
    fi
    log "  runtime-probe $img: reports $ver_str (<26.04) -> reject"
    return 1
}

# Combined entry point used by both branches below: trust the tag
# if it parses, else runtime-probe. 0=accept, 1=reject.
nre_version_at_least_26_04() {
    local img="$1" rc
    nre_tag_at_least_26_04 "$img"; rc=$?
    case $rc in
        0) return 0 ;;
        1) return 1 ;;
        2) nre_runtime_at_least_26_04 "$img" ;;
    esac
}

is_renderer_image_name() {
    local img="$1"
    [[ "$img" == *nre* ]] || return 1
    [[ "$img" == *tools* || "$img" == *isaac* ]] && return 1
    return 0
}

if [[ -n "${NRE_IMAGE:-}" ]]; then
    log "discovery: NRE_IMAGE override = '$NRE_IMAGE' -- skipping cache scan"
    if ! docker image inspect "$NRE_IMAGE" >/dev/null 2>&1; then
        echo "STOP [warm-server]: NRE_IMAGE='$NRE_IMAGE' is not cached locally." >&2
        echo "  Pull it first or unset NRE_IMAGE to fall back to cache discovery." >&2
        exit 2
    fi
    if ! nre_version_at_least_26_04 "$NRE_IMAGE"; then
        echo "STOP [warm-server]: NRE_IMAGE='$NRE_IMAGE' did not pass the >=26.04 check (see tag-check / runtime-probe lines above)." >&2
        exit 3
    fi
else
    # Build a version-sorted candidate list (highest tag-version first)
    # so we probe the most-likely-to-pass image first.
    declare -a all_images=()
    declare -a candidates=()

    while read -r line; do
        all_images+=("$line")
    done < <(docker images --format '{{.Repository}}:{{.Tag}}' | grep -v '<none>:<none>')
    log "discovery: ${#all_images[@]} image(s) cached, filtering for NRE renderer..."

    while read -r candidate; do
        is_renderer_image_name "$candidate" || continue
        candidates+=("$candidate")
        log "candidate: $candidate"
        if nre_version_at_least_26_04 "$candidate"; then
            NRE_IMAGE="$candidate"
            break
        fi
    done < <(
        printf '%s\n' "${all_images[@]}" \
            | while read -r line; do
                tag="${line##*:}"
                v="$(extract_semver "$tag")"
                printf '%s\t%s\n' "${v:-0.0.0}" "$line"
              done \
            | sort -V -r | cut -f2
    )

    if [[ -z "${NRE_IMAGE:-}" ]]; then
        if (( ${#candidates[@]} == 0 )); then
            echo "STOP [warm-server]: no NRE renderer images found in local cache." >&2
            echo "  Filter: image name contains 'nre' but not 'tools' or 'isaac'." >&2
            echo "  Current cache (${#all_images[@]} image(s)):" >&2
            if (( ${#all_images[@]} > 0 )); then
                printf '    %s\n' "${all_images[@]}" >&2
            else
                echo "    (cache is empty)" >&2
            fi
        else
            echo "STOP [warm-server]: ${#candidates[@]} candidate NRE image(s) found, none reported >=26.04:" >&2
            for c in "${candidates[@]}"; do echo "    $c" >&2; done
            echo "  See per-candidate tag-check / runtime-probe lines above for the" >&2
            echo "  per-image rejection reason." >&2
        fi
        echo "  Falling back to per-turn CLI is the right call -- the agent" >&2
        echo "  should warn the user once that this session is on the slower" >&2
        echo "  CLI path because no >=26.04 NRE image was usable." >&2
        exit 3
    fi
fi
log "image: $NRE_IMAGE (NRE 26.04+, warm server supported)"

if [[ -z "$USDZ_HOST_DIR" ]]; then
    echo "STOP [warm-server]: NRE_GRPC_USDZ_HOST_DIR is not set." >&2
    echo "  Export it to the host directory that holds your USDZ tree." >&2
    echo "  Example: NRE_GRPC_USDZ_HOST_DIR=/data/usdz bash session_warm_server.sh" >&2
    exit 2
fi
if [[ ! -d "$USDZ_HOST_DIR" ]]; then
    echo "STOP [warm-server]: USDZ host dir '$USDZ_HOST_DIR' does not exist." >&2
    echo "  Re-check NRE_GRPC_USDZ_HOST_DIR or create the directory before retrying." >&2
    exit 2
fi

log "booting '$CONTAINER_NAME' on port $PORT (artifact-glob=$ARTIFACT_GLOB cache-size=$CACHE_SIZE)"
docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all --shm-size 64g \
    --network host \
    -v "$USDZ_HOST_DIR:/inputs/sdg:ro" \
    "$NRE_IMAGE" \
    serve-grpc \
        --artifact-glob "$ARTIFACT_GLOB" \
        --renderer default \
        --cache-size "$CACHE_SIZE" \
        --host 0.0.0.0 --port "$PORT" \
    >/dev/null

cid="$(docker inspect --format '{{.Id}}' "$CONTAINER_NAME")"
printf '%s %s\n' "$cid" "$PORT" > "$STATE_FILE"

# Poll for readiness. `Serving on` is the canonical readiness marker
# emitted by `nre serve-grpc` once the gRPC server is accepting
# connections.
log "waiting for 'Serving on' (timeout ${BOOT_TIMEOUT}s)..."
deadline=$(( $(date +%s) + BOOT_TIMEOUT ))
while (( $(date +%s) < deadline )); do
    if docker logs "$CONTAINER_NAME" 2>&1 | grep -q "Serving on"; then
        log "READY  container=$CONTAINER_NAME  port=$PORT  state=$STATE_FILE"
        exit 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        log "container died during boot; last 50 log lines:"
        docker logs --tail 50 "$CONTAINER_NAME" >&2 || true
        exit 1
    fi
    sleep 2
done

log "TIMEOUT waiting for serve-grpc readiness; last 50 log lines:"
docker logs --tail 50 "$CONTAINER_NAME" >&2 || true
exit 1
