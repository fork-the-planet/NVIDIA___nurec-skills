#!/usr/bin/env bash
# session_teardown.sh -- stop + remove the warm `nre serve-grpc`
# container booted by session_warm_server.sh.
#
# Idempotent: noop if the container or state file is already gone.
#
# Verifies that `docker ps -a` no longer reports the container before
# returning. `docker rm -f` returns as soon as it has issued SIGKILL,
# but the daemon may take up to a few seconds to actually reap the
# container record on a busy host. Returning before reap means a
# subsequent `session_warm_server.sh` boot can race with a leftover
# container of the same name and fail with "name already in use".
#
# Environment overrides:
#   NRE_GRPC_CONTAINER -- Container name (default nre-grpc-session)
#   NRE_GRPC_TEARDOWN_TIMEOUT -- Max seconds to wait for the container
#                                to actually disappear from `docker ps -a`
#                                (default 15).

set -euo pipefail

CONTAINER_NAME="${NRE_GRPC_CONTAINER:-nre-grpc-session}"
STATE_FILE="/tmp/nre-grpc-session.state"
TEARDOWN_TIMEOUT="${NRE_GRPC_TEARDOWN_TIMEOUT:-15}"

log() {
    printf '[teardown] %s\n' "$*" >&2
}

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    log "stopping + removing container '$CONTAINER_NAME'"
    docker rm -f "$CONTAINER_NAME" >/dev/null

    deadline=$(( $(date +%s) + TEARDOWN_TIMEOUT ))
    while (( $(date +%s) < deadline )); do
        if ! docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
            break
        fi
        sleep 1
    done
    if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        log "WARN: container '$CONTAINER_NAME' still listed after ${TEARDOWN_TIMEOUT}s"
        log "  status: $(docker ps -a --filter name="^${CONTAINER_NAME}$" --format '{{.Status}}')"
        log "  next boot may collide; rerun teardown or 'docker rm -f $CONTAINER_NAME' manually"
        exit 1
    fi
    log "container '$CONTAINER_NAME' confirmed gone"
else
    log "no container named '$CONTAINER_NAME'; nothing to stop"
fi

if [[ -f "$STATE_FILE" ]]; then
    rm -f "$STATE_FILE"
fi

log "DONE"
