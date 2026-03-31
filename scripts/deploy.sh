#!/usr/bin/env bash
set -euo pipefail

LOCAL_PORT="${LOCAL_PORT:-8890}"
REMOTE_PORT="${REMOTE_PORT:-8890}"
STATE_FILE="/tmp/nvalchemi-deploy.env"
REMOTE_SCRIPT="/tmp/docker-dev.sh"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    cat <<'EOF'
Usage: deploy.sh <command> [args]

Commands:
  setup  <login-host> <compute-node>   First-time: build, start container, open tunnel
  restart                               Pull latest code and restart container
  pull-changes                          Sync remote Jupyter edits to local working directory
  status                                Show Jupyter URL with token
  stop                                  Close tunnel and stop container

Environment variables:
  LOCAL_PORT    Local port for Jupyter (default: 8890)
  REMOTE_PORT   Remote port for Jupyter (default: 8890)
EOF
    exit 1
}

save_state() {
    cat > "$STATE_FILE" <<EOF
LOGIN_HOST=$1
COMPUTE_NODE=$2
EOF
}

load_state() {
    if [ ! -f "$STATE_FILE" ]; then
        echo "No active deployment. Run 'deploy.sh setup <login-host> <compute-node>' first."
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$STATE_FILE"
}

remote_exec() {
    ssh -J "$LOGIN_HOST" "$COMPUTE_NODE" "$@"
}

open_tunnel() {
    close_tunnel 2>/dev/null || true
    ssh -f -N -o ExitOnForwardFailure=yes -J "$LOGIN_HOST" -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "$COMPUTE_NODE"
    sleep 1
    if lsof -ti:"${LOCAL_PORT}" > /dev/null 2>&1; then
        echo "Tunnel open: localhost:${LOCAL_PORT} -> ${COMPUTE_NODE}:${REMOTE_PORT} via ${LOGIN_HOST}"
    else
        echo "Failed to open SSH tunnel."
        return 1
    fi
}

close_tunnel() {
    local pid
    pid=$(lsof -ti:"${LOCAL_PORT}" 2>/dev/null) || true
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        echo "Tunnel closed."
    fi
}

cmd_setup() {
    [ -z "${1:-}" ] || [ -z "${2:-}" ] && usage
    LOGIN_HOST="$1"
    COMPUTE_NODE="$2"
    save_state "$LOGIN_HOST" "$COMPUTE_NODE"

    echo "Copying docker-dev.sh to ${COMPUTE_NODE}..."
    scp -o ProxyJump="$LOGIN_HOST" "$SCRIPT_DIR/docker-dev.sh" "${COMPUTE_NODE}:/tmp/docker-dev.sh"

    echo "Building and starting container on ${COMPUTE_NODE}..."
    remote_exec "PORT=${REMOTE_PORT} bash /tmp/docker-dev.sh"

    echo "Opening SSH tunnel..."
    open_tunnel

    echo ""
    echo "Setup complete. Access JupyterLab at http://localhost:${LOCAL_PORT}"
}

cmd_restart() {
    load_state
    echo "Restarting container on ${COMPUTE_NODE}..."
    remote_exec "bash ${REMOTE_SCRIPT} restart"
}

cmd_status() {
    load_state
    remote_exec "bash ${REMOTE_SCRIPT} status"

    echo ""
    if lsof -ti:"${LOCAL_PORT}" > /dev/null 2>&1; then
        echo "Tunnel: active (localhost:${LOCAL_PORT})"
    else
        echo "Tunnel: inactive"
    fi
}

cmd_pull_changes() {
    load_state
    echo "Pulling changes from remote..."
    remote_exec "cd /tmp/nvalchemi-toolkit && git diff" > /tmp/nvalchemi-remote.patch
    if [ -s /tmp/nvalchemi-remote.patch ]; then
        git apply /tmp/nvalchemi-remote.patch
        rm -f /tmp/nvalchemi-remote.patch
        echo "Changes applied locally. Review with 'git diff', then commit and push."
    else
        rm -f /tmp/nvalchemi-remote.patch
        echo "No uncommitted changes on remote."
    fi
}

cmd_stop() {
    load_state
    close_tunnel
    echo "Stopping container on ${COMPUTE_NODE}..."
    remote_exec "bash ${REMOTE_SCRIPT} stop"
    rm -f "$STATE_FILE"
}

case "${1:-}" in
    setup)        shift; cmd_setup "$@" ;;
    restart)      cmd_restart ;;
    pull-changes) cmd_pull_changes ;;
    status)       cmd_status ;;
    stop)         cmd_stop ;;
    *)            usage ;;
esac
