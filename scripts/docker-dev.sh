#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/Ryan-Reese/nvalchemi-toolkit.git"
REPO_DIR="/tmp/nvalchemi-toolkit"
IMAGE="nvalchemi"
CONTAINER="nvalchemi"
PORT="${PORT:-8890}"

cmd_start() {
    # Clone or pull repo — if a previous run left root-owned files, wipe contents and re-clone
    if [ -d "$REPO_DIR" ] && ! git -C "$REPO_DIR" fetch origin 2>/dev/null; then
        docker run --rm -v "$REPO_DIR:$REPO_DIR" alpine sh -c "rm -rf $REPO_DIR/* $REPO_DIR/.*" 2>/dev/null || true
    fi
    if [ -d "$REPO_DIR/.git" ]; then
        git -C "$REPO_DIR" reset --hard origin/main
    else
        git clone "$REPO_URL" "$REPO_DIR"
    fi

    # Build image
    docker build -t "$IMAGE" "$REPO_DIR"

    # Remove stale container if it exists
    docker rm -f "$CONTAINER" 2>/dev/null || true

    # Run detached with GPU access and repo bind-mount
    docker run --gpus all --name "$CONTAINER" -d --network host \
        -v "$REPO_DIR:/nvalchemi-toolkit" \
        "$IMAGE" \
        jupyter lab --ip=0.0.0.0 --port="$PORT" --no-browser --allow-root

    echo "Container started."
    sleep 3
    cmd_status
}

cmd_restart() {
    # Pull latest code into the bind-mounted repo
    docker exec "$CONTAINER" chown -R "$(id -u):$(id -g)" /nvalchemi-toolkit/.git
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" reset --hard origin/main
    # Restart the container to pick up changes
    docker restart "$CONTAINER"
    echo "Container restarted with latest changes."
    sleep 3
    cmd_status
}

cmd_rebuild() {
    # Pull latest and do a full image rebuild
    docker run --rm -v "$REPO_DIR:$REPO_DIR" alpine chown -R "$(id -u):$(id -g)" "$REPO_DIR/.git"
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" reset --hard origin/main
    docker rm -f "$CONTAINER" 2>/dev/null || true
    docker build -t "$IMAGE" "$REPO_DIR"
    docker run --gpus all --name "$CONTAINER" -d --network host \
        -v "$REPO_DIR:/nvalchemi-toolkit" \
        "$IMAGE" \
        jupyter lab --ip=0.0.0.0 --port="$PORT" --no-browser --allow-root
    echo "Image rebuilt and container started."
    sleep 3
    cmd_status
}

cmd_status() {
    echo "=== Container Status ==="
    docker ps --filter "name=$CONTAINER" --format "table {{.Names}}\t{{.Status}}"

    echo ""
    echo "=== Jupyter URL ==="
    docker logs "$CONTAINER" 2>&1 \
        | grep -oE 'http://127\.0\.0\.1:[0-9]+/lab\?token=[a-z0-9]+' \
        | tail -1 || echo "Jupyter not ready yet."
}

cmd_stop() {
    docker rm -f "$CONTAINER" 2>/dev/null || true
    echo "Container stopped and removed."
}

case "${1:-}" in
    restart) cmd_restart ;;
    rebuild) cmd_rebuild ;;
    status)  cmd_status ;;
    stop)    cmd_stop ;;
    *)       cmd_start ;;
esac
