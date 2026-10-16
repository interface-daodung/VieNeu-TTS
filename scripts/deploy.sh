#!/usr/bin/env bash
# Deploy VieNeu-TTS on a remote SSH server (Linux).
#
# Usage:
#   ssh user@server "bash -s" < scripts/deploy.sh
#
# Or run directly on the server:
#   bash scripts/deploy.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== VieNeu-TTS deploy ==="
echo "Project: $PROJECT_DIR"

# ── 1. Python environment ────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[setup] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "[setup] Syncing dependencies..."
uv sync

# ── 2. Download INT8 model (with retry) ────────────────────────────────────
if [ -f "scripts/download_model.py" ]; then
    echo "[model] Checking/downloading INT8 model..."
    MAX_RETRIES=3
    for i in $(seq 1 $MAX_RETRIES); do
        if uv run python scripts/download_model.py; then
            echo "[model] Model ready."
            break
        fi
        if [ "$i" -eq "$MAX_RETRIES" ]; then
            echo "[model] ERROR: download failed after $MAX_RETRIES attempts."
            echo "[model] Try manual: uv run python scripts/download_model.py"
            exit 1
        fi
        echo "[model] Retry $i/$MAX_RETRIES in 10s..."
        sleep 10
    done

    # Integrity check
    echo "[model] Verifying model integrity..."
    if ! uv run python scripts/download_model.py --check; then
        echo "[model] WARNING: integrity check failed. Re-downloading..."
        rm -rf model/onnx_int8 model/codec
        uv run python scripts/download_model.py
    fi
else
    echo "[model] download_model.py not found — skipping."
fi

# ── 3. Deploy via systemd (preferred) or nohup fallback ──────────────────
TOKEN_LINE=""
if [ -f ".env" ] && grep -q "^VIENEU_TOKEN=" .env; then
    TOKEN_VAL=$(grep "^VIENEU_TOKEN=" .env | cut -d= -f2- | tr -d '"' | tr -d "'")
    if [ -n "$TOKEN_VAL" ]; then
        TOKEN_LINE="Environment=VIENEU_TOKEN=$TOKEN_VAL"
    fi
fi

if command -v systemctl &>/dev/null && [ "$(id -u)" -eq 0 ]; then
    echo "[server] Installing systemd service..."
    SERVICE_FILE="/etc/systemd/system/vieneu-tts.service"
    sed -e "s|%USER%|$(whoami)|g" \
        -e "s|%WORKDIR%|$PROJECT_DIR|g" \
        -e "s|%PYTHON%|$PROJECT_DIR/.venv/bin/python|g" \
        -e "s|%ENV_FILE%|${TOKEN_LINE}|g" \
        scripts/vieneu-tts.service | sudo tee "$SERVICE_FILE" > /dev/null

    sudo systemctl daemon-reload
    sudo systemctl enable vieneu-tts
    sudo systemctl restart vieneu-tts
    sleep 2
    sudo systemctl status vieneu-tts --no-pager || true
    echo "[server] Service active. Logs: sudo journalctl -u vieneu-tts -f"
else
    echo "[server] systemd not available (or not root) — using nohup..."

    # Kill existing instance
    if [ -f ".server.pid" ]; then
        OLD_PID=$(cat .server.pid || true)
        if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
            echo "[server] Killing old server (PID $OLD_PID)..."
            kill "$OLD_PID" 2>/dev/null || true
            sleep 2
        fi
        rm -f .server.pid
    fi

    # Load token from .env if present
    EXTRA_ENV=""
    if [ -f ".env" ] && grep -q "^VIENEU_TOKEN=" .env; then
        export VIENEU_TOKEN=$(grep "^VIENEU_TOKEN=" .env | cut -d= -f2- | tr -d '"' | tr -d "'")
    fi

    nohup uv run python -m apps.web_stream >> server.log 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > .server.pid
    echo "[server] Started PID $SERVER_PID"
    echo "[server] Logs: tail -f server.log"
fi

echo ""
echo "=== Deploy complete ==="
echo "Check: curl -H \"Authorization: Bearer \$VIENEU_TOKEN\" http://127.0.0.1:8001/health"
