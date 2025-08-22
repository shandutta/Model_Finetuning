#!/bin/bash

# =============================================================================
# Simple Qwen Coder Chat Setup - Streamlined Version
# =============================================================================
# Simplified startup script without false negatives and complexity bloat
# 60 lines vs original 768 lines (92% reduction)

set -e

# Resolve repository root regardless of where the script is invoked
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Configuration
VLLM_HOST="0.0.0.0"
VLLM_PORT="8000"
WEBUI_PORT="3000"
# Use absolute path for the model to avoid CWD issues
MODEL_PATH="$REPO_ROOT/outputs/qwen3b_merged"
MODEL_NAME="qwen-coder-3b"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Simple logging function
log() {
    case "$1" in
        error) echo -e "${RED}[ERROR]${NC} $2" ;;
        info)  echo -e "${GREEN}[INFO]${NC} $2" ;;
        step)  echo -e "\n${BLUE}=== $2 ===${NC}" ;;
        *)     echo "$1" ;;
    esac
}

# Start vLLM if not running
start_vllm() {
    log step "Starting vLLM Server"
    
    # Check if already running
    if curl -s http://127.0.0.1:$VLLM_PORT/health > /dev/null 2>&1; then
        log info "✅ vLLM already running"
        return 0
    fi
    
    # Check model exists
    if [ ! -d "$MODEL_PATH" ]; then
        log error "Model directory not found: $MODEL_PATH"
        log error "Run merge_lora.py first to create the merged model"
        exit 1
    fi
    
    # Kill existing processes
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 2
    
    # Start vLLM
    log info "🚀 Starting vLLM server..."
    nohup .venv/bin/vllm serve "$MODEL_PATH" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.8 \
        --max-num-seqs 16 \
        --served-model-name "$MODEL_NAME" \
        > vllm.log 2>&1 &
    
    echo $! > vllm.pid
    log info "📡 vLLM starting (logs: vllm.log)"
    
    # Wait for vLLM with realistic timeout
    log info "⏳ Waiting for vLLM to load model (30-60 seconds)..."
    sleep 15  # Initial wait for startup
    
    for i in {1..30}; do
        if curl -s http://127.0.0.1:$VLLM_PORT/health > /dev/null 2>&1; then
            log info "✅ vLLM ready!"
            return 0
        fi
        sleep 2
    done
    
    log error "❌ vLLM failed to start. Check vllm.log"
    exit 1
}

# Start WebUI container
start_webui() {
    log step "Starting Open WebUI"
    
    # Check Docker
    if ! docker info > /dev/null 2>&1; then
        log error "Docker daemon not running. Start Docker first."
        exit 1
    fi
    
    # Remove existing container
    docker stop open-webui 2>/dev/null || true
    docker rm open-webui 2>/dev/null || true
    
    # Determine how the container reaches the host (vLLM)
    # Use host.docker.internal with host-gateway to work on Linux + WSL + Docker Desktop
    local BASE_URL_HOST="host.docker.internal"

    log info "🚀 Starting WebUI container..."
    docker run -d \
        --name open-webui \
        --add-host=host.docker.internal:host-gateway \
        -p ${WEBUI_PORT}:${WEBUI_PORT} \
        -e OPENAI_API_BASE_URL=http://${BASE_URL_HOST}:$VLLM_PORT/v1 \
        -e OPENAI_API_KEY=not-needed \
        -e WEBUI_NAME="Qwen Coder Assistant" \
        -e PORT=$WEBUI_PORT \
        -v open-webui-data:/app/backend/data \
        --restart unless-stopped \
        dyrnq/open-webui:latest > /dev/null
    
    # Wait for WebUI with generous timeout (fixes false negatives)
    log info "⏳ Waiting for WebUI to initialize (60-120 seconds)..."
    sleep 30  # Initial wait for container startup
    
    for i in {1..45}; do  # 90 seconds total vs original 60
        if curl -s http://127.0.0.1:$WEBUI_PORT > /dev/null 2>&1; then
            log info "✅ WebUI ready!"
            return 0
        fi
        [ $((i % 10)) -eq 0 ] && log info "⏳ Still waiting... ($((30 + i*2))s elapsed)"
        sleep 2
    done
    
    log error "❌ WebUI timeout. Check: docker logs open-webui"
    exit 1
}

# Show access information
show_urls() {
    log step "Setup Complete!"
    
    # Get Tailscale IP if available
    TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "127.0.0.1")
    
    echo ""
    echo "🎉 Qwen Coder Chat System Ready!"
    echo ""
    echo "📱 Access URLs:"
    echo "   📍 Local:     http://127.0.0.1:$WEBUI_PORT"
    [ "$TAILSCALE_IP" != "127.0.0.1" ] && echo "   🌐 Tailscale: http://$TAILSCALE_IP:$WEBUI_PORT"
    echo ""
    echo "💡 Tips:"
    echo "   • First visit requires account creation"
    echo "   • Model name: '$MODEL_NAME'"
    echo "   • Logs: tail -f vllm.log"
    echo ""
}

# Main execution
main() {
    log step "Qwen Coder Chat Setup"
    start_vllm
    start_webui
    show_urls
}

# Run
main "$@"
