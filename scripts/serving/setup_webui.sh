#!/bin/bash

# =============================================================================
# Open WebUI Setup Script for vLLM + Tailscale
# =============================================================================
# This script sets up Open WebUI accessible via Tailscale network
# Run in tmux: tmux new-session -s webui "./setup_webui.sh"
# Options: --verbose (show detailed output), --debug (full debugging)

set -e  # Exit on any error

# Resolve repository root regardless of where the script is invoked
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Parse arguments
VERBOSE=false
DEBUG=false
LOGFILE=""

for arg in "$@"; do
    case $arg in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --debug)
            DEBUG=true
            VERBOSE=true
            set -x  # Enable bash debugging
            LOGFILE="setup_debug_$(date +%Y%m%d_%H%M%S).log"
            exec > >(tee -a "$LOGFILE") 2>&1
            shift
            ;;
        *)
            # Keep other arguments for main function
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
VLLM_HOST="0.0.0.0"
VLLM_PORT="8000"
WEBUI_PORT="3000"
# Use absolute path for the model to avoid CWD issues
MODEL_PATH="$REPO_ROOT/outputs/qwen3b_merged"
MODEL_NAME="qwen-coder-3b"
START_TIME=$(date +%s)

# Logging functions with timestamps
get_timestamp() {
    date '+[%Y-%m-%d %H:%M:%S]'
}

log_info() {
    echo -e "${GREEN}$(get_timestamp) [INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}$(get_timestamp) [WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}$(get_timestamp) [ERROR]${NC} $1"
}

log_debug() {
    if [ "$DEBUG" = true ]; then
        echo -e "${CYAN}$(get_timestamp) [DEBUG]${NC} $1"
    fi
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${CYAN}$(get_timestamp) [VERBOSE]${NC} $1"
    fi
}

log_step() {
    echo -e "\n${BLUE}$(get_timestamp) === $1 ===${NC}"
    if [ "$DEBUG" = true ]; then
        log_debug "Step started at $(date)"
        log_debug "Environment: USER=$USER, HOME=$HOME, PWD=$PWD"
        log_debug "Available memory: $(free -h | grep '^Mem:' | awk '{print $7}') available"
        log_debug "Available disk: $(df -h . | tail -1 | awk '{print $4}') available"
    fi
}

log_step_complete() {
    local duration=$(($(date +%s) - $1))
    log_info "Step completed in ${duration}s"
}

# Get Tailscale IP
get_tailscale_ip() {
    log_debug "Checking Tailscale configuration..."
    if command -v tailscale &> /dev/null; then
        log_verbose "Tailscale command found, getting IP..."
        TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "")
        if [ -n "$TAILSCALE_IP" ]; then
            log_info "Tailscale IP detected: $TAILSCALE_IP"
            echo "$TAILSCALE_IP"
        else
            log_warn "Tailscale installed but not connected, using localhost only"
            echo "127.0.0.1"
        fi
    else
        log_warn "Tailscale not installed, using localhost only"
        echo "127.0.0.1"
    fi
}

# Check if vLLM is running
check_vllm() {
    log_step "Checking vLLM Server Status"
    local check_start=$(date +%s)
    
    log_verbose "Testing connection to http://127.0.0.1:$VLLM_PORT/health"
    if curl -s http://127.0.0.1:$VLLM_PORT/health > /dev/null; then
        log_info "✅ vLLM server is running on port $VLLM_PORT"
        
        # Check model availability
        log_verbose "Querying available models..."
        MODELS=$(curl -s http://127.0.0.1:$VLLM_PORT/v1/models)
        if echo "$MODELS" | grep -q "$MODEL_NAME"; then
            log_info "✅ Model '$MODEL_NAME' is loaded and available"
            log_step_complete $check_start
            return 0
        else
            log_warn "❌ Model '$MODEL_NAME' not found. Available models:"
            if [ "$VERBOSE" = true ]; then
                echo "$MODELS" | jq -r '.data[].id' 2>/dev/null || echo "$MODELS"
            else
                echo "$MODELS" | jq -r '.data[].id' 2>/dev/null | head -3 || echo "Unable to parse models"
            fi
            return 1
        fi
    else
        log_error "❌ vLLM server not responding on port $VLLM_PORT"
        log_debug "Check if vLLM process is running: ps aux | grep vllm"
        return 1
    fi
}

# Start vLLM server
start_vllm() {
    log_step "Starting vLLM Server"
    local start_time=$(date +%s)
    
    # First check if already running
    if check_vllm; then
        log_info "✅ vLLM already running, skipping start"
        return 0
    fi
    
    # Kill existing vLLM processes
    log_info "🛑 Stopping existing vLLM processes..."
    local existing_pids=$(pgrep -f "vllm serve" || true)
    if [ -n "$existing_pids" ]; then
        log_verbose "Found existing vLLM PIDs: $existing_pids"
        pkill -f "vllm serve" || true
        log_info "Waiting 3 seconds for graceful shutdown..."
        sleep 3
    else
        log_verbose "No existing vLLM processes found"
    fi
    
    # Check if model exists
    log_verbose "Checking model path: $MODEL_PATH"
    if [ ! -d "$MODEL_PATH" ]; then
        log_error "❌ Model directory not found at $MODEL_PATH"
        log_error "Directory contents:"
        ls -la "$(dirname "$MODEL_PATH")" 2>/dev/null || log_error "Parent directory doesn't exist"
        log_error "💡 Run merge_lora.py first to create the merged model"
        exit 1
    fi
    
    # Show model info
    local model_size=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1 || echo "unknown")
    log_info "📁 Model size: $model_size"
    log_verbose "Model files: $(ls "$MODEL_PATH" | wc -l) files"
    
    log_info "🚀 Starting vLLM server with network access..."
    
    # Start vLLM in background (use full path to ensure it works outside venv)
    local vllm_cmd=".venv/bin/vllm"
    if [ ! -f "$vllm_cmd" ]; then
        vllm_cmd="vllm"  # Fallback to PATH
    fi
    
    local full_cmd="$vllm_cmd serve $MODEL_PATH --host $VLLM_HOST --port $VLLM_PORT --max-model-len 2048 --gpu-memory-utilization 0.8 --max-num-seqs 16 --served-model-name $MODEL_NAME"
    log_verbose "Command: $full_cmd"
    
    nohup "$vllm_cmd" serve "$MODEL_PATH" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.8 \
        --max-num-seqs 16 \
        --served-model-name "$MODEL_NAME" \
        > vllm.log 2>&1 &
    
    VLLM_PID=$!
    echo $VLLM_PID > vllm.pid
    log_info "📡 vLLM starting with PID $VLLM_PID (logs: vllm.log)"
    
    # Wait for vLLM to be ready with enhanced progress tracking
    log_info "⏳ Waiting for vLLM to initialize (typically 30-60 seconds for model loading)..."
    log_info "🔄 Giving vLLM 10 seconds to start before health checks..."
    
    # Initial delay to allow vLLM to start initializing
    sleep 10
    
    local attempt=0
    local max_attempts=60
    
    for i in {1..60}; do
        attempt=$i
        local elapsed=$(($(date +%s) - start_time))
        
        # Check if vLLM process is still running before trying health check
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            log_error "❌ vLLM process (PID $VLLM_PID) has died"
            log_error "📋 Check vllm.log for error details (last 10 lines):"
            if [ -f vllm.log ]; then
                tail -10 vllm.log
            fi
            exit 1
        fi
        
        if check_vllm; then
            log_info "🎉 vLLM server ready! (started in ${elapsed}s after $attempt attempts)"
            log_step_complete $start_time
            return 0
        fi
        
        # Show progress every 5 attempts with better context
        if [ $((i % 5)) -eq 0 ]; then
            local status_msg="⏳ Attempt $i/$max_attempts (${elapsed}s elapsed)"
            
            # Show what vLLM is doing based on recent log entries
            if [ -f vllm.log ]; then
                local last_log=$(tail -1 vllm.log 2>/dev/null)
                if echo "$last_log" | grep -q "Loading.*checkpoint"; then
                    status_msg="$status_msg - Loading model weights..."
                elif echo "$last_log" | grep -q "Capturing.*graph"; then
                    status_msg="$status_msg - Optimizing CUDA graphs..."
                elif echo "$last_log" | grep -q "Optimizing CUDA graphs"; then
                    status_msg="$status_msg - Optimizing CUDA graphs..."
                elif echo "$last_log" | grep -q "Started server"; then
                    status_msg="$status_msg - Server starting up..."
                elif echo "$last_log" | grep -q "Memory profiling"; then
                    status_msg="$status_msg - Profiling GPU memory..."
                else
                    status_msg="$status_msg - Initializing..."
                fi
            fi
            
            log_info "$status_msg"
            
            if [ "$DEBUG" = true ] && [ -f vllm.log ]; then
                log_debug "Recent vLLM log entries:"
                tail -3 vllm.log | while read line; do log_debug "$line"; done
            fi
        fi
        
        sleep 2
    done
    
    local total_time=$(($(date +%s) - start_time))
    log_error "❌ vLLM failed to start within 2 minutes ($total_time seconds)"
    log_error "📋 Check vllm.log for details (last 20 lines):"
    if [ -f vllm.log ]; then
        tail -20 vllm.log
    else
        log_error "vllm.log file not found"
    fi
    log_error "💡 Try checking: GPU memory, CUDA drivers, model corruption"
    exit 1
}

# Setup Docker and Open WebUI
setup_webui() {
    log_step "Setting Up Open WebUI"
    local setup_start=$(date +%s)
    
    # Check Docker installation
    log_verbose "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        log_error "❌ Docker not found. Please install Docker first."
        log_error "💡 Install from: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check Docker daemon
    log_verbose "Checking Docker daemon status..."
    if ! docker info > /dev/null 2>&1; then
        log_error "❌ Docker daemon not running. Please start Docker."
        log_error "💡 Try: sudo systemctl start docker (Linux) or start Docker Desktop"
        exit 1
    fi
    
    local docker_version=$(docker --version | cut -d' ' -f3 | tr -d ',')
    log_info "🐳 Docker detected: $docker_version"
    
    # Get Tailscale IP
    TAILSCALE_IP=$(get_tailscale_ip)
    
    # Stop existing container
    log_info "🛑 Cleaning up existing Open WebUI container..."
    local existing_container=$(docker ps -a -q -f name=open-webui 2>/dev/null)
    if [ -n "$existing_container" ]; then
        log_verbose "Found existing container: $existing_container"
        docker stop open-webui 2>/dev/null || true
        docker rm open-webui 2>/dev/null || true
        log_info "✅ Removed existing container"
    else
        log_verbose "No existing Open WebUI container found"
    fi
    
    # Try different image sources with enhanced logging
    WEBUI_IMAGES=(
        "ghcr.io/open-webui/open-webui:ollama"
        "ghcr.io/open-webui/open-webui:main"
        "dyrnq/open-webui:latest"       # Another Docker Hub mirror
        "backplane/open-webui:0"         # Backplane alternative
    )
    
    WEBUI_IMAGE=""
    log_info "📥 Attempting to pull Open WebUI Docker image..."
    for i in "${!WEBUI_IMAGES[@]}"; do
        local img="${WEBUI_IMAGES[$i]}"
        local attempt=$((i + 1))
        log_info "🔄 Attempt $attempt/${#WEBUI_IMAGES[@]}: Pulling $img..."
        
        if [ "$VERBOSE" = true ]; then
            if docker pull "$img"; then
                WEBUI_IMAGE="$img"
                log_info "✅ Successfully pulled $img"
                break
            else
                log_warn "❌ Failed to pull $img"
            fi
        else
            if docker pull "$img" >/dev/null 2>&1; then
                WEBUI_IMAGE="$img"
                log_info "✅ Successfully pulled $img"
                break
            else
                log_warn "❌ Failed to pull $img"
                if [ "$DEBUG" = true ]; then
                    log_debug "Docker pull error for $img"
                fi
            fi
        fi
        
        # Show progress
        if [ $attempt -lt ${#WEBUI_IMAGES[@]} ]; then
            log_verbose "Trying next image source..."
        fi
    done
    
    if [ -z "$WEBUI_IMAGE" ]; then
        log_error "❌ Failed to pull any Open WebUI image after ${#WEBUI_IMAGES[@]} attempts"
        log_error "🔐 Try running: docker login ghcr.io"
        log_error "🌐 Or check your internet connection"
        log_error "📊 Docker registry status: $(curl -s -o /dev/null -w "%{http_code}" https://ghcr.io/ || echo "unreachable")"
        exit 1
    fi
    
    # Start Open WebUI with enhanced configuration
    log_info "🚀 Starting Open WebUI container..."
    log_info "🔗 Access URLs will be:"
    log_info "   📍 Local: http://127.0.0.1:$WEBUI_PORT"
    if [ "$TAILSCALE_IP" != "127.0.0.1" ]; then
        log_info "   🌐 Tailscale: http://$TAILSCALE_IP:$WEBUI_PORT"
    fi
    
    # Show Docker run command in verbose mode
    if [ "$VERBOSE" = true ]; then
        log_verbose "Docker run command:"
        log_verbose "docker run -d --name open-webui --network host \\"
        log_verbose "  -e OPENAI_API_BASE_URL=http://127.0.0.1:$VLLM_PORT/v1 \\"
        log_verbose "  -e WEBUI_NAME=\"Qwen Coder Assistant\" \\"
        log_verbose "  -e PORT=$WEBUI_PORT $WEBUI_IMAGE"
    fi
    
    docker run -d \
        --name open-webui \
        --add-host=host.docker.internal:host-gateway \
        -p ${WEBUI_PORT}:${WEBUI_PORT} \
        -e OPENAI_API_BASE_URL=http://host.docker.internal:$VLLM_PORT/v1 \
        -e OPENAI_API_KEY=not-needed \
        -e WEBUI_NAME="Qwen Coder Assistant" \
        -e WEBUI_AUTH=true \
        -e PORT=$WEBUI_PORT \
        -v open-webui-data:/app/backend/data \
        --restart unless-stopped \
        "$WEBUI_IMAGE"
    
    local container_id=$(docker ps -q -f name=open-webui)
    log_info "📦 Container started with ID: ${container_id:0:12}"
    
    # Wait for WebUI to start with enhanced progress tracking
    log_info "⏳ Waiting for Open WebUI to initialize..."
    local webui_attempt=0
    local max_webui_attempts=30
    
    for i in {1..30}; do
        webui_attempt=$i
        local elapsed=$(($(date +%s) - setup_start))
        
        if curl -s http://127.0.0.1:$WEBUI_PORT > /dev/null; then
            log_info "🎉 Open WebUI ready! (initialized in ${elapsed}s after $webui_attempt attempts)"
            log_step_complete $setup_start
            return 0
        fi
        
        # Show progress every 5 attempts
        if [ $((i % 5)) -eq 0 ]; then
            log_verbose "⏳ Attempt $i/$max_webui_attempts (${elapsed}s elapsed) - still waiting for WebUI..."
            if [ "$DEBUG" = true ]; then
                log_debug "Container status: $(docker ps --filter name=open-webui --format 'table {{.Status}}' | tail -1)"
            fi
        fi
        
        sleep 2
    done
    
    local total_time=$(($(date +%s) - setup_start))
    log_error "❌ Open WebUI failed to start within 1 minute ($total_time seconds)"
    log_error "📋 Container logs (last 15 lines):"
    docker logs --tail 15 open-webui 2>/dev/null || log_error "Could not retrieve container logs"
    log_error "🔍 Container status: $(docker ps -a --filter name=open-webui --format 'table {{.Status}}' | tail -1)"
    log_error "💡 Try: docker logs -f open-webui (in another terminal)"
    exit 1
}

# Create docker-compose file
create_compose() {
    log_step "Creating Docker Compose Configuration"
    local step_start=$(date +%s)
    
    TAILSCALE_IP=$(get_tailscale_ip)
    
    cat > docker-compose.yml << EOF
version: '3.8'

services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    extra_hosts:
      - "host.docker.internal:host-gateway"
    ports:
      - "${WEBUI_PORT}:${WEBUI_PORT}"
    environment:
      - OPENAI_API_BASE_URL=http://host.docker.internal:$VLLM_PORT/v1
      - OPENAI_API_KEY=not-needed
      - WEBUI_NAME=Qwen Coder Assistant
      - WEBUI_AUTH=true
      - PORT=$WEBUI_PORT
      - WEBUI_URL=http://$TAILSCALE_IP:$WEBUI_PORT
    volumes:
      - ./data:/app/backend/data
      - ./uploads:/app/backend/uploads
    restart: unless-stopped
    depends_on:
      - vllm-healthcheck
  
  vllm-healthcheck:
    image: curlimages/curl:latest
    container_name: vllm-healthcheck
    extra_hosts:
      - "host.docker.internal:host-gateway"
    command: ["sh", "-c", "until curl -f http://host.docker.internal:$VLLM_PORT/health; do echo 'Waiting for vLLM...'; sleep 5; done; echo 'vLLM is ready!'"]
EOF

    log_info "Created docker-compose.yml"
    log_step_complete $step_start
}

# Create management scripts
create_scripts() {
    log_step "Creating Management Scripts"
    local step_start=$(date +%s)
    
    TAILSCALE_IP=$(get_tailscale_ip)
    
    # Start script
    cat > start_chat.sh << 'EOF'
#!/bin/bash
# Start the complete chat system

echo "🚀 Starting Qwen Coder Chat System..."

# Start vLLM
echo "📡 Starting vLLM server..."
scripts/serving/setup_webui.sh start_vllm_only

# Start WebUI
echo "🌐 Starting Web Interface..."
docker-compose up -d

echo "✅ System started!"
echo "Access at: http://$(tailscale ip -4 2>/dev/null || echo 127.0.0.1):3000"
EOF

    # Stop script
    cat > stop_chat.sh << 'EOF'
#!/bin/bash
# Stop the complete chat system

echo "🛑 Stopping Qwen Coder Chat System..."

# Stop WebUI
echo "🌐 Stopping Web Interface..."
docker-compose down

# Stop vLLM
echo "📡 Stopping vLLM server..."
pkill -f "vllm serve" || true

echo "✅ System stopped!"
EOF

    # Status script
    cat > status_chat.sh << 'EOF'
#!/bin/bash
# Check system status

echo "📊 Qwen Coder Chat System Status"
echo "================================"

# Check vLLM
echo -n "vLLM Server: "
if curl -s http://127.0.0.1:8000/health > /dev/null; then
    echo "✅ Running"
    echo "  Models: $(curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[].id' 2>/dev/null | tr '\n' ' ')"
else
    echo "❌ Not running"
fi

# Check WebUI
echo -n "Web Interface: "
if curl -s http://127.0.0.1:3000 > /dev/null; then
    echo "✅ Running"
    echo "  URL: http://$(tailscale ip -4 2>/dev/null || echo 127.0.0.1):3000"
else
    echo "❌ Not running"
fi

# Show URLs
echo ""
echo "🔗 Access URLs:"
echo "  Local: http://127.0.0.1:3000"
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null)
if [ -n "$TAILSCALE_IP" ]; then
    echo "  Tailscale: http://$TAILSCALE_IP:3000"
fi
EOF

    chmod +x start_chat.sh stop_chat.sh status_chat.sh
    log_info "Created management scripts: start_chat.sh, stop_chat.sh, status_chat.sh"
    log_step_complete $step_start
}

# Test connections
test_system() {
    log_step "Testing System"
    local step_start=$(date +%s)
    
    # Test vLLM
    log_info "Testing vLLM API..."
    if ! check_vllm; then
        log_error "vLLM test failed"
        return 1
    fi
    
    # Test WebUI
    log_info "Testing Web Interface..."
    if curl -s http://127.0.0.1:$WEBUI_PORT > /dev/null; then
        log_info "✅ Web Interface responding"
    else
        log_error "❌ Web Interface not responding"
        return 1
    fi
    
    # Test API integration
    log_info "Testing API integration..."
    RESPONSE=$(curl -s -X POST http://127.0.0.1:$VLLM_PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{ \
            "model": "'$MODEL_NAME'", \
            "messages": [{"role": "user", "content": "Hello, respond with just OK"}], \
            "max_tokens": 10 \
        }')
    
    if echo "$RESPONSE" | grep -q "choices"; then
        log_info "✅ API integration working"
    else
        log_warn "⚠️  API test failed, response: $RESPONSE"
    fi
    
    log_step_complete $step_start
}

# Show final status
show_status() {
    log_step "Setup Complete!"
    local total_time=$(($(date +%s) - START_TIME))
    
    TAILSCALE_IP=$(get_tailscale_ip)
    
    echo ""
    echo "🎉 Qwen Coder Chat System is ready! (Total setup time: ${total_time}s)"
    echo ""
    echo "📱 Access URLs:"
    echo "   📍 Local:     http://127.0.0.1:$WEBUI_PORT"
    if [ "$TAILSCALE_IP" != "127.0.0.1" ]; then
        echo "   🌐 Tailscale: http://$TAILSCALE_IP:$WEBUI_PORT"
    fi
    echo ""
    echo "🔧 Management Commands:"
    echo "   🚀 Start:  ./start_chat.sh"
    echo "   🛑 Stop:   ./stop_chat.sh"
    echo "   📊 Status: ./status_chat.sh"
    echo ""
    echo "📁 Generated Files:"
    echo "   📋 docker-compose.yml - Docker orchestration"
    echo "   📜 vllm.log          - vLLM server logs"
    echo "   🔢 vllm.pid          - vLLM process ID"
    if [ -n "$LOGFILE" ]; then
        echo "   🐛 $LOGFILE    - Debug session log"
    fi
    echo ""
    echo "🛠️ Troubleshooting Commands:"
    echo "   📋 vLLM logs:   tail -f vllm.log"
    echo "   🐳 WebUI logs:  docker logs -f open-webui"
    echo "   🔍 Full test:   ./status_chat.sh"
    echo "   🔄 Restart:     ./stop_chat.sh && ./start_chat.sh"
    echo ""
    echo "💡 Tips:"
    echo "   • First visit will require account creation"
    echo "   • Model name in WebUI: '$MODEL_NAME'"
    echo "   • For debugging: $0 --verbose or $0 --debug"
    if [ "$DEBUG" = true ] || [ "$VERBOSE" = true ]; then
        echo ""
        echo "🔧 Debug Info:"
        echo "   • Script run with: $([ "$DEBUG" = true ] && echo "DEBUG" || echo "VERBOSE") mode"
        echo "   • vLLM PID: $(cat vllm.pid 2>/dev/null || echo "not found")"
        echo "   • WebUI container: $(docker ps --filter name=open-webui --format '{{.Status}}' 2>/dev/null || echo "not found")"
        echo "   • Model size: $(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1 || echo "unknown")"
    fi
}

# Show usage information
show_usage() {
    echo "🚀 Qwen Coder Chat Setup Script"
    echo ""
    echo "USAGE:"
    echo "  $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "OPTIONS:"
    echo "  --verbose    Show detailed output during execution"
    echo "  --debug      Enable full debugging with timestamped log file"
    echo "  --help       Show this help message"
    echo ""
    echo "COMMANDS:"
    echo "  full              Complete setup (default) - vLLM + WebUI + scripts"
    echo "  start_vllm_only   Start only the vLLM server"
    echo "  start_webui_only  Start only the WebUI container"
    echo "  test              Test system connectivity and integration"
    echo "  status            Show current system status and URLs"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                           # Full setup with normal output"
    echo "  $0 --verbose                 # Full setup with detailed logging"
    echo "  $0 --debug                   # Full setup with debug log file"
    echo "  $0 start_vllm_only          # Start only vLLM server"
    echo "  $0 test                      # Test if everything is working"
    echo ""
    echo "TROUBLESHOOTING:"
    echo "  • vLLM logs:     tail -f vllm.log"
    echo "  • WebUI logs:    docker logs -f open-webui"
    echo "  • Full restart:  ./stop_chat.sh && ./start_chat.sh"
    echo ""
}

# Main execution
main() {
    # Handle help flag
    for arg in "$@"; do
        case $arg in
            --help|-h|help)
                show_usage
                exit 0
                ;;
        esac
    done
    
    log_step "Qwen Coder Chat Setup"
    log_info "🕐 Started at $(date)"
    if [ "$DEBUG" = true ]; then
        log_debug "Debug mode enabled - full logging to: $LOGFILE"
    elif [ "$VERBOSE" = true ]; then
        log_verbose "Verbose mode enabled - detailed output"
    fi
    
    # Filter out our flags to get the actual command
    local command="full"
    for arg in "$@"; do
        case $arg in
            --verbose|--debug|--help|-h)
                # Skip our flags
                ;;
            *)
                command="$arg"
                break
                ;;
        esac
    done
    
    log_verbose "Executing command: $command"
    
    case "$command" in
        "start_vllm_only")
            log_info "🎯 Mode: vLLM server only"
            start_vllm
            ;;
        "start_webui_only")
            log_info "🎯 Mode: WebUI container only"
            setup_webui
            ;;
        "test")
            log_info "🎯 Mode: System testing"
            test_system
            ;;
        "status")
            log_info "🎯 Mode: Status display"
            show_status
            ;;
        "full"|"")
            log_info "🎯 Mode: Complete setup"
            start_vllm
            setup_webui
            create_compose
            create_scripts
            test_system
            show_status
            ;;
        *)
            log_error "❌ Unknown command: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
    
    local final_time=$(($(date +%s) - START_TIME))
    log_info "🏁 Script completed in ${final_time} seconds"
}

# Trap to cleanup on exit
cleanup() {
    log_info "Cleaning up..."
    exit 0
}

trap cleanup EXIT

# Run main function
main "$@"
