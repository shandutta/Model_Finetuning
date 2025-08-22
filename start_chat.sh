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
