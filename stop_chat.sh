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
