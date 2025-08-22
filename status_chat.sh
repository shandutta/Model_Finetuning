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
