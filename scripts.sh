#!/bin/bash
# Convenience scripts for Model_Finetuning project
# Usage: ./scripts.sh <command>

set -e  # Exit on error

case "$1" in
    # Training commands
    "train")
        python scripts/training/sft_qlora_8bit.py
        ;;
    "train-log")
        mkdir -p logs
        python scripts/training/sft_qlora_8bit.py 2>&1 | tee -a "logs/run_$(date +%F_%H-%M).log"
        ;;
    
    # Tmux training session management
    "tmux-train")
        # Check if session already exists
        if tmux has-session -t train 2>/dev/null; then
            echo "⚠️  Training session 'train' already exists!"
            read -p "🔄 Kill existing session and create new one? (y/n): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                tmux kill-session -t train
                echo "💀 Killed existing session"
            else
                echo "📎 Use './scripts.sh tmux-attach' to connect to existing session"
                exit 0
            fi
        fi
        
        echo "🚀 Starting tmux training session..."
        tmux new-session -d -s train "mkdir -p logs; python scripts/training/sft_qlora_8bit.py 2>&1 | tee -a logs/run_\$(date +%F_%H-%M).log"
        tmux new-window -t train: -n gpu "watch -n 1 nvidia-smi"
        tmux new-window -t train: -n tb "tensorboard --logdir runs/qwen3b_8bit --port 6006 --bind_all 2>&1 | tee -a logs/tb_\$(date +%F_%H-%M).log"
        echo "✅ Training session created successfully!"
        echo "📊 Windows: 0=training, 1=gpu, 2=tensorboard"
        echo "🌐 TensorBoard: http://localhost:6006"
        echo ""
        read -p "🚀 Attach to training session now? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Attaching to session..."
            tmux attach -t train
        else
            echo "Session running in background. Use './scripts.sh tmux-attach' to connect later."
        fi
        ;;
    "tmux-attach")
        tmux attach -t train
        ;;
    "tmux-kill")
        tmux kill-session -t train
        ;;
    "tmux-status")
        tmux list-sessions | grep train || echo "No training session active"
        ;;
    
    # Quick monitoring
    "gpu")
        nvidia-smi
        ;;
    "logs")
        tail -f logs/run_*.log | head -50
        ;;
    "tensorboard")
        tensorboard --logdir runs/qwen3b_8bit --port 6006
        ;;
    
    # Help for scripts requiring arguments
    "compare-help")
        python scripts/testing/compare_baseline_vs_lora.py --help
        ;;
    "data-prep-help")
        python scripts/data_prep/merge_and_split.py --help
        ;;
    "claude-to-sft-help")
        python scripts/data_prep/claude_logs/claude2sft.py --help
        ;;
    "repo-to-fim-help")
        python scripts/data_prep/actual_code/repo2fim.py --help
        ;;
    
    # Show examples for complex commands
    "examples")
        echo "=== Data Preparation Examples ==="
        echo "python scripts/data_prep/claude_logs/claude2sft.py --src data/claude_logs/ --out data/train.clean.jsonl"
        echo "python scripts/data_prep/actual_code/repo2fim.py --repo /path/to/repo --out data/train.fim.jsonl --per_file 6 --max_len 1600"
        echo "python scripts/data_prep/merge_and_split.py data/train1.jsonl data/train2.jsonl --name combined --eval_frac 0.15"
        echo ""
        echo "=== Model Comparison Example ==="
        echo "python scripts/testing/compare_baseline_vs_lora.py --prompt \"Which part of my source code includes the logic for how users like or pass on properties?\" --ckpt outputs/qwen3b_lora_8bit/checkpoint-650"
        ;;
    
    # Help
    "help"|"--help"|"-h"|"")
        echo "Model Finetuning Convenience Scripts"
        echo ""
        echo "Training:"
        echo "  ./scripts.sh train              # Start 8-bit training"
        echo "  ./scripts.sh train-log          # Training with logging"
        echo ""
        echo "Testing:"
        echo "  ./scripts.sh test-baseline      # Test baseline model"
        echo "  ./scripts.sh check-lengths      # Analyze dataset token lengths"
        echo ""
        echo "Tmux Session Management:"
        echo "  ./scripts.sh tmux-train         # Start training session (3 windows)"
        echo "  ./scripts.sh tmux-attach        # Attach to training session"
        echo "  ./scripts.sh tmux-status        # Check session status"
        echo "  ./scripts.sh tmux-kill          # Kill training session"
        echo ""
        echo "Monitoring:"
        echo "  ./scripts.sh gpu                # Check GPU status"
        echo "  ./scripts.sh logs               # View recent training logs"
        echo "  ./scripts.sh tensorboard        # Start TensorBoard"
        ;;
    
    *)
        echo "Unknown command: $1"
        echo "Use './scripts.sh help' to see available commands"
        exit 1
        ;;
esac
