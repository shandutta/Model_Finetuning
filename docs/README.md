# Model Fine-tuning Setup & Monitoring Guide

## Tmux Training Session Setup

Start a complete training environment with monitoring in separate tmux windows:

```bash
# 0) from your project root
source .venv/bin/activate

# 1) start a detached session with TRAINING in window 0
tmux new-session -d -s train "mkdir -p logs; python scripts/training/sft_qlora_8bit.py 2>&1 | tee -a logs/run_$(date +%F_%H-%M).log"
tmux new-window -t train: -n run "python scripts/training/sft_qlora_8bit.py 2>&1 | tee -a logs/run_$(date +%F_%H-%M).log"

# 1a) Optional, delete old run logs from Tensorboard
rm -rf runs/qwen3b_8bit/*

# 2) add GPU monitoring window
tmux new-window -t train: -n gpu "watch -n 1 nvidia-smi"

# 3) add TensorBoard window (phone-friendly bind)
tmux new-window -t train: -n tb "tensorboard --logdir runs/qwen3b_8bit --port 6006 --bind_all 2>&1 | tee -a logs/tb_$(date +%F_%H-%M).log"

# 4) attach to session
tmux attach -t train
```

### Tmux Navigation

**Basic Controls:**

- Switch windows: `Ctrl+b` then `0`/`1`/`2` or `Ctrl+b` then `n`/`p`
- Detach: `Ctrl+b` then `d`
- Reattach: `tmux attach -t train`

**From Normal Shell (not in tmux):**

```bash
# See what windows you have
tmux list-windows -t train
# e.g. 0: run, 1: gpu, 2: tb

# Attach and go straight to specific window (e.g. training window 0)
tmux attach -t train \; select-window -t train:0
```

**Quality of Life Tips:**

```bash
# Rename windows for clarity
tmux rename-window -t train:0 run

# Jump to window by name
tmux select-window -t train:run
```

## Remote Access via Tailscale

**Tailscale IP (WSL):** 100.84.236.65

### SSH from Phone

```bash
ssh shan@100.84.236.65
tmux attach -t train
```

### Remote TensorBoard Access

Open in browser: `http://100.84.236.65:6006`

## Monitoring Details

### GPU Monitoring

Monitor GPU usage during training:

```bash
watch -n 1 nvidia-smi
```

### TensorBoard Metrics

Access via browser to monitor:

- Training/eval loss curves
- Learning rate schedule  
- Gradient norms
- Training speed (steps/sec)

**Note:** Training logs every step (`logging_steps=1`) for real-time updates.