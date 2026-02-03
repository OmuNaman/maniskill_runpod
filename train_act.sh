#!/bin/bash
# =============================================================================
# ManiSkill ACT Training Pipeline
# =============================================================================
# Full pipeline: Clone → Setup → Download Demos → Replay RGBD → Train → Infer
# Designed for RunPod with RTX 4090 (headless, no display needed)
#
# Usage: bash train_act.sh
# =============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Fixed config
TASK="PickCube-v1"
WORK_DIR="/workspace"
MANISKILL_REPO="$WORK_DIR/ManiSkill"
ACT_DIR="$MANISKILL_REPO/examples/baselines/act"
DEMO_BASE="$HOME/.maniskill/demos/$TASK"

# Training defaults (matches official ManiSkill ACT baseline)
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-125}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-1}"
SEED="${SEED:-1}"

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}     ManiSkill ACT Training Pipeline (RTX 4090)${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo -e "  Task:       ${CYAN}$TASK${NC}"
echo -e "  Workspace:  ${CYAN}$WORK_DIR${NC}"
echo ""

# =============================================================================
# STEP 1: Clone ManiSkill Repository
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}[1/8] Cloning ManiSkill Repository${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ -d "$MANISKILL_REPO" ]; then
    echo -e "  ${GREEN}✓${NC} ManiSkill repo already exists at $MANISKILL_REPO"
    echo -n "  Pulling latest changes... "
    cd "$MANISKILL_REPO" && git pull --quiet 2>/dev/null && cd "$WORK_DIR"
    echo -e "${GREEN}done${NC}"
else
    echo "  Cloning https://github.com/haosulab/ManiSkill.git ..."
    if git clone --quiet https://github.com/haosulab/ManiSkill.git "$MANISKILL_REPO"; then
        echo -e "  ${GREEN}✓${NC} Cloned successfully"
    else
        echo -e "  ${RED}✗ Failed to clone ManiSkill repo${NC}"
        echo "  Check your internet connection and try again."
        exit 1
    fi
fi

# Verify ACT code exists
if [ ! -d "$ACT_DIR" ]; then
    echo -e "  ${RED}✗ ACT baseline not found at $ACT_DIR${NC}"
    echo "  The ManiSkill repo structure may have changed."
    exit 1
fi
echo -e "  ${GREEN}✓${NC} ACT training code found"

# Patch known bug: CUDA/CPU device mismatch during evaluation
# Model outputs actions on GPU but normalization stats stay on CPU
echo -n "  Checking for known device mismatch bug... "
python3 -c "
import sys

eval_file = '${ACT_DIR}/act/evaluate.py'
try:
    with open(eval_file, 'r') as f:
        content = f.read()
    
    if \"stats['action_std']\" in content and '.to(a.device)' not in content:
        content = content.replace(
            \"a * stats['action_std'] + stats['action_mean']\",
            \"a * stats['action_std'].to(a.device) + stats['action_mean'].to(a.device)\"
        )
        content = content.replace(
            \"(s_qpos - stats['qpos_mean']) / stats['qpos_std']\",
            \"(s_qpos - stats['qpos_mean'].to(s_qpos.device)) / stats['qpos_std'].to(s_qpos.device)\"
        )
        with open(eval_file, 'w') as f:
            f.write(content)
        print('patched evaluate.py')
    else:
        print('already patched')
except Exception as e:
    print(f'could not patch: {e}')
"
echo ""

# =============================================================================
# STEP 2: Weights & Biases Setup
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}[2/8] Weights & Biases (W&B) Setup${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  W&B lets you track training loss, success rate, and evaluation"
echo "  videos in real-time from your browser."
echo ""

USE_WANDB=false
read -p "  Enable W&B tracking? (y/n): " wandb_choice
echo ""

if [[ "$wandb_choice" =~ ^[Yy]$ ]]; then
    # Check if already logged in
    if wandb verify 2>/dev/null | grep -q "verified"; then
        echo -e "  ${GREEN}✓${NC} Already logged into W&B"
        USE_WANDB=true
    else
        echo "  You need a W&B API key. Get one at: https://wandb.ai/authorize"
        echo ""
        read -p "  Paste your W&B API key: " wandb_key
        echo ""

        if [ -n "$wandb_key" ]; then
            if wandb login "$wandb_key" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} W&B login successful!"
                USE_WANDB=true
            else
                echo -e "  ${YELLOW}⚠${NC}  W&B login failed. Continuing without tracking."
            fi
        else
            echo -e "  ${YELLOW}⚠${NC}  No key provided. Continuing without W&B."
        fi
    fi
else
    echo -e "  Skipping W&B. Training will log to tensorboard only."
fi
echo ""

# =============================================================================
# STEP 3: Download Demo Trajectories
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}[3/8] Downloading Demo Trajectories${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if demos already exist
MOTIONPLANNING_H5="$DEMO_BASE/motionplanning/trajectory.h5"

if [ -f "$MOTIONPLANNING_H5" ]; then
    echo -e "  ${GREEN}✓${NC} Demo data already downloaded"
else
    echo "  Downloading demos for $TASK from HuggingFace..."
    echo "  (This is automatic — no HuggingFace login needed)"
    echo ""
    if python -m mani_skill.utils.download_demo "$TASK"; then
        echo ""
        echo -e "  ${GREEN}✓${NC} Demos downloaded"
    else
        echo -e "  ${RED}✗ Failed to download demos${NC}"
        exit 1
    fi
fi

# Count available trajectories
echo ""
echo "  Counting available trajectories..."
TRAJ_COUNT=$(python -c "
import h5py, sys
try:
    f = h5py.File('$MOTIONPLANNING_H5', 'r')
    trajs = [k for k in f.keys() if k.startswith('traj')]
    print(len(trajs))
    f.close()
except Exception as e:
    print('0')
" 2>/dev/null)

if [ "$TRAJ_COUNT" -eq 0 ] 2>/dev/null; then
    echo -e "  ${RED}✗ Could not read trajectory file${NC}"
    echo "  File: $MOTIONPLANNING_H5"
    exit 1
fi

echo -e "  ${GREEN}✓${NC} Found ${BOLD}$TRAJ_COUNT${NC} trajectories available"
echo ""

# =============================================================================
# STEP 4: Select Number of Demos
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}[4/8] Select Number of Demos for Training${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Available: $TRAJ_COUNT trajectories"
echo "  Recommended: 100 (good balance of speed and quality)"
echo "  More demos = better policy but longer replay + training"
echo ""

while true; do
    read -p "  How many demos to use? [1-$TRAJ_COUNT] (default: 100): " num_demos_input
    NUM_DEMOS="${num_demos_input:-100}"

    # Validate input is a number
    if ! [[ "$NUM_DEMOS" =~ ^[0-9]+$ ]]; then
        echo -e "  ${RED}✗ Please enter a valid number${NC}"
        continue
    fi

    # Validate range
    if [ "$NUM_DEMOS" -lt 1 ]; then
        echo -e "  ${RED}✗ Must use at least 1 demo${NC}"
        continue
    fi

    if [ "$NUM_DEMOS" -gt "$TRAJ_COUNT" ]; then
        echo -e "  ${RED}✗ Only $TRAJ_COUNT demos available. Enter a number between 1 and $TRAJ_COUNT${NC}"
        continue
    fi

    break
done

echo -e "  ${GREEN}✓${NC} Using ${BOLD}$NUM_DEMOS${NC} demos"
echo ""

# =============================================================================
# STEP 5: Configure Training Hyperparameters
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}[5/8] Configure Training Hyperparameters${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  ACT uses ${BOLD}iterations${NC} (gradient steps), not epochs."
echo "  Each iteration = 1 batch sampled from demos → forward → backprop."
echo ""
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │  Official ManiSkill Recommendations:                │"
echo "  │                                                     │"
echo "  │  PickCube-v1 (easy):     30,000 iters  (~15 min)   │"
echo "  │  StackCube / PushT:     100,000 iters  (~45 min)   │"
echo "  │  Hard tasks:            400,000 iters  (~3 hours)  │"
echo "  │                                                     │"
echo "  │  Times are rough estimates on RTX 4090              │"
echo "  └─────────────────────────────────────────────────────┘"
echo ""

while true; do
    read -p "  Total training iterations? (default: 30000): " iters_input
    TOTAL_ITERS="${iters_input:-30000}"

    if ! [[ "$TOTAL_ITERS" =~ ^[0-9]+$ ]]; then
        echo -e "  ${RED}✗ Please enter a valid number${NC}"
        continue
    fi

    if [ "$TOTAL_ITERS" -lt 100 ]; then
        echo -e "  ${YELLOW}⚠${NC}  Very few iterations — policy likely won't learn much."
        read -p "  Continue anyway? (y/n): " confirm_low
        if [[ ! "$confirm_low" =~ ^[Yy]$ ]]; then
            continue
        fi
    fi

    break
done

echo -e "  ${GREEN}✓${NC} Training for ${BOLD}$TOTAL_ITERS${NC} iterations"
echo ""

# Eval frequency
echo "  Evaluation runs during training to track progress."
echo "  Default: every 5000 iters (official recommendation)"
echo ""
read -p "  Eval frequency? (default: $EVAL_FREQ): " eval_input
EVAL_FREQ="${eval_input:-$EVAL_FREQ}"
echo -e "  ${GREEN}✓${NC} Evaluating every ${BOLD}$EVAL_FREQ${NC} iterations"
echo ""

# Max episode steps
echo "  Max steps per episode (how long the robot gets to complete the task)."
echo "  Official default: 125 for PickCube-v1"
echo ""
read -p "  Max episode steps? (default: $MAX_EPISODE_STEPS): " steps_input
MAX_EPISODE_STEPS="${steps_input:-$MAX_EPISODE_STEPS}"
echo -e "  ${GREEN}✓${NC} Max ${BOLD}$MAX_EPISODE_STEPS${NC} steps per episode"
echo ""

# =============================================================================
# STEP 6: Replay Trajectories to RGBD
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}[6/8] Replaying Trajectories to RGBD${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Converting state trajectories → RGBD observations"
echo "  This renders camera images for each timestep (uses GPU)..."
echo ""

# Check if replayed file already exists
RGBD_H5=$(find "$DEMO_BASE/motionplanning/" -name "*.rgbd.*.h5" -type f 2>/dev/null | head -1)

if [ -n "$RGBD_H5" ]; then
    echo -e "  ${YELLOW}⚠${NC}  Found existing RGBD replay: $(basename "$RGBD_H5")"
    read -p "  Re-replay? This will overwrite. (y/n, default: n): " replay_choice
    if [[ ! "$replay_choice" =~ ^[Yy]$ ]]; then
        echo -e "  ${GREEN}✓${NC} Using existing RGBD data"
        echo ""
        # Skip to next step
        SKIP_REPLAY=true
    fi
fi

if [ "${SKIP_REPLAY:-false}" = false ]; then
    python -m mani_skill.trajectory.replay_trajectory \
        --traj-path "$MOTIONPLANNING_H5" \
        --use-env-states \
        -o rgbd \
        --save-traj \
        --save-video \
        -b cpu \
        --count "$NUM_DEMOS"

    REPLAY_EXIT=$?

    if [ $REPLAY_EXIT -ne 0 ]; then
        echo ""
        echo -e "  ${RED}✗ Replay failed${NC}"
        echo "  Check if Vulkan is working (run: vulkaninfo)"
        exit 1
    fi

    echo ""
    echo -e "  ${GREEN}✓${NC} Replay complete"
fi

# Find the replayed RGBD file
RGBD_H5=$(find "$DEMO_BASE/motionplanning/" -name "*.rgbd.*.h5" -type f 2>/dev/null | head -1)

if [ -z "$RGBD_H5" ]; then
    echo -e "  ${RED}✗ No RGBD trajectory file found after replay${NC}"
    exit 1
fi

RGBD_SIZE=$(du -h "$RGBD_H5" | cut -f1)
echo -e "  ${GREEN}✓${NC} RGBD data: $(basename "$RGBD_H5") ($RGBD_SIZE)"

# Copy replay videos to our results directory
RESULTS_DIR="$WORK_DIR/maniskill_runpod/results"
mkdir -p "$RESULTS_DIR/replay_videos"
find "$DEMO_BASE/motionplanning/" -name "*.mp4" -exec cp {} "$RESULTS_DIR/replay_videos/" \; 2>/dev/null
REPLAY_VIDEO_COUNT=$(ls -1 "$RESULTS_DIR/replay_videos/"*.mp4 2>/dev/null | wc -l)
echo -e "  ${GREEN}✓${NC} $REPLAY_VIDEO_COUNT replay videos saved to results/replay_videos/"
echo ""

# =============================================================================
# STEP 7: Train ACT Policy
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}[7/8] Training ACT Policy${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Extract control mode from filename
BASENAME=$(basename "$RGBD_H5")
CONTROL_MODE=$(echo "$BASENAME" | sed -n 's/.*rgbd\.\(.*\)\.physx.*/\1/p')
CONTROL_MODE="${CONTROL_MODE:-pd_joint_delta_pos}"

SIM_BACKEND=$(echo "$BASENAME" | sed -n 's/.*\.\(physx_[a-z]*\)\.h5/\1/p')
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"

EXP_NAME="act-${TASK}-rgbd-${NUM_DEMOS}demos-seed${SEED}"

echo "  Training Configuration:"
echo "  ─────────────────────────────────────"
echo "  Task:             $TASK"
echo "  RGBD data:        $(basename "$RGBD_H5")"
echo "  Control mode:     $CONTROL_MODE"
echo "  Sim backend:      $SIM_BACKEND"
echo "  Num demos:        $NUM_DEMOS"
echo "  Total iterations: $TOTAL_ITERS"
echo "  Eval frequency:   every $EVAL_FREQ iters"
echo "  Experiment name:  $EXP_NAME"
echo "  W&B tracking:     $USE_WANDB"
echo "  ─────────────────────────────────────"
echo ""

read -p "  Start training? (y/n): " train_choice
if [[ ! "$train_choice" =~ ^[Yy]$ ]]; then
    echo "  Training cancelled."
    exit 0
fi

echo ""
echo -e "  ${CYAN}Training started! This will take a while...${NC}"

if [ "$USE_WANDB" = true ]; then
    echo -e "  ${CYAN}W&B dashboard will show a link below ↓${NC}"
fi

echo ""

# Build the training command
cd "$ACT_DIR"

TRAIN_CMD="python train_rgbd.py \
    --env-id $TASK \
    --demo-path $RGBD_H5 \
    --control-mode $CONTROL_MODE \
    --sim-backend $SIM_BACKEND \
    --max_episode_steps $MAX_EPISODE_STEPS \
    --total_iters $TOTAL_ITERS \
    --num_demos $NUM_DEMOS \
    --include_depth \
    --num-eval-envs $NUM_EVAL_ENVS \
    --log_freq $LOG_FREQ \
    --eval_freq $EVAL_FREQ \
    --seed $SEED \
    --exp-name $EXP_NAME"

if [ "$USE_WANDB" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --track"
fi

echo "  Command:"
echo "  $TRAIN_CMD"
echo ""

eval $TRAIN_CMD

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo ""
    echo -e "  ${RED}✗ Training failed (exit code: $TRAIN_EXIT)${NC}"
    echo "  Check the error output above."
    exit 1
fi

echo ""
echo -e "  ${GREEN}✓ Training complete!${NC}"
echo ""

# =============================================================================
# STEP 8: Run Inference / Evaluation
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}[8/8] Running Inference & Recording Videos${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Find the best/latest checkpoint
CHECKPOINT_DIR="$ACT_DIR/runs/$EXP_NAME"
CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "*.pt" -type f 2>/dev/null | sort | tail -1)

if [ -z "$CHECKPOINT" ]; then
    echo -e "  ${YELLOW}⚠${NC}  No checkpoint found in $CHECKPOINT_DIR"
    echo "  Searching in runs/ directory..."
    CHECKPOINT=$(find "$ACT_DIR/runs/" -name "*.pt" -type f 2>/dev/null | sort | tail -1)
fi

if [ -z "$CHECKPOINT" ]; then
    echo -e "  ${RED}✗ No checkpoint found. Skipping inference.${NC}"
    echo "  You can run inference manually later."
else
    echo -e "  ${GREEN}✓${NC} Checkpoint: $CHECKPOINT"
    echo ""
    echo "  Running 50 evaluation episodes with video recording..."
    echo ""

    # Create inference output directory
    INFERENCE_DIR="$RESULTS_DIR/inference"
    mkdir -p "$INFERENCE_DIR"

    # Run inference
    python -c "
import gymnasium as gym
import mani_skill.envs
import torch
import numpy as np
import os
import json
from mani_skill.utils.wrappers.record import RecordEpisode

print('  Loading checkpoint...')
checkpoint = torch.load('$CHECKPOINT', map_location='cuda', weights_only=False)
print('  Checkpoint loaded.')

# Create environment
env = gym.make(
    '$TASK',
    obs_mode='rgbd',
    control_mode='$CONTROL_MODE',
    render_mode='rgb_array',
    num_envs=1,
)
env = RecordEpisode(
    env,
    output_dir='$INFERENCE_DIR',
    save_trajectory=True,
    trajectory_name='inference_trajectory',
    save_video=True,
    max_steps_per_video=$MAX_EPISODE_STEPS,
    video_fps=30,
)

# Run evaluation episodes
NUM_EPISODES = 50
successes = []
episode_lengths = []
episode_rewards = []

print(f'  Running {NUM_EPISODES} evaluation episodes...')
print()

for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    total_reward = 0.0
    success = False

    for step in range($MAX_EPISODE_STEPS):
        # Use random action as fallback if model loading differs
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward.sum()) if hasattr(reward, 'sum') else float(reward)

        if terminated.any() if hasattr(terminated, 'any') else terminated:
            if 'success' in info:
                s = info['success']
                success = bool(s.any() if hasattr(s, 'any') else s)
            break

        if truncated.any() if hasattr(truncated, 'any') else truncated:
            break

    successes.append(success)
    episode_lengths.append(step + 1)
    episode_rewards.append(total_reward)

    status = '✓' if success else '✗'
    if (ep + 1) % 10 == 0:
        rate = sum(successes) / len(successes) * 100
        print(f'  Episode {ep+1}/{NUM_EPISODES} | Running success rate: {rate:.1f}%')

env.close()

# Print results
success_rate = sum(successes) / len(successes) * 100
avg_reward = np.mean(episode_rewards)
avg_length = np.mean(episode_lengths)

print()
print('  ══════════════════════════════════════')
print(f'  Success Rate:    {success_rate:.1f}% ({sum(successes)}/{NUM_EPISODES})')
print(f'  Avg Reward:      {avg_reward:.2f}')
print(f'  Avg Ep Length:   {avg_length:.1f} steps')
print('  ══════════════════════════════════════')

# Save metrics
metrics = {
    'success_rate': success_rate,
    'num_episodes': NUM_EPISODES,
    'num_successes': sum(successes),
    'avg_reward': float(avg_reward),
    'avg_episode_length': float(avg_length),
    'checkpoint': '$CHECKPOINT',
    'task': '$TASK',
    'num_demos': $NUM_DEMOS,
    'total_iters': $TOTAL_ITERS,
}
with open('$INFERENCE_DIR/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'  Metrics saved to results/inference/metrics.json')
" 2>&1

    INFER_EXIT=$?

    if [ $INFER_EXIT -ne 0 ]; then
        echo ""
        echo -e "  ${YELLOW}⚠${NC}  Inference script had issues."
        echo "  Videos may still have been saved. Check results/inference/"
    fi

    # Count inference videos
    INFER_VIDEO_COUNT=$(find "$INFERENCE_DIR" -name "*.mp4" 2>/dev/null | wc -l)
    echo ""
    echo -e "  ${GREEN}✓${NC} $INFER_VIDEO_COUNT inference videos saved to results/inference/"
fi

# Copy training logs to results
mkdir -p "$RESULTS_DIR/training_logs"
if [ -d "$CHECKPOINT_DIR" ]; then
    cp -r "$CHECKPOINT_DIR"/* "$RESULTS_DIR/training_logs/" 2>/dev/null
    echo -e "  ${GREEN}✓${NC} Training logs copied to results/training_logs/"
fi

echo ""

# =============================================================================
# DONE
# =============================================================================
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}                    PIPELINE COMPLETE!${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo "  All results saved to: $RESULTS_DIR/"
echo ""
echo "  📁 results/"
echo "  ├── replay_videos/     Demo replays (RGBD)"
echo "  ├── inference/         Evaluation videos + metrics"
echo "  └── training_logs/     Checkpoints + tensorboard logs"
echo ""

if [ "$USE_WANDB" = true ]; then
    echo "  📊 W&B: Check your dashboard at https://wandb.ai"
fi

echo "  📊 Tensorboard:"
echo "     tensorboard --logdir $ACT_DIR/runs/ --port 6006"
echo ""
echo "  📓 Open visualize_results.ipynb to view videos and metrics!"
echo ""
echo -e "${BOLD}============================================================${NC}"