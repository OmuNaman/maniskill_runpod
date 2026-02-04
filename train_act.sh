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

# Number of eval episodes
echo "  How many episodes to run per evaluation?"
echo "  More episodes = more reliable metric but slower eval."
echo ""
echo "  ┌──────────────────────────────────────────────────┐"
echo "  │  10 episodes  → ~1.5 min  (quick test)          │"
echo "  │  50 episodes  → ~7 min    (decent signal)       │"
echo "  │  100 episodes → ~15 min   (official, reliable)  │"
echo "  └──────────────────────────────────────────────────┘"
echo ""
read -p "  Episodes per evaluation? (default: 100): " eval_eps_input
NUM_EVAL_EPISODES="${eval_eps_input:-100}"
echo -e "  ${GREEN}✓${NC} Running ${BOLD}$NUM_EVAL_EPISODES${NC} episodes per evaluation"
echo ""

# Number of parallel eval environments
echo "  How many parallel environments during evaluation?"
echo "  More = faster eval, but uses more memory."
echo "  ⚠  Values >1 may cause issues on some setups."
echo ""
echo "  ┌──────────────────────────────────────────────────┐"
echo "  │  1  → safest, ~15 min/eval with 100 eps         │"
echo "  │  10 → ~10x faster eval, moderate memory          │"
echo "  │  25 → very fast eval, higher memory              │"
echo "  └──────────────────────────────────────────────────┘"
echo ""
read -p "  Parallel eval environments? (default: $NUM_EVAL_ENVS): " eval_envs_input
NUM_EVAL_ENVS="${eval_envs_input:-$NUM_EVAL_ENVS}"
echo -e "  ${GREEN}✓${NC} Using ${BOLD}$NUM_EVAL_ENVS${NC} parallel eval environment(s)"
echo ""

# Calculate and show estimated eval time
ESTIMATED_EVAL_MINS=$(python3 -c "
eps = $NUM_EVAL_EPISODES
envs = $NUM_EVAL_ENVS
steps = $MAX_EPISODE_STEPS
# Rough estimate: ~0.1 sec per step per env
time_sec = (eps / envs) * steps * 0.1
num_evals = ($TOTAL_ITERS // $EVAL_FREQ) + 1
total_min = (time_sec * num_evals) / 60
print(f'{total_min:.0f}')
" 2>/dev/null)

NUM_EVALS=$(( (TOTAL_ITERS / EVAL_FREQ) + 1 ))
echo "  ┌──────────────────────────────────────────────────┐"
echo "  │  Summary:                                        │"
echo "  │  Evals will run $NUM_EVALS times during training"
echo "  │  Each eval: $NUM_EVAL_EPISODES eps × $MAX_EPISODE_STEPS steps ÷ $NUM_EVAL_ENVS envs"
echo "  │  Estimated total eval time: ~${ESTIMATED_EVAL_MINS} minutes"
echo "  └──────────────────────────────────────────────────┘"
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

# Detect GPU for display
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "CUDA GPU")

echo "  Training Configuration:"
echo "  ─────────────────────────────────────"
echo -e "  Training device:  ${GREEN}GPU ($GPU_NAME)${NC}"
echo "  Eval sim backend: $SIM_BACKEND (physics only)"
echo "  Task:             $TASK"
echo "  RGBD data:        $(basename "$RGBD_H5")"
echo "  Control mode:     $CONTROL_MODE"
echo "  Num demos:        $NUM_DEMOS"
echo "  Total iterations: $TOTAL_ITERS"
echo "  Eval frequency:   every $EVAL_FREQ iters"
echo "  Eval episodes:    $NUM_EVAL_EPISODES per eval"
echo "  Eval envs:        $NUM_EVAL_ENVS parallel"
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
    --num_eval_episodes $NUM_EVAL_EPISODES \
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
# STEP 8: Run Inference & Record Videos
# =============================================================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}[8/8] Running Inference & Recording Videos${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Find the best checkpoint
CHECKPOINT_DIR="$ACT_DIR/runs/$EXP_NAME"
CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "best_eval_success_once.pt" -type f 2>/dev/null | head -1)

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "*.pt" -type f 2>/dev/null | sort | tail -1)
fi

if [ -z "$CHECKPOINT" ]; then
    echo -e "  ${YELLOW}⚠${NC}  No checkpoint found in $CHECKPOINT_DIR"
    echo "  Searching in runs/ directory..."
    CHECKPOINT=$(find "$ACT_DIR/runs/" -name "best_eval_success_once.pt" -type f 2>/dev/null | head -1)
    if [ -z "$CHECKPOINT" ]; then
        CHECKPOINT=$(find "$ACT_DIR/runs/" -name "*.pt" -type f 2>/dev/null | sort | tail -1)
    fi
fi

if [ -z "$CHECKPOINT" ]; then
    echo -e "  ${RED}✗ No checkpoint found. Skipping inference.${NC}"
    echo "  You can run inference manually later."
else
    echo -e "  ${GREEN}✓${NC} Checkpoint: $CHECKPOINT"
    echo ""

    # Create inference output directories
    INFERENCE_DIR="$RESULTS_DIR/inference_videos"
    mkdir -p "$INFERENCE_DIR"

    NUM_INFER_EPISODES=10
    echo "  Running $NUM_INFER_EPISODES evaluation episodes with video recording..."
    echo ""

    # Run inference using the SAME code as training (importlib approach)
    # This imports train_rgbd.py as a module to get the exact Agent class,
    # FlattenRGBDObservationWrapper, and evaluate() function.
    python3 -c "
import sys, os, json
import importlib.util
import torch
import numpy as np
from functools import partial

ACT_DIR = '$ACT_DIR'
sys.path.insert(0, ACT_DIR)

# Import train_rgbd.py as module (without running __main__)
spec = importlib.util.spec_from_file_location('train_rgbd', os.path.join(ACT_DIR, 'train_rgbd.py'))
mod = importlib.util.module_from_spec(spec)
sys.modules['train_rgbd'] = mod
spec.loader.exec_module(mod)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load checkpoint
print('  Loading checkpoint...')
ckpt = torch.load('$CHECKPOINT', map_location=device, weights_only=False)
print(f'  Checkpoint keys: {list(ckpt.keys())}')

# Setup args (Agent.get_action() references module-level args)
agent_args = mod.Args()
agent_args.include_depth = True
agent_args.env_id = '$TASK'
agent_args.control_mode = '$CONTROL_MODE'
agent_args.sim_backend = '$SIM_BACKEND'
agent_args.max_episode_steps = $MAX_EPISODE_STEPS
mod.args = agent_args

# Create eval environment WITH video recording
from act.make_env import make_eval_envs

video_dir = '$INFERENCE_DIR'
env_kwargs = dict(
    control_mode='$CONTROL_MODE',
    reward_mode='sparse',
    obs_mode='rgbd',
    render_mode='rgb_array',
    max_episode_steps=$MAX_EPISODE_STEPS,
)
wrappers = [partial(mod.FlattenRGBDObservationWrapper, depth=True)]
eval_envs = make_eval_envs(
    '$TASK', 1, '$SIM_BACKEND',
    env_kwargs, None,
    video_dir=video_dir,
    wrappers=wrappers
)
print('  ✓ Environment ready (with video recording)')

# Build Agent and load EMA weights (EMA usually performs better)
agent = mod.Agent(eval_envs, agent_args).to(device)
weight_key = 'ema_agent' if 'ema_agent' in ckpt else 'agent'
agent.load_state_dict(ckpt[weight_key])
param_count = sum(p.numel() for p in agent.parameters())
print(f'  ✓ Loaded {weight_key} weights ({param_count:,} params)')

# Run evaluation using the SAME evaluate() as training
from act.evaluate import evaluate

eval_kwargs = {
    'stats': ckpt['norm_stats'],
    'num_queries': agent_args.num_queries,
    'temporal_agg': agent_args.temporal_agg,
    'max_timesteps': $MAX_EPISODE_STEPS,
    'device': device,
    'sim_backend': '$SIM_BACKEND',
}

print(f'  Running $NUM_INFER_EPISODES episodes...')
print()
eval_metrics = evaluate($NUM_INFER_EPISODES, agent, eval_envs, eval_kwargs)
eval_envs.close()

# Results
success_once = float(np.mean(eval_metrics.get('success_once', np.array([0]))))
success_end = float(np.mean(eval_metrics.get('success_at_end', np.array([0]))))
avg_return = float(np.mean(eval_metrics.get('return', np.array([0]))))
avg_ep_len = float(np.mean(eval_metrics.get('episode_len', np.array([$MAX_EPISODE_STEPS]))))
avg_reward = float(np.mean(eval_metrics.get('reward', np.array([0]))))

print()
print('  ══════════════════════════════════════════════════')
print(f'  Success Once:    {success_once*100:.1f}%')
print(f'  Success at End:  {success_end*100:.1f}%')
print(f'  Avg Return:      {avg_return:.2f}')
print(f'  Avg Ep Length:   {avg_ep_len:.1f} steps')
print(f'  Weights used:    {weight_key}')
print('  ══════════════════════════════════════════════════')

# Save metrics
results = {
    'success_once': success_once,
    'success_at_end': success_end,
    'avg_return': avg_return,
    'avg_reward': avg_reward,
    'avg_episode_length': avg_ep_len,
    'num_episodes': $NUM_INFER_EPISODES,
    'checkpoint': '$CHECKPOINT',
    'weight_type': weight_key,
    'env_id': '$TASK',
    'num_demos': $NUM_DEMOS,
    'total_iters': $TOTAL_ITERS,
}
with open(os.path.join(video_dir, 'metrics.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f'  Metrics saved to {video_dir}/metrics.json')

# List recorded videos
videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
print(f'  Videos recorded: {len(videos)}')
" 2>&1

    INFER_EXIT=$?

    if [ $INFER_EXIT -ne 0 ]; then
        echo ""
        echo -e "  ${YELLOW}⚠${NC}  Inference had issues. Check error above."
        echo "  You can run inference manually with run_inference.py"
    fi

    # Count inference videos
    INFER_VIDEO_COUNT=$(find "$INFERENCE_DIR" -name "*.mp4" 2>/dev/null | wc -l)
    if [ "$INFER_VIDEO_COUNT" -gt 0 ]; then
        echo ""
        echo -e "  ${GREEN}✓${NC} $INFER_VIDEO_COUNT inference videos saved to results/inference_videos/"
    fi
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
echo "  ├── replay_videos/       Expert demo replays (RGBD)"
echo "  ├── inference_videos/    Trained policy videos + metrics"
echo "  └── training_logs/       Checkpoints + tensorboard logs"
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