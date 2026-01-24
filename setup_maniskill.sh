#!/bin/bash
# ============================================
# ManiSkill Setup Script for RunPod
# Run this after connecting to your RunPod
# ============================================

# DO NOT use set -e - we want to continue even if something fails

echo "======================================"
echo "ManiSkill Setup Script for RunPod"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track what worked and what didn't
VULKAN_OK=false
MANISKILL_OK=false
STATE_ENV_OK=false
RGBD_ENV_OK=false

# Step 1: System Update
echo ""
echo -e "${YELLOW}[1/7] Updating system packages...${NC}"
if apt-get update -qq 2>/dev/null; then
    echo -e "${GREEN}✓ Package lists updated${NC}"
else
    echo -e "${YELLOW}⚠ Package update had issues (continuing anyway)${NC}"
fi

# Step 2: Install Vulkan Dependencies
echo ""
echo -e "${YELLOW}[2/7] Installing Vulkan dependencies...${NC}"
apt-get install -y -qq \
    vulkan-tools \
    libvulkan1 \
    libvulkan-dev \
    mesa-vulkan-drivers \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1-mesa \
    libgbm1 \
    libxkbcommon0 \
    wget \
    unzip \
    git \
    htop \
    ffmpeg \
    2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Vulkan dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Some packages may not have installed (continuing anyway)${NC}"
fi

# Step 3: Create Vulkan Configuration Files
echo ""
echo -e "${YELLOW}[3/7] Creating Vulkan configuration files...${NC}"

# Create nvidia_icd.json
mkdir -p /usr/share/vulkan/icd.d
cat > /usr/share/vulkan/icd.d/nvidia_icd.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.280"
    }
}
EOF

# Create EGL vendor file
mkdir -p /usr/share/glvnd/egl_vendor.d
cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF

# Create implicit layer file
mkdir -p /etc/vulkan/implicit_layer.d
cat > /etc/vulkan/implicit_layer.d/nvidia_layers.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "layer": {
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.280",
        "implementation_version" : "1",
        "description" : "NVIDIA Optimus layer"
    }
}
EOF

echo -e "${GREEN}✓ Vulkan config files created${NC}"

# Step 4: Check Vulkan Installation
echo ""
echo -e "${YELLOW}[4/7] Testing Vulkan installation...${NC}"
if vulkaninfo 2>/dev/null | head -5; then
    echo -e "${GREEN}✓ Vulkan is working!${NC}"
    VULKAN_OK=true
else
    echo -e "${RED}✗ Vulkan not working on this host${NC}"
    echo -e "${YELLOW}  Note: You can still use state-based simulation${NC}"
    VULKAN_OK=false
fi

# Step 5: Install Python Dependencies
echo ""
echo -e "${YELLOW}[5/7] Installing ManiSkill and dependencies...${NC}"
pip install --upgrade pip -q 2>/dev/null

# Install packages one by one to track failures
PACKAGES=(
    "mani_skill"
    "torch"
    "torchvision"
    "h5py"
    "zarr"
    "matplotlib"
    "wandb"
    "tensorboard"
    "einops"
    "diffusers"
    "accelerate"
    "transformers"
    "gymnasium"
    "opencv-python"
    "imageio"
    "imageio-ffmpeg"
    "tqdm"
)

FAILED_PACKAGES=()

for pkg in "${PACKAGES[@]}"; do
    if pip install --break-system-packages -q "$pkg" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $pkg"
    else
        echo -e "  ${RED}✗${NC} $pkg (failed)"
        FAILED_PACKAGES+=("$pkg")
    fi
done

if [ ${#FAILED_PACKAGES[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All packages installed successfully${NC}"
else
    echo -e "${YELLOW}⚠ Some packages failed: ${FAILED_PACKAGES[*]}${NC}"
fi

# Step 6: Create project directories
echo ""
echo -e "${YELLOW}[6/7] Creating project directories...${NC}"
mkdir -p /workspace/maniskill_project/{data,checkpoints,logs,videos,rgbd_data}
mkdir -p ~/.maniskill/data
echo -e "${GREEN}✓ Directories created${NC}"

# Step 7: Test ManiSkill
echo ""
echo -e "${YELLOW}[7/7] Testing ManiSkill installation...${NC}"

# Test basic import
echo "Testing ManiSkill import..."
if python -c "import mani_skill; print('  ManiSkill version:', mani_skill.__version__ if hasattr(mani_skill, '__version__') else 'installed')" 2>/dev/null; then
    echo -e "${GREEN}✓ ManiSkill package installed correctly${NC}"
    MANISKILL_OK=true
else
    echo -e "${RED}✗ ManiSkill import failed${NC}"
    MANISKILL_OK=false
fi

# Test state-based environment (always works)
echo ""
echo "Testing state-based environment (no rendering)..."
if python -c "
import gymnasium as gym
import mani_skill.envs
env = gym.make('PickCube-v1', obs_mode='state', render_mode=None, num_envs=1)
obs, _ = env.reset()
print(f'  State observation shape: {obs.shape}')
env.close()
" 2>/dev/null; then
    echo -e "${GREEN}✓ State-based simulation working${NC}"
    STATE_ENV_OK=true
else
    echo -e "${RED}✗ State-based simulation failed${NC}"
    STATE_ENV_OK=false
fi

# Test rendering if Vulkan works
if [ "$VULKAN_OK" = true ]; then
    echo ""
    echo "Testing RGBD rendering..."
    if timeout 60 python -c "
import gymnasium as gym
import mani_skill.envs
env = gym.make('PickCube-v1', obs_mode='rgbd', render_mode='rgb_array', num_envs=1)
obs, _ = env.reset()
print(f'  RGBD observation keys: {list(obs.keys())}')
if 'sensor_data' in obs:
    print(f'  Cameras: {list(obs[\"sensor_data\"].keys())}')
env.close()
" 2>/dev/null; then
        echo -e "${GREEN}✓ RGBD rendering working${NC}"
        RGBD_ENV_OK=true
    else
        echo -e "${YELLOW}⚠ RGBD rendering has issues${NC}"
        RGBD_ENV_OK=false
    fi
fi

# ============================================
# SUMMARY
# ============================================
echo ""
echo "======================================"
echo "          SETUP SUMMARY"
echo "======================================"
echo ""

# Show status table
echo "Component Status:"
echo "─────────────────────────────────────"
if [ "$VULKAN_OK" = true ]; then
    echo -e "  Vulkan Driver:      ${GREEN}✓ Working${NC}"
else
    echo -e "  Vulkan Driver:      ${RED}✗ Not Working${NC}"
fi

if [ "$MANISKILL_OK" = true ]; then
    echo -e "  ManiSkill Package:  ${GREEN}✓ Installed${NC}"
else
    echo -e "  ManiSkill Package:  ${RED}✗ Failed${NC}"
fi

if [ "$STATE_ENV_OK" = true ]; then
    echo -e "  State-based Sim:    ${GREEN}✓ Working${NC}"
else
    echo -e "  State-based Sim:    ${RED}✗ Failed${NC}"
fi

if [ "$RGBD_ENV_OK" = true ]; then
    echo -e "  RGBD Rendering:     ${GREEN}✓ Working${NC}"
elif [ "$VULKAN_OK" = false ]; then
    echo -e "  RGBD Rendering:     ${YELLOW}⚠ Skipped (no Vulkan)${NC}"
else
    echo -e "  RGBD Rendering:     ${RED}✗ Failed${NC}"
fi

echo "─────────────────────────────────────"
echo ""

# Project info
echo "Project Directory: /workspace/maniskill_project/"
echo "  ├── data/        : Store datasets"
echo "  ├── rgbd_data/   : Replayed RGBD trajectories"
echo "  ├── checkpoints/ : Model weights"
echo "  ├── logs/        : Training logs"
echo "  └── videos/      : Recorded episodes"
echo ""

# Recommendations based on status
if [ "$RGBD_ENV_OK" = true ]; then
    echo -e "${GREEN}🎉 Everything is working! You can proceed with the notebook.${NC}"
elif [ "$VULKAN_OK" = false ]; then
    echo -e "${YELLOW}⚠ Vulkan is not working on this host.${NC}"
    echo ""
    echo "Options:"
    echo "  1. Stop pod and redeploy (different host might work)"
    echo "  2. Use state-based training only (no visual observations)"
    echo "  3. Use pre-collected RGBD data"
elif [ "$STATE_ENV_OK" = true ]; then
    echo -e "${YELLOW}⚠ State-based works but RGBD has issues.${NC}"
    echo "Try running the notebook - sometimes it works despite test failures."
fi

echo ""
echo "======================================"
echo "         SETUP COMPLETE"
echo "======================================"
echo ""
echo "Next step: Open and run the ManiSkill notebook!"
echo ""
