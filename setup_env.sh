#!/bin/bash

# Check the operating system
OS_NAME=$(uname -s)
OS_VERSION=""

if [[ "$OS_NAME" == "Linux" ]]; then
    if command -v lsb_release &>/dev/null; then
        OS_VERSION=$(lsb_release -rs)
    elif [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS_VERSION=$VERSION_ID
    fi
    if [[ "$OS_VERSION" != "22.04" ]]; then
        echo "Warning: This script has only been tested on Ubuntu 22.04"
        echo "Your system is running Ubuntu $OS_VERSION."
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Installation cancelled."
            exit 1
        fi
    fi
else
    echo "Unsupported operating system: $OS_NAME"
    exit 1
fi

echo "Operating system check passed: $OS_NAME $OS_VERSION"

# Resolve script directory so files can be referenced reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# Conda environment yaml file (canonical name)
CONDA_ENV_FILE="$SCRIPT_DIR/conda_environment.yaml"
if [[ ! -f "$CONDA_ENV_FILE" ]]; then
    echo "Error: conda environment yaml not found: $CONDA_ENV_FILE"
    exit 1
fi

# Function to create environment
create_environment() {
    local CONDA_CMD=$1
    local ENV_NAME=$2

    # Deactivate current environment if any (use conda deactivate for both conda and mamba)
    conda deactivate 2>/dev/null || true

    # Remove existing environment if it exists
    if $CONDA_CMD env list | grep -q "^$ENV_NAME "; then
        echo "Removing existing environment '$ENV_NAME'..."
        $CONDA_CMD env remove -n "$ENV_NAME" -y
    fi

    # Create new environment from conda_environment.yaml
    $CONDA_CMD env create -f "$CONDA_ENV_FILE" -n "$ENV_NAME"

    echo "$CONDA_CMD environment '$ENV_NAME' created from: $CONDA_ENV_FILE"

    echo -e "[INFO] Created $CONDA_CMD environment named '$ENV_NAME'.\n"
    echo -e "\t\t1. To activate the environment, run:                $CONDA_CMD activate $ENV_NAME"
    echo -e "\t\t2. To install conda dependencies, run:              bash $SCRIPT_NAME --install"
    echo -e "\t\t3. To deactivate the environment, run:              conda deactivate"
    echo -e "\n"
}

# Check if an environment name is provided
if [[ -n "$2" ]]; then
    ENV_NAME="$2"
else
    ENV_NAME="lerobot-xense"
fi

# Check if the --conda parameter is passed
if [[ "$1" == "--conda" ]]; then
    # Initialize conda
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "Conda initialization script not found. Please install Miniconda3 or Anaconda3 or Miniforge3."
        exit 1
    fi
    create_environment "conda" "$ENV_NAME"

# Check if the --mamba parameter is passed
elif [[ "$1" == "--mamba" ]]; then
    # Initialize mamba (miniforge)
    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniforge3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
        . "$HOME/mambaforge/etc/profile.d/conda.sh"
    else
        echo "Mamba initialization script not found. Please install Miniforge3 or Mambaforge."
        exit 1
    fi
    # Also source mamba.sh if available for full mamba support
    if [ -f "$HOME/miniforge3/etc/profile.d/mamba.sh" ]; then
        . "$HOME/miniforge3/etc/profile.d/mamba.sh"
    elif [ -f "$HOME/mambaforge/etc/profile.d/mamba.sh" ]; then
        . "$HOME/mambaforge/etc/profile.d/mamba.sh"
    fi
    create_environment "mamba" "$ENV_NAME"

# Check if the --install parameter is passed
elif [[ "$1" == "--install" ]]; then
    # Get the currently activated conda environment name
    if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
        echo "Error: No conda/mamba environment is currently activated."
        echo "Please activate an environment first with: conda/mamba activate <env_name>"
        exit 1
    fi
    ENV_NAME=${CONDA_DEFAULT_ENV}

    # Detect conda/mamba command
    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        CONDA_CMD="mamba"
    else
        CONDA_CMD="conda"
    fi

    echo "[INFO] Updating conda environment '$ENV_NAME' from: $CONDA_ENV_FILE"
    $CONDA_CMD env update -f "$CONDA_ENV_FILE" -n "$ENV_NAME"


    echo -e "\n[INFO] Conda dependencies installed/updated for '$ENV_NAME'.\n"
    pip install uv
    uv pip install --upgrade pip
    # Ensure editable installs (PEP 660) work: setuptools must provide build_editable
    uv pip install --upgrade "setuptools>=71.0.0,<81.0.0" wheel

    # Workaround for Python ctypes.util.find_library("udev") on conda envs:
    # Hacking the udev library discovery to avoid issues with pyudev/xensesdk.
    # If $CONDA_PREFIX/lib/udev exists as a directory, Python may return that directory as the "udev" library,
    # causing pyudev/xensesdk to crash with: "OSError: .../lib/udev: Is a directory".
    if [[ -n "${CONDA_PREFIX}" && -d "${CONDA_PREFIX}/lib/udev" ]]; then
        echo "[INFO] Fixing libudev discovery for pyudev (renaming ${CONDA_PREFIX}/lib/udev)..."
        if [[ -e "${CONDA_PREFIX}/lib/udev.rules.d" ]]; then
            mv "${CONDA_PREFIX}/lib/udev" "${CONDA_PREFIX}/lib/udev.rules.d.bak.$(date +%s)" || true
        else
            mv "${CONDA_PREFIX}/lib/udev" "${CONDA_PREFIX}/lib/udev.rules.d" || true
        fi
    fi
    if [[ -n "${CONDA_PREFIX}" && -e "${CONDA_PREFIX}/lib/libudev.so.1" && ! -e "${CONDA_PREFIX}/lib/libudev.so" ]]; then
        ln -s libudev.so.1 "${CONDA_PREFIX}/lib/libudev.so" || true
    fi

    # project root directory
    PROJECT_ROOT=$(pwd)
    ARX5_SDK_DIR="$PROJECT_ROOT/src/lerobot/robots/bi_arx5/ARX5_SDK"
    if [[ -d "$ARX5_SDK_DIR" ]]; then
        echo "[INFO] Building ARX5 SDK..."
        cd "$ARX5_SDK_DIR"
        rm -rf build
        mkdir build
        cd build
        cmake ..
        sudo make install -j4
        echo "[INFO] ARX5 SDK built successfully!"
        
        # Set real-time scheduling capability for Python (required by ARX5 SDK)
        echo "[INFO] Setting real-time scheduling capability for Python..."
        PYTHON_REAL_PATH=$(readlink -f "$CONDA_PREFIX/bin/python")
        sudo setcap cap_sys_nice=ep "$PYTHON_REAL_PATH"
        echo "[INFO] Real-time scheduling capability set for: $PYTHON_REAL_PATH"
        
        # Create sitecustomize.py to preload conda's libstdc++ (fixes CXXABI version issues)
        echo "[INFO] Creating sitecustomize.py for C++ ABI compatibility..."
        PY_VER="$(python -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
        SITE_PACKAGES_DIR="${CONDA_PREFIX}/lib/${PY_VER}/site-packages"
        SITECUSTOMIZE_FILE="${SITE_PACKAGES_DIR}/sitecustomize.py"
        
        cat > "$SITECUSTOMIZE_FILE" << 'EOF'
"""
Sitecustomize for conda environment.

This file is automatically executed when Python starts.
It preloads the conda environment's libstdc++.so.6 to ensure C++ extensions
compiled with GCC 14.3.0 can find the required CXXABI_1.3.15 symbols.
"""
import os
import ctypes

conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    libstdcxx_path = os.path.join(conda_prefix, 'lib', 'libstdc++.so.6')
    if os.path.exists(libstdcxx_path):
        try:
            # Preload with RTLD_GLOBAL so all subsequently loaded modules can use it
            ctypes.CDLL(libstdcxx_path, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            # Silently fail if preloading doesn't work
            pass
EOF
        echo "[INFO] sitecustomize.py created at: $SITECUSTOMIZE_FILE"
    fi
    cd "$PROJECT_ROOT"
    echo "[INFO] Installing Lerobot from pyproject.toml"
    if uv pip install -e .; then
        echo "[INFO] Lerobot installed successfully!"
    else
        echo "[ERROR] Lerobot installation failed. See the error output above."
        exit 1
    fi
    echo "[INFO] Installing xensesdk..."

    uv pip install onnxruntime-gpu==1.19.2
    if uv pip install xensesdk==1.6.5; then
        uv pip install av==15.1.0
        echo "[INFO] xensesdk installed successfully!"
        # Workaround:
        # After installing xensesdk, remove OpenCV's bundled Qt platform plugin if present.
        # This avoids Qt/XCB plugin loading issues inside conda environments.
        if [[ -n "${CONDA_PREFIX}" ]]; then
            PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
            QXCB_PATH="${CONDA_PREFIX}/lib/python${PY_VER}/site-packages/cv2/qt/plugins/platforms/libqxcb.so"
            QXCB_PATH_310="${CONDA_PREFIX}/lib/python3.10/site-packages/cv2/qt/plugins/platforms/libqxcb.so"

            if [[ -f "$QXCB_PATH" ]]; then
                echo "[INFO] Removing OpenCV Qt plugin: $QXCB_PATH"
                rm -f "$QXCB_PATH"
            elif [[ -f "$QXCB_PATH_310" ]]; then
                echo "[INFO] Removing OpenCV Qt plugin: $QXCB_PATH_310"
                rm -f "$QXCB_PATH_310"
            else
                echo "[INFO] OpenCV Qt plugin (libqxcb.so) not found; skipping removal."
            fi
        else
            echo "[WARN] CONDA_PREFIX is not set; cannot remove OpenCV Qt plugin."
        fi

    else
        echo "[ERROR] xensesdk installation failed. See the error output above."
        exit 1
    fi

    # Install xensegripper without overriding xensesdk version
    echo "[INFO] Installing xensegripper (keeping xensesdk==1.6.5)..."
    # Use --no-deps to prevent xensegripper from pulling in xensesdk 1.7.0
    if uv pip install xensegripper --no-deps; then
        # Reinstall xensesdk to ensure correct version after any potential override
        echo "[INFO] Reinstalling xensesdk==1.6.5..."
        uv pip install xensesdk==1.6.5 --force-reinstall --no-deps
        echo "[INFO] xensegripper installed successfully!"
    else
        echo "[ERROR] xensegripper installation failed. See the error output above."
        exit 1
    fi

    # Install xensevr_pc_service_sdk for pico4 teleoperator
    echo "[INFO] Installing xensevr_pc_service_sdk..."
    
    # Save the project root directory
    PROJECT_ROOT=$(pwd)
    XENSEVR_PC_SERVICE_PYBIND_DIR="$PROJECT_ROOT/src/lerobot/teleoperators/pico4/xensevr-pc-service-pybind"

    # Install the required packages
    cd "$XENSEVR_PC_SERVICE_PYBIND_DIR"
    mkdir -p dependencies
    cd dependencies

    # Clone if not already cloned
    if [ ! -d "XenseVR-PC-Service" ]; then
        git clone https://github.com/Vertax42/XenseVR-PC-Service.git
    fi
    cd XenseVR-PC-Service/RoboticsService/PXREARobotSDK
    bash build.sh
    
    # Go back to xensevr-pc-service-pybind directory
    cd "$XENSEVR_PC_SERVICE_PYBIND_DIR"
    mkdir -p lib
    mkdir -p include

    # Copy files from the cloned repo
    SDK_DIR="$XENSEVR_PC_SERVICE_PYBIND_DIR/dependencies/XenseVR-PC-Service/RoboticsService/PXREARobotSDK"
    cp "$SDK_DIR/PXREARobotSDK.h" include/
    cp -r "$SDK_DIR/nlohmann" include/
    cp "$SDK_DIR/build/libPXREARobotSDK.so" lib/

    pip uninstall -y xensevr_pc_service_sdk 2>/dev/null || true
    python setup.py install
    echo -e "[INFO] xensevr_pc_service_sdk is installed in $CONDA_CMD environment '$ENV_NAME'.\n"
    
    # Verify critical package versions
    echo "[INFO] Verifying package versions..."
    cd "$PROJECT_ROOT"
    TORCHCODEC_VER=$(python -c "import torchcodec; print(torchcodec.__version__)" 2>/dev/null || echo "NOT INSTALLED")
    AV_VER=$(python -c "import av; print(av.__version__)" 2>/dev/null || echo "NOT INSTALLED")
    TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT INSTALLED")
    echo "  - torch: $TORCH_VER"
    echo "  - torchcodec: $TORCHCODEC_VER (should be 0.7.0)"
    echo "  - av (pyav): $AV_VER (should be 15.1.0)"
    
    if [[ "$TORCHCODEC_VER" != "0.7.0" ]]; then
        echo "[WARN] torchcodec version mismatch! Expected 0.7.0, got $TORCHCODEC_VER"
        echo "[INFO] Attempting to fix torchcodec version..."
        uv pip install torchcodec==0.7.0 --force-reinstall
    fi
    
    if [[ "$AV_VER" != "15.1.0" ]]; then
        echo "[WARN] av (pyav) version mismatch! Expected 15.1.0, got $AV_VER"
        echo "[INFO] Attempting to fix av version..."
        uv pip install av==15.1.0 --force-reinstall
    fi
    
    echo "[INFO] Lerobot-Xense installation completed successfully!"
    exit 0
else
    echo "Invalid argument. Usage:"
    echo "  --conda [env_name]   Create a conda environment (requires Miniconda/Anaconda)"
    echo "  --mamba [env_name]   Create a mamba environment (requires Miniforge)"
    echo "  --install            Install the package in the currently activated environment"
    exit 1
fi