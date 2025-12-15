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
    if uv pip install xensesdk==1.6.5 --no-deps; then
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
    exit 0
else
    echo "Invalid argument. Usage:"
    echo "  --conda [env_name]   Create a conda environment (requires Miniconda/Anaconda)"
    echo "  --mamba [env_name]   Create a mamba environment (requires Miniforge)"
    echo "  --install            Install the package in the currently activated environment"
    exit 1
fi