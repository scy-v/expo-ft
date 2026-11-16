# Source from any directory; only activates the local UR actor environment.
EXPO_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$EXPO_REPO_ROOT/expo-setup/uv-cache"
export UV_PYTHON_INSTALL_DIR="$EXPO_REPO_ROOT/expo-setup/python"
export MPLCONFIGDIR="$EXPO_REPO_ROOT/logs/matplotlib"
source "$EXPO_REPO_ROOT/client/.venv/bin/activate"
