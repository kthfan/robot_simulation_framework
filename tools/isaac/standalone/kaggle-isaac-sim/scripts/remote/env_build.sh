#!/usr/bin/env bash
# Kaggle, GPU kernel, once per driver (isaac_kaggle.sh rebuild): pack a prebuilt Isaac Sim
# environment with a warm RTX shader cache. Runs after setup.sh, inside the runner's venv.
# Output (the env kernel's, mounted by every `isaac_kaggle.sh run`):
#   env.tar              uncompressed, small: the uv Python (the runner extracts it itself)
#   isaac-venv.tar.zst   the venv (~25 GB, ~11 GB compressed: Kaggle keeps at most 20 GB of
#                        output, and the GPU image has no zstd for the runner's own tar)
#   nvidia-vk.tar.zst    Vulkan for this driver (vulkan_setup.sh) plus the loader, vulkaninfo, zstd
#   env.json             what was built, including the driver ("driver")
set -uo pipefail
OUT=${KGT_OUT:-/kaggle/working}
KIT=${ISAAC_KIT:-/tmp/isaac-kit}
VENV=${KGT_VENV:-/tmp/kgt-venv}
UVPY=${KGT_UV_PY:-/tmp/kgt-uv-python}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$KIT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p /tmp/isaac-warm
source "$KIT/vulkan_setup.sh" > /tmp/isaac-warm/vulkan_setup.log 2>&1; grep -E "VULKAN" /tmp/isaac-warm/vulkan_setup.log
du -sh "$VENV" | sed 's/^/[env] venv before warm-up: /'
# warm-up: compile the RTX shaders once (~10 min on a T4) so they ship with the environment
t=$(date +%s)
timeout 2400 python "$KIT/render.py" --demo --png /tmp/isaac-warm/warm.png --out "$OUT/warm_render.json"
echo "[env] warm-up render: $(( $(date +%s) - t )) s"
du -sh "$VENV" | sed 's/^/[env] venv after warm-up: /'
command -v zstd >/dev/null || apt-get install -y -qq zstd >/dev/null 2>&1
VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
dpkg -L libvulkan1 vulkan-tools zstd 2>/dev/null | while read -r f; do [ -f "$f" ] || [ -L "$f" ] && echo "${f#/}"; done \
    > /tmp/vk-files.txt
echo "${VK_DRIVER_DIR:-/opt/nvidia-vk}" | sed 's#^/##' >> /tmp/vk-files.txt
tar -C / -cf - -T /tmp/vk-files.txt | zstd -T0 -3 -q -o "$OUT/nvidia-vk.tar.zst"
echo "[env] nvidia-vk.tar.zst $(numfmt --to=iec $(stat -c %s "$OUT/nvidia-vk.tar.zst")) for driver $VER"
find "$VENV" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
t=$(date +%s)
tar -C / -cf "$OUT/env.tar" "${UVPY#/}"
tar -C / -cf - "${VENV#/}" root/.cache/ov root/.nvidia-omniverse 2>/dev/null | zstd -T0 -3 -q -o "$OUT/isaac-venv.tar.zst"
bytes=$(stat -c %s "$OUT/isaac-venv.tar.zst")
echo "[env] isaac-venv.tar.zst $(numfmt --to=iec $bytes) packed in $(( $(date +%s) - t )) s"
isaac=$("$VENV/bin/python" -c "import importlib.metadata as m; print(m.version('isaacsim'))" 2>/dev/null)
cat > "$OUT/env.json" <<JSON
{"name": "${ISAAC_ENV_NAME:-isaac62}", "python": "$("$VENV/bin/python" -V 2>&1)", "torch": "", "isaacsim": "$isaac",
 "driver": "$VER", "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)",
 "extra_pip": "${ISAAC_EXTRA_PIP:-}",
 "venv_archive": "isaac-venv.tar.zst", "vulkan_archive": "nvidia-vk.tar.zst", "venv_path": "$VENV", "uv_python_dir": "$UVPY",
 "bytes": $bytes, "built_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
cat "$OUT/env.json"
[ "$bytes" -lt 19000000000 ] || { echo "[env] too big for Kaggle's 20 GB output"; exit 1; }
