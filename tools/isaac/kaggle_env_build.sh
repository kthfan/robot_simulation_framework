#!/usr/bin/env bash
# Kaggle, GPU kernel, once: a prebuilt Isaac Sim environment with a warm RTX shader cache.
#   kaggle_run.sh run --python 3.12 --slug kgt-env-isaac62 --timeout 90m \
#       --setup "bash tools/isaac/kaggle_setup.sh" --cmd "bash tools/isaac/kaggle_env_build.sh"
# Output (the env kernel's, mounted by `kaggle_run.sh run --env isaac62`):
#   env.tar              uncompressed, small: the uv Python (the runner extracts it itself)
#   isaac-venv.tar.zst   the venv (~25 GB, ~11 GB compressed: Kaggle keeps at most 20 GB of
#                        output, and its GPU image has no zstd for the runner's own tar),
#                        restored by tools/isaac/kaggle_env_restore.sh
#   nvidia-vk.tar.zst    Vulkan for this driver (tools/isaac/vulkan_setup.sh: NVIDIA's graphics
#                        libraries, the loader, vulkaninfo, zstd); env.json "driver" names it.
#                        A run on another driver reinstalls Vulkan and marks the cache stale.
set -uo pipefail
OUT=${KGT_OUT:-/kaggle/working}
VENV=${KGT_VENV:-/tmp/kgt-venv}
UVPY=${KGT_UV_PY:-/tmp/kgt-uv-python}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src
mkdir -p reports/isaac
source tools/isaac/vulkan_setup.sh > reports/isaac/vulkan_setup.log 2>&1; grep -E "VULKAN" reports/isaac/vulkan_setup.log
du -sh "$VENV" | sed 's/^/[env] venv before warm-up: /'
# warm-up: compile the RTX shaders once (~10 min on a T4), so they ship with the environment
timeout 600 python - <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from rsf.model.layout import load_world
from rsf.compile.usd import world_stage
world_stage(load_world("cells/tm_tray/layouts/strict.yaml"), "reports/isaac/strict.usda")
app.close()
PY
t=$(date +%s)
timeout 2400 python tools/isaac/render_probe.py --usd reports/isaac/strict.usda --png reports/isaac/warm.png \
    --out reports/isaac/warm.json
echo "[env] warm-up render: $(( $(date +%s) - t )) s"
du -sh "$VENV" | sed 's/^/[env] venv after warm-up: /'
du -sh "$VENV"/lib/python3.12/site-packages/isaacsim/kit/cache "$VENV"/lib/python3.12/site-packages/isaacsim/kit/data \
    ~/.cache/ov ~/.nvidia-omniverse 2>/dev/null | sed 's/^/[env] cache: /'
command -v zstd >/dev/null || apt-get install -y -qq zstd >/dev/null 2>&1
VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
# Vulkan: the driver's libraries and the apt packages it needed (the registration in /etc is
# rewritten at restore by vulkan_setup.sh)
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
cat > "$OUT/env.json" <<JSON
{"name": "isaac62", "python": "$("$VENV/bin/python" -V 2>&1)", "torch": "", "isaacsim": "6.2.0.0",
 "driver": "$VER", "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)",
 "venv_archive": "isaac-venv.tar.zst", "vulkan_archive": "nvidia-vk.tar.zst", "venv_path": "$VENV", "uv_python_dir": "$UVPY",
 "bytes": $bytes, "built_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
rm -f reports/isaac/strict.usda
[ "$bytes" -lt 19000000000 ] || { echo "[env] too big for Kaggle's 20 GB output"; exit 1; }
