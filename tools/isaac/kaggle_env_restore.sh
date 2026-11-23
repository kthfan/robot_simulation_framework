#!/usr/bin/env bash
# Restore the prebuilt Isaac Sim environment (tools/isaac/kaggle_env_build.sh) in a Kaggle run
# started with `kaggle_run.sh run --env isaac62 --setup "bash tools/isaac/kaggle_env_restore.sh"`:
# the venv, and Vulkan for the cached driver. If Kaggle's driver is not the cached one, Vulkan
# is reinstalled for the running driver (the run still works) and the cache is marked stale:
# $KGT_OUT/ISAAC_ENV_STALE and a line "ISAAC_ENV_STALE: ..." (tools/isaac/kaggle_isaac.sh then
# rebuilds it). The RTX shader cache is per driver too: a stale cache recompiles (~10 min).
set -uo pipefail
OUT=${KGT_OUT:-/kaggle/working}
t=$(date +%s)
A=$(find /kaggle/input -name isaac-venv.tar.zst 2>/dev/null | head -1)
[ -n "$A" ] || { echo "[restore] isaac-venv.tar.zst not found under /kaggle/input"; exit 1; }
DIR=$(dirname "$A")
unz() {   # zstd -> stdout; zstd may be missing before the Vulkan archive brings it
  if command -v zstd >/dev/null; then zstd -dc -T0 "$1"
  else python3 -c "import zstandard,sys; zstandard.ZstdDecompressor().copy_stream(open(sys.argv[1],'rb'), sys.stdout.buffer)" "$1"; fi
}
command -v zstd >/dev/null || python3 -c "import zstandard" 2>/dev/null || apt-get install -y -qq zstd >/dev/null 2>&1 \
  || pip install -q zstandard
[ -f "$DIR/nvidia-vk.tar.zst" ] && unz "$DIR/nvidia-vk.tar.zst" | tar -x -C /
unz "$A" | tar -x -C /
echo "[restore] venv $(numfmt --to=iec $(stat -c %s "$A")) -> $(du -sh /tmp/kgt-venv | cut -f1) in $(( $(date +%s) - t )) s"
# test hook: pretend the cache was built on another driver (exercises the stale path)
if [ -n "${ISAAC_ENV_ASSUME_CACHED_DRIVER:-}" ] && [ -f /opt/nvidia-vk/DRIVER_VERSION ]; then
  echo "$ISAAC_ENV_ASSUME_CACHED_DRIVER" > /opt/nvidia-vk/DRIVER_VERSION
fi
t=$(date +%s)
mkdir -p reports/isaac
source tools/isaac/vulkan_setup.sh > reports/isaac/vulkan_setup.log 2>&1
grep -E "VULKAN|\[vk\] (DRIVER|driver lib|installed)" reports/isaac/vulkan_setup.log
echo "[restore] vulkan $(( $(date +%s) - t )) s"
cached=${ISAAC_ENV_ASSUME_CACHED_DRIVER:-}
[ -n "$cached" ] || cached=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('driver',''))" "$DIR/env.json" 2>/dev/null)
now=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
if [ -n "$VK_DRIVER_CHANGED" ] || { [ -n "$cached" ] && [ "$cached" != "$now" ]; }; then
  msg="driver ${cached:-?} -> $now: rebuild the environment (tools/isaac/kaggle_isaac.sh rebuild)"
  echo "$msg" > "$OUT/ISAAC_ENV_STALE"
  echo "ISAAC_ENV_STALE: $msg"
fi
grep -q "VULKAN: ok" reports/isaac/vulkan_setup.log || echo "[restore] WARNING: no Vulkan (physics still works; no RTX images)"
