#!/usr/bin/env bash
# Restore the prebuilt Isaac Sim environment (tools/isaac/kaggle_env_build.sh) in a Kaggle
# run started with `kaggle_run.sh run --env isaac62 --setup "bash tools/isaac/kaggle_env_restore.sh ..."`.
# Also sets up Vulkan (tools/isaac/vulkan_setup.sh) for RTX rendering.
set -uo pipefail
t=$(date +%s)
A=$(find /kaggle/input -name isaac-venv.tar.zst 2>/dev/null | head -1)
[ -n "$A" ] || { echo "[restore] isaac-venv.tar.zst not found under /kaggle/input"; exit 1; }
command -v zstd >/dev/null || apt-get install -y -qq zstd >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq zstd; }
zstd -dc -T0 "$A" | tar -x -C /
echo "[restore] $(numfmt --to=iec $(stat -c %s "$A")) -> $(du -sh /tmp/kgt-venv | cut -f1) in $(( $(date +%s) - t )) s"
t=$(date +%s)
mkdir -p reports/isaac
source tools/isaac/vulkan_setup.sh > reports/isaac/vulkan_setup.log 2>&1
grep VULKAN reports/isaac/vulkan_setup.log; echo "[restore] vulkan $(( $(date +%s) - t )) s"
