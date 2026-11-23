#!/usr/bin/env bash
# Vulkan only (no Isaac): driver libraries, ICD, loader diagnostics.
set -uo pipefail
mkdir -p reports/isaac
source tools/isaac/vulkan_setup.sh > reports/isaac/vulkan_setup.log 2>&1; grep -v "ldconfig.real" reports/isaac/vulkan_setup.log
cp /tmp/vulkaninfo.txt /tmp/vulkan_loader_debug.txt reports/isaac/ 2>/dev/null
cat /proc/driver/nvidia/version 2>&1 | head -2; ls -la /dev/nvidia* | head; grep -c . /proc/devices; grep nvidia /proc/devices
ls /tmp/nvdrv/x | grep -iE "icd|vulkan|layers" ; nvidia-smi -q | grep -iE "Display|Persistence|Compute Mode|vGPU|Virtualization" | head
grep -q "VULKAN: ok" reports/isaac/vulkan_setup.log
