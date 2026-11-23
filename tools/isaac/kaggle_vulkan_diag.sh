#!/usr/bin/env bash
# Vulkan only (no Isaac): driver libraries, ICD, loader diagnostics.
set -uo pipefail
mkdir -p reports/isaac
source tools/isaac/vulkan_setup.sh > reports/isaac/vulkan_setup.log 2>&1; grep -v "ldconfig.real" reports/isaac/vulkan_setup.log
cp /tmp/vulkaninfo.txt /tmp/vulkan_loader_debug.txt reports/isaac/ 2>/dev/null
cat /proc/driver/nvidia/version 2>&1 | head -2; ls -la /dev/nvidia* | head; grep -c . /proc/devices; grep nvidia /proc/devices
ls /tmp/nvdrv/x | grep -iE "icd|vulkan|layers" ; nvidia-smi -q | grep -iE "Display|Persistence|Compute Mode|vGPU|Virtualization" | head
ls /opt/nvidia-vk | grep -c "so.5"; ls /opt/nvidia-vk | grep "so.580" | tr '\n' ' '; echo
python3 -c "import os; fd=os.open('/dev/nvidia-modeset', os.O_RDWR); print('open modeset ok')" 2>&1 | tail -1
(apt-get install -y -qq strace >/dev/null 2>&1 && strace -f -e trace=openat,ioctl -o /tmp/vk.strace vulkaninfo --summary >/dev/null 2>&1;
 grep -E "nvidia|EACCES|EPERM|ENOENT.*(nvidia|vulkan)" /tmp/vk.strace | grep -vE "ENOENT.*(python|locale)" | grep -E "= -1|nvidia-modeset|nvidiactl" | head -30; cp /tmp/vk.strace reports/isaac/)
grep -q "VULKAN: ok" reports/isaac/vulkan_setup.log
