#!/usr/bin/env bash
# Vulkan for Isaac Sim's RTX renderer in a container that only has NVIDIA's compute
# libraries (Kaggle: NVIDIA_DRIVER_CAPABILITIES without "graphics", no Vulkan ICD).
# Installs the Vulkan loader and the user-space graphics libraries of the *running*
# driver version (from NVIDIA's own installer, extracted, no kernel module), and
# registers the ICD. Prints a verdict line "VULKAN: ok|fail ...".
#   source tools/isaac/vulkan_setup.sh     (exports VK_ICD_FILENAMES, LD_LIBRARY_PATH)
set -u
VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
DST=${VK_DRIVER_DIR:-/opt/nvidia-vk}
echo "[vk] driver $VER  NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-unset}"
echo "[vk] devices: $(ls /dev/nvidia* 2>/dev/null | tr '\n' ' ')"
echo "[vk] mounted graphics libs: $(ldconfig -p | grep -cE 'libGLX_nvidia|libnvidia-glcore|libnvidia-rtcore') ; icd: $(ls /etc/vulkan/icd.d /usr/share/vulkan/icd.d 2>/dev/null | tr '\n' ' ')"
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libvulkan1 vulkan-tools >/dev/null) 2>&1 | tail -2
if ! ldconfig -p | grep -q libGLX_nvidia.so.0; then
  mkdir -p "$DST" /tmp/nvdrv && cd /tmp/nvdrv
  for u in "https://us.download.nvidia.com/tesla/$VER/NVIDIA-Linux-x86_64-$VER.run" \
           "https://us.download.nvidia.com/XFree86/Linux-x86_64/$VER/NVIDIA-Linux-x86_64-$VER.run"; do
    curl -fsSL -o drv.run "$u" && { echo "[vk] downloaded $u"; break; }
  done
  sh drv.run --extract-only --target x >/dev/null 2>&1 || echo "[vk] extract failed"
  # graphics/RT user-space libraries of this driver version (compute ones are mounted already)
  for f in x/libGLX_nvidia.so.$VER x/libnvidia-glcore.so.$VER x/libnvidia-glvkspirv.so.$VER \
           x/libnvidia-gpucomp.so.$VER x/libnvidia-rtcore.so.$VER x/libnvoptix.so.$VER \
           x/libEGL_nvidia.so.$VER x/libnvidia-eglcore.so.$VER x/libnvidia-glsi.so.$VER \
           x/libnvidia-tls.so.$VER x/libnvidia-ngx.so.$VER; do
    [ -f "$f" ] && cp "$f" "$DST/"
  done
  cd "$DST"
  for f in *.so.$VER; do b=${f%.so.$VER}; ln -sf "$f" "$b.so.1"; ln -sf "$f" "$b.so.0"; ln -sf "$f" "$b.so"; done
  api=$(grep -o '"api_version" *: *"[0-9.]*"' /tmp/nvdrv/x/nvidia_icd.json* 2>/dev/null | head -1 | grep -o '[0-9.]*"$' | tr -d '"')
  cat > "$DST/nvidia_icd.json" <<JSON
{"file_format_version": "1.0.1", "ICD": {"library_path": "$DST/libGLX_nvidia.so.0", "api_version": "${api:-1.3.0}"}}
JSON
  mkdir -p /etc/vulkan/icd.d && cp "$DST/nvidia_icd.json" /etc/vulkan/icd.d/nvidia_icd.json
  echo "$DST" > /etc/ld.so.conf.d/zz-nvidia-vk.conf && ldconfig     # every process finds them
  export LD_LIBRARY_PATH="$DST${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export VK_ICD_FILENAMES="$DST/nvidia_icd.json"
  echo "[vk] installed: $(ls $DST | grep -c "so.$VER") libs into $DST (api ${api:-?})"
  cd - >/dev/null
fi
if vulkaninfo --summary > /tmp/vulkaninfo.txt 2>&1; then
  echo "VULKAN: ok $(grep -m2 -E 'deviceName|apiVersion' /tmp/vulkaninfo.txt | tr -s ' ' | tr '\n' ' ')"
else
  echo "VULKAN: fail $(tail -3 /tmp/vulkaninfo.txt | tr '\n' ' ')"
fi
