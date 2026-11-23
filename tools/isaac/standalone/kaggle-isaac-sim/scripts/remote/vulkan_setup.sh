#!/usr/bin/env bash
# Vulkan for Isaac Sim's RTX renderer in a container that only has NVIDIA's compute
# libraries (Kaggle: NVIDIA_DRIVER_CAPABILITIES=compute,utility, no Vulkan ICD).
#
# Installs, user space only (the kernel module is the host's):
#   * the Vulkan loader and vulkaninfo (apt: libvulkan1 vulkan-tools);
#   * the graphics libraries of the *running* driver version, from NVIDIA's own
#     installer (extracted, every library the container lacks), into $DST;
# and registers them: ld.so.conf, the Vulkan ICD and layers, NVIDIA's EGL vendor for
# glvnd (the Vulkan driver finds its GPU through EGL; without it glvnd falls back to
# Mesa and vkCreateInstance fails with ERROR_INCOMPATIBLE_DRIVER).
#
# Cached: $DST/DRIVER_VERSION records the driver the libraries belong to. When $DST was
# restored from a cache (env_build.sh) and matches the running
# driver, nothing is downloaded. When the driver changed, the libraries are reinstalled
# and VK_DRIVER_CHANGED="<cached> -> <running>" is exported (the cache is stale).
#
# Prints "VULKAN: ok|fail ..." last.     source tools/isaac/vulkan_setup.sh
set -u
_VK_CWD=$PWD
VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
DST=${VK_DRIVER_DIR:-/opt/nvidia-vk}
STAMP="$DST/DRIVER_VERSION"
VK_DRIVER_CHANGED=""
echo "[vk] driver $VER  NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-unset}"

# 1. the loader (cached with the libraries: no apt when present)
if command -v vulkaninfo >/dev/null && ldconfig -p | grep -q 'libvulkan.so.1 '; then
  echo "[vk] loader present"
else
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libvulkan1 vulkan-tools >/dev/null) 2>&1 | tail -2
fi

# 2. the driver's graphics libraries, for this driver version
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" != "$VER" ]; then
  VK_DRIVER_CHANGED="$(cat "$STAMP") -> $VER"
  echo "[vk] DRIVER CHANGED: $VK_DRIVER_CHANGED (cached libraries discarded)"
  rm -rf "$DST"
fi
export VK_DRIVER_CHANGED
if [ -f "$STAMP" ]; then
  echo "[vk] driver libraries cached ($(ls "$DST" | grep -c "so.$VER$") for $VER)"
else
  t=$(date +%s)
  mkdir -p "$DST/json" /tmp/nvdrv && cd /tmp/nvdrv
  for u in "https://us.download.nvidia.com/tesla/$VER/NVIDIA-Linux-x86_64-$VER.run" \
           "https://us.download.nvidia.com/XFree86/Linux-x86_64/$VER/NVIDIA-Linux-x86_64-$VER.run"; do
    curl -fsSL -o drv.run "$u" && { echo "[vk] downloaded $u"; break; }
  done
  sh drv.run --extract-only --target x >/dev/null 2>&1 || echo "[vk] extract failed"
  # every user-space library of this driver version the container does not have
  # (the compute ones are mounted already; never replace a mounted one)
  for f in x/lib*.so.$VER; do
    b=$(basename "$f")
    m=$(ldconfig -p | grep -m1 -E "^[[:space:]]*${b%.so.$VER}\.so" | awk '{print $NF}')
    [ -n "$m" ] && [ -e "$m" ] && case "$m" in "$DST"/*) ;; *) continue;; esac
    cp "$f" "$DST/"
  done
  cp x/nvidia_icd.json x/nvidia_layers.json x/*_nvidia_*.json "$DST/json/" 2>/dev/null
  (cd "$DST" && for f in *.so.$VER; do b=${f%.so.$VER}; ln -sf "$f" "$b.so.1"; ln -sf "$f" "$b.so.0"; ln -sf "$f" "$b.so"; done)
  echo "$VER" > "$STAMP"
  rm -rf /tmp/nvdrv
  echo "[vk] installed $(ls "$DST" | grep -c "so.$VER$") libraries into $DST in $(( $(date +%s) - t )) s"
fi
cd "$_VK_CWD"

# 3. registration (outside $DST: always rewritten, cheap)
api=$(grep -o '"api_version" *: *"[0-9.]*"' "$DST"/json/nvidia_icd.json 2>/dev/null | head -1 | grep -o '[0-9.]*"$' | tr -d '"')
cat > "$DST/nvidia_icd.json" <<JSON
{"file_format_version": "1.0.1", "ICD": {"library_path": "$DST/libGLX_nvidia.so.0", "api_version": "${api:-1.3.0}"}}
JSON
mkdir -p /etc/vulkan/icd.d /etc/vulkan/implicit_layer.d /usr/share/glvnd/egl_vendor.d /usr/share/egl/egl_external_platform.d
cp "$DST/nvidia_icd.json" /etc/vulkan/icd.d/nvidia_icd.json
[ -f "$DST/json/nvidia_layers.json" ] && \
  sed "s#libGLX_nvidia.so.0#$DST/libGLX_nvidia.so.0#" "$DST/json/nvidia_layers.json" > /etc/vulkan/implicit_layer.d/nvidia_layers.json
echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' \
  > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
for j in "$DST"/json/*_nvidia_*.json; do [ -f "$j" ] && cp "$j" /usr/share/egl/egl_external_platform.d/; done
echo "$DST" > /etc/ld.so.conf.d/zz-nvidia-vk.conf && ldconfig 2>/dev/null
export LD_LIBRARY_PATH="$DST${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES="$DST/nvidia_icd.json"
# the Vulkan driver looks for the modeset device; create it if the container lacks it
if [ ! -e /dev/nvidia-modeset ] && [ "${VK_MKNOD_MODESET:-1}" = 1 ]; then
  mknod -m 666 /dev/nvidia-modeset c 195 254 2>/dev/null || echo "[vk] cannot create /dev/nvidia-modeset"
fi

# 4. verdict
if vulkaninfo --summary > /tmp/vulkaninfo.txt 2>&1; then
  echo "VULKAN: ok $(grep -m2 -E 'deviceName|apiVersion' /tmp/vulkaninfo.txt | tr -s ' \t' ' ' | tr '\n' ' ')"
else
  echo "VULKAN: fail $(tail -3 /tmp/vulkaninfo.txt | tr '\n' ' ')"
  VK_LOADER_DEBUG=error,warn,driver vulkaninfo --summary > /tmp/vulkan_loader_debug.txt 2>&1
  grep -iE "error|fail" /tmp/vulkan_loader_debug.txt | sort -u | head -10
  echo "[vk] missing for libGLX_nvidia: $(ldd "$DST/libGLX_nvidia.so.0" 2>&1 | grep 'not found' | tr -s ' \t' ' ' | tr '\n' ';')"
fi
