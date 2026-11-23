#!/usr/bin/env bash
# Kaggle: Vulkan from NVIDIA's own installer, then an Isaac RTX render of the strict cell.
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src
mkdir -p reports/isaac
source tools/isaac/vulkan_setup.sh > reports/isaac/vulkan_setup.log 2>&1; grep -E "^\[vk\]|VULKAN" reports/isaac/vulkan_setup.log
echo "zstd: $(command -v zstd)  tar: $(tar --version | head -1)"
cp /tmp/vulkaninfo.txt reports/isaac/ 2>/dev/null
timeout 600 python - <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from rsf.model.layout import load_world
from rsf.compile.usd import world_stage
world_stage(load_world("cells/tm_tray/layouts/strict.yaml"), "reports/isaac/strict.usda")
app.close()
PY
timeout 2400 python tools/isaac/render_probe.py --usd reports/isaac/strict.usda --png reports/isaac/render.png \
    --out reports/isaac/render_probe.json; rc=$?
grep -E "vkCreateInstance|RTX|GPU Foundation|rtx.*Error|Vulkan" -m 20 ~/.local/share/ov/data/Kit/*/*/logs/*.log 2>/dev/null | tail -20
du -sh /tmp/kgt-venv ~/.cache/ov ~/.local/share/ov ~/.nvidia-omniverse 2>/dev/null
rm -f reports/isaac/strict.usda
exit $rc
