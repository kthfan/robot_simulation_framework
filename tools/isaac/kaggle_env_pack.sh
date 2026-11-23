#!/usr/bin/env bash
# Kaggle (CPU kernel, no GPU quota): pack the Isaac Sim venv as a prebuilt environment that
# `kaggle_run.sh run --env <name>` restores. The venv is ~24 GB, over Kaggle's 20 GB output
# limit, so it is written zstd-compressed under the runner's name (env.tar): GNU tar
# recognises the compression when it extracts.
#   kaggle_run.sh run --cpu --python 3.12 --slug kgt-env-isaac62 --timeout 60m \
#       --setup "bash tools/isaac/kaggle_setup.sh" --cmd "bash tools/isaac/kaggle_env_pack.sh isaac62"
set -euo pipefail
NAME=${1:-isaac62}
OUT=${KGT_OUT:-/kaggle/working}
VENV=${KGT_VENV:-/tmp/kgt-venv}
UVPY=${KGT_UV_PY:-/tmp/kgt-uv-python}
command -v zstd >/dev/null || (apt-get install -y -qq zstd >/dev/null 2>&1 || pip install -q zstd)
find "$VENV" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "[pack] venv $(du -sh "$VENV" | cut -f1), python $(du -sh "$UVPY" | cut -f1), tar $(tar --version | head -1)"
t=$(date +%s)
tar -C / -cf - "${VENV#/}" "${UVPY#/}" | zstd -T0 -3 -q -o "$OUT/env.tar"
bytes=$(stat -c %s "$OUT/env.tar")
echo "[pack] env.tar (zstd) $(numfmt --to=iec $bytes) in $(( $(date +%s) - t )) s"
"$VENV/bin/pip" freeze > "$OUT/env-freeze.txt" 2>/dev/null || true
cat > "$OUT/env.json" <<JSON
{"name": "$NAME", "python": "$("$VENV/bin/python" -V 2>&1)", "torch": "", "compression": "zstd",
 "setup": "bash tools/isaac/kaggle_setup.sh", "venv_path": "$VENV", "uv_python_dir": "$UVPY", "bytes": $bytes,
 "built_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
[ "$bytes" -lt 20000000000 ] || { echo "[pack] too big for Kaggle's 20 GB output"; exit 1; }
