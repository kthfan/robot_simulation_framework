#!/usr/bin/env bash
# Isaac Sim on Kaggle (2 x T4) from a machine without a GPU: run a command in the cached
# Isaac environment, keep that cache current, report it.
#
#   tools/isaac/kaggle_isaac.sh run --cmd "CMD" [--slug S] [--timeout 60m] [--collect GLOB]... [--detach]
#   tools/isaac/kaggle_isaac.sh fetch [--slug S]      # after a --detach run
#   tools/isaac/kaggle_isaac.sh rebuild [--wait]      # (re)build the cache (GPU, ~25 min)
#   tools/isaac/kaggle_isaac.sh status                # what the cache holds (env.json)
#
# `run` restores the cache (Isaac Sim 6.2 venv + RTX shader cache + Vulkan for the cached
# driver; tools/isaac/kaggle_env_restore.sh) and runs CMD at the repository root of the
# pushed commit. When the run reports ISAAC_ENV_STALE (Kaggle's NVIDIA driver changed), the
# cache is rebuilt automatically (detached; disable with KGT_ISAAC_AUTOREBUILD=0).
#
# KGT_ISAAC_RESTORE_ENV="VAR=value ..." is put before the restore command (e.g. the test hook
# ISAAC_ENV_ASSUME_CACHED_DRIVER=0.0, which makes the run see a stale cache).
#
# Needs the kaggle-gpu-test skill's kaggle_run.sh (found under ~/.claude/skills, or
# $KGT_RUN) and Kaggle credentials (KAGGLE_USERNAME + KAGGLE_API_TOKEN). Kaggle clones the
# pushed commit: push first.
set -uo pipefail
ENV_NAME=${KGT_ISAAC_ENV:-isaac62}
ENV_SLUG="kgt-env-$ENV_NAME"
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "run inside the repository"; exit 2; }
cd "$ROOT"
KGT_RUN=${KGT_RUN:-$(find "$HOME/.claude/skills" /root/.claude/skills -path '*kaggle-gpu-test/scripts/kaggle_run.sh' 2>/dev/null | head -1)}
[ -x "$KGT_RUN" ] || [ -f "$KGT_RUN" ] || { echo "kaggle_run.sh (kaggle-gpu-test skill) not found; set KGT_RUN"; exit 2; }
REPO=${KGT_REPO:-$(git remote get-url origin | sed -E 's#^git@github.com:#https://github.com/#; s#\.git$##')}
case "$REPO" in https://github.com/*) ;; *) echo "origin ($REPO) is not a public GitHub URL: set KGT_REPO"; exit 2;; esac

stale_check() {   # $1: the fetched output directory of a run
  local d=$1
  if [ -f "$d/ISAAC_ENV_STALE" ] || grep -q "ISAAC_ENV_STALE" "$d/test_output.log" 2>/dev/null; then
    echo "[isaac] cache is stale: $(cat "$d/ISAAC_ENV_STALE" 2>/dev/null || grep -m1 ISAAC_ENV_STALE "$d/test_output.log")"
    if [ "${KGT_ISAAC_AUTOREBUILD:-1}" = 1 ]; then
      echo "[isaac] rebuilding the cache (detached)"
      rebuild
    fi
  fi
}

rebuild() {
  bash "$KGT_RUN" run --detach --python 3.12 --slug "$ENV_SLUG" --timeout 100m --repo "$REPO" \
    --setup "bash tools/isaac/kaggle_setup.sh" --cmd "bash tools/isaac/kaggle_env_build.sh"
}

cmd=${1:-}; shift || true
case "$cmd" in
  run)
    slug="rsf-isaac-run"; detach=0; args=()
    while [ $# -gt 0 ]; do
      case "$1" in
        --slug) slug=$2; shift 2;;
        --detach) detach=1; shift;;
        *) args+=("$1"); shift;;
      esac
    done
    if [ $detach = 1 ]; then
      bash "$KGT_RUN" run --detach --env "$ENV_NAME" --slug "$slug" --repo "$REPO" \
        --setup "${KGT_ISAAC_RESTORE_ENV:+$KGT_ISAAC_RESTORE_ENV }bash tools/isaac/kaggle_env_restore.sh" "${args[@]}"; rc=$?
      echo "[isaac] then: tools/isaac/kaggle_isaac.sh fetch --slug $slug"
      exit $rc
    fi
    bash "$KGT_RUN" run --env "$ENV_NAME" --slug "$slug" --repo "$REPO" \
      --setup "${KGT_ISAAC_RESTORE_ENV:+$KGT_ISAAC_RESTORE_ENV }bash tools/isaac/kaggle_env_restore.sh" "${args[@]}"
    rc=$?
    stale_check "$(ls -td .kaggle-gpu-test/out/"$slug"-* 2>/dev/null | head -1)"
    exit $rc;;
  fetch)
    slug="rsf-isaac-run"; [ "${1:-}" = "--slug" ] && slug=$2
    bash "$KGT_RUN" fetch --slug "$slug"; rc=$?
    stale_check "$(ls -td .kaggle-gpu-test/out/"$slug"-* 2>/dev/null | head -1)"
    exit $rc;;
  rebuild)
    rebuild
    if [ "${1:-}" = "--wait" ]; then
      until kaggle kernels status "$(kaggle config view 2>/dev/null | awk '/username/{print $NF}' || echo "$KAGGLE_USERNAME")/$ENV_SLUG" 2>&1 \
          | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 180; done
      "$0" status
    fi;;
  status)
    # never `fetch` the env kernel: its output is a 10+ GB archive. Only the small files.
    d=$(mktemp -d)
    user=${KAGGLE_USERNAME:-$(kaggle config view 2>/dev/null | awk '/username/{print $NF}')}
    kaggle kernels status "$user/$ENV_SLUG" 2>&1 | tail -1
    kaggle kernels output "$user/$ENV_SLUG" -p "$d" --file-pattern '^env\.json$' >/dev/null 2>&1
    cat "$d/env.json" 2>/dev/null || echo "[isaac] no env.json in the latest version of $ENV_SLUG (never built, or that version is queued, running, failed or cancelled; runs mount the last completed version, if any)"
    rm -rf "$d";;
  *)
    sed -n 2,17p "$0"; exit 2;;
esac
