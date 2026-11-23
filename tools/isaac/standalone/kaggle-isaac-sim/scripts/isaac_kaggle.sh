#!/usr/bin/env bash
# Isaac Sim on a Kaggle GPU (2 x T4) from a machine without a GPU, for any public GitHub repo.
#
#   isaac_kaggle.sh check                          # CLI, credentials, repo (no quota used)
#   isaac_kaggle.sh status                         # what the cache holds (env.json only)
#   isaac_kaggle.sh rebuild [--wait]               # build / rebuild the cache (GPU, ~30 min + queue)
#   isaac_kaggle.sh run --cmd "CMD" [--setup "CMD"] [--slug S] [--timeout 60m] [--collect GLOB]... [--detach]
#   isaac_kaggle.sh fetch [--slug S]               # after run --detach
#
# Run it from the root of the git repository (its pushed commit is cloned on Kaggle). On
# Kaggle, CMD runs at the repository root after the cache is restored, with Isaac Sim's
# Python first on PATH, OMNI_KIT_ACCEPT_EULA=YES, ISAAC_KIT=/tmp/isaac-kit (this skill's
# scripts/remote: probe.py, render.py, isaac_util.py) and $ISAAC_KIT on PYTHONPATH.
# --setup runs after the restore (e.g. "pip install -e ." for the project itself).
#
# Environment:
#   ISAAC_KAGGLE_ENV       cache name (default isaac62: kernel kgt-env-isaac62)
#   ISAAC_VERSION          Isaac Sim pip version for rebuild (default 6.2.0.0)
#   ISAAC_EXTRA_PIP        packages added to the cache at rebuild (default: numpy pyyaml trimesh pillow pytest zstandard)
#   ISAAC_AUTOREBUILD=0    do not rebuild automatically when a run reports a changed driver
#   ISAAC_RESTORE_ENV      "VAR=value ..." put before the restore (test hook: ISAAC_ENV_ASSUME_CACHED_DRIVER=0.0)
#   KGT_RUN                another kaggle_run.sh (default: the copy bundled in scripts/kgt)
#   KAGGLE_API_TOKEN, or KAGGLE_USERNAME + KAGGLE_KEY, or ~/.kaggle/kaggle.json
set -uo pipefail
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KGT_RUN=${KGT_RUN:-$SELF/kgt/kaggle_run.sh}
ENV_NAME=${ISAAC_KAGGLE_ENV:-isaac62}
ENV_SLUG="kgt-env-$ENV_NAME"
KIT=/tmp/isaac-kit
say() { printf '[isaac] %s\n' "$*" >&2; }
[ -f "$KGT_RUN" ] || { say "kaggle_run.sh not found ($KGT_RUN)"; exit 2; }
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || { say "run inside the git repository whose pushed commit Kaggle should clone"; exit 2; }
cd "$ROOT"

# the Kaggle-side scripts travel inside the setup command (the clone does not have them)
KIT_B64=$(tar -C "$SELF/remote" -czf - --exclude=__pycache__ . | base64 -w0)
KIT_SETUP="rm -rf $KIT && mkdir -p $KIT && echo $KIT_B64 | base64 -d | tar -xz -C $KIT"
CMD_ENV="export OMNI_KIT_ACCEPT_EULA=YES ISAAC_KIT=$KIT PYTHONPATH=$KIT\${PYTHONPATH:+:\$PYTHONPATH};"

kaggle_user() {
  [ -n "${KGT_USER:-}" ] && { echo "$KGT_USER"; return; }
  [ -n "${KAGGLE_USERNAME:-}" ] && { echo "$KAGGLE_USERNAME"; return; }
  local f="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
  [ -f "$f" ] && python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("username",""))' "$f" && return
  kaggle config view 2>/dev/null | sed -n 's/^- username: *//p' | head -1
}

stale_check() {   # $1: the fetched output directory of a run
  local d=${1:-}
  [ -n "$d" ] || return 0
  if [ -f "$d/ISAAC_ENV_STALE" ] || grep -q "ISAAC_ENV_STALE" "$d/test_output.log" 2>/dev/null; then
    say "cache is stale: $(cat "$d/ISAAC_ENV_STALE" 2>/dev/null || grep -m1 ISAAC_ENV_STALE "$d/test_output.log")"
    if [ "${ISAAC_AUTOREBUILD:-1}" = 1 ]; then say "rebuilding the cache (detached)"; rebuild; fi
  fi
  if grep -q "isaac-venv.tar.zst not found" "$d/test_output.log" 2>/dev/null; then
    say "no cache on Kaggle yet (or its last build failed or was cancelled): isaac_kaggle.sh rebuild"
  fi
}

rebuild() {
  bash "$KGT_RUN" run --detach --python 3.12 --slug "$ENV_SLUG" --timeout 100m "$@" \
    --setup "$KIT_SETUP && ISAAC_VERSION='${ISAAC_VERSION:-6.2.0.0}' ISAAC_EXTRA_PIP='${ISAAC_EXTRA_PIP:-numpy pyyaml trimesh pillow pytest zstandard}' bash $KIT/setup.sh" \
    --cmd "ISAAC_KIT=$KIT ISAAC_ENV_NAME=$ENV_NAME ISAAC_EXTRA_PIP='${ISAAC_EXTRA_PIP:-}' bash $KIT/env_build.sh"
  say "the cache kernel $ENV_SLUG is queued; it is ready when 'isaac_kaggle.sh status' shows env.json. Never 'fetch' it (10+ GB)."
}

latest_out() { ls -td .kaggle-gpu-test/out/"$1"-* 2>/dev/null | head -1; }

cmd=${1:-}; shift || true
case "$cmd" in
  check) bash "$KGT_RUN" check;;
  run)
    slug="isaac-run"; detach=0; usercmd=""; usersetup=""; args=()
    while [ $# -gt 0 ]; do
      case "$1" in
        --slug) slug=$2; shift 2;;
        --detach) detach=1; shift;;
        --cmd) usercmd=$2; shift 2;;
        --setup) usersetup=$2; shift 2;;
        --env|--python) say "$1 is fixed by the cache ($ENV_NAME)"; exit 2;;
        *) args+=("$1"); shift;;
      esac
    done
    [ -n "$usercmd" ] || { say "--cmd is required"; exit 2; }
    case "$slug" in kgt-env-*) say "kgt-env-* slugs are cache kernels; pick another --slug"; exit 2;; esac
    setup="$KIT_SETUP && ${ISAAC_RESTORE_ENV:+$ISAAC_RESTORE_ENV }ISAAC_KIT=$KIT bash $KIT/env_restore.sh"
    [ -n "$usersetup" ] && setup="$setup && $usersetup"
    run=(bash "$KGT_RUN" run --env "$ENV_NAME" --slug "$slug" --setup "$setup" --cmd "$CMD_ENV $usercmd" "${args[@]}")
    if [ $detach = 1 ]; then
      "${run[@]}" --detach; rc=$?
      say "then: isaac_kaggle.sh fetch --slug $slug"
      exit $rc
    fi
    "${run[@]}"; rc=$?
    stale_check "$(latest_out "$slug")"
    exit $rc;;
  fetch)
    slug="isaac-run"; [ "${1:-}" = "--slug" ] && slug=$2
    case "$slug" in kgt-env-*) say "never fetch a cache kernel (10+ GB): use 'isaac_kaggle.sh status'"; exit 2;; esac
    bash "$KGT_RUN" fetch --slug "$slug"; rc=$?
    stale_check "$(latest_out "$slug")"
    exit $rc;;
  rebuild)
    wait=0; rargs=()
    while [ $# -gt 0 ]; do case "$1" in --wait) wait=1; shift;; *) rargs+=("$1"); shift;; esac; done
    rebuild "${rargs[@]}"
    if [ $wait = 1 ]; then
      until kaggle kernels status "$(kaggle_user)/$ENV_SLUG" 2>&1 | grep -qiE "COMPLETE|ERROR|CANCEL"; do sleep 180; done
      "$0" status
    fi;;
  status)
    # never `fetch` the env kernel: its output is a 10+ GB archive. Only env.json.
    d=$(mktemp -d); user=$(kaggle_user)
    kaggle kernels status "$user/$ENV_SLUG" 2>&1 | tail -1
    kaggle kernels output "$user/$ENV_SLUG" -p "$d" --file-pattern '^env\.json$' >/dev/null 2>&1
    cat "$d/env.json" 2>/dev/null || say "no env.json in the latest version of $ENV_SLUG (never built, or that version is queued, running, failed or cancelled; runs mount the last completed version, if any)"
    rm -rf "$d";;
  *) sed -n 2,26p "$0"; exit 2;;
esac
