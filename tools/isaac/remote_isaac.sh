#!/usr/bin/env bash
# Isaac Sim on another computer (the GPU box), the rest of the work here, over SSH.
#
#   tools/isaac/remote_isaac.sh check                     # SSH, remote venv, isaacsim, GPU
#   tools/isaac/remote_isaac.sh sync                      # this checkout's files -> the GPU box
#   tools/isaac/remote_isaac.sh run --cmd "CMD" [--send PATH]... [--collect PATH]...
#   tools/isaac/remote_isaac.sh fetch PATH...             # remote -> here (paths relative to the repo)
#
# `run` syncs, sends the extra files (--send: git-ignored inputs such as a plan), runs CMD at the
# remote repository root in the Isaac venv (OMNI_KIT_ACCEPT_EULA=YES, PYTHONPATH=src), and
# brings back --collect paths (files or directories, relative to the repo), even when CMD fails.
#
# Configuration (environment):
#   RSF_ISAAC_HOST   SSH destination: user@host or a ~/.ssh/config alias (required)
#   RSF_ISAAC_DIR    the repository on the GPU box (default: rsf; relative to the remote home)
#   RSF_ISAAC_VENV   the Isaac venv on the GPU box (default: venvs/rsf-isaac)
#   RSF_ISAAC_SSH    the ssh command (default: ssh; e.g. "ssh -p 2222 -i ~/.ssh/gpu_key")
#
# What is synced: every file git tracks plus untracked files that are not ignored (code,
# layouts, assets, teach tables, plan summaries), and a REVISION file naming the commit and
# whether the checkout had local changes. Plans and reports are git-ignored: --send them.
set -uo pipefail
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "run inside the repository"; exit 2; }
cd "$ROOT"
HOST=${RSF_ISAAC_HOST:-}
RDIR=${RSF_ISAAC_DIR:-rsf}
RVENV=${RSF_ISAAC_VENV:-venvs/rsf-isaac}
SSH=${RSF_ISAAC_SSH:-ssh}
say() { printf '[remote] %s\n' "$*" >&2; }
[ -n "$HOST" ] || { say "set RSF_ISAAC_HOST (user@host or an ~/.ssh/config alias)"; exit 2; }
command -v rsync >/dev/null || { say "rsync is needed here and on the GPU box"; exit 2; }

remote() {   # run a shell command on the GPU box (bash -lc, at the remote home)
  # shellcheck disable=SC2086
  $SSH "$HOST" "bash -lc $(printf '%q' "$1")"
}

sync_tree() {
  local list
  list=$(mktemp)
  git ls-files -co --exclude-standard > "$list"
  printf '%s%s\n' "$(git rev-parse HEAD)" "$([ -n "$(git status --porcelain)" ] && echo ' +local changes')" > .rsf_revision
  echo .rsf_revision >> "$list"
  remote "mkdir -p $(printf '%q' "$RDIR")" || return 3
  rsync -az --files-from="$list" -e "$SSH" ./ "$HOST:$RDIR/" || { rm -f "$list" .rsf_revision; return 3; }
  rm -f "$list" .rsf_revision
  say "synced $(git ls-files -co --exclude-standard | wc -l) files to $HOST:$RDIR ($(git rev-parse --short HEAD))"
}

send() {
  local p
  for p in "$@"; do
    [ -e "$p" ] || { say "--send $p: not found"; return 2; }
    rsync -az -R -e "$SSH" "./$p" "$HOST:$RDIR/" || return 3
    say "sent $p"
  done
}

fetch() {
  local p rc=0
  for p in "$@"; do
    mkdir -p "$(dirname "$p")"
    if rsync -az -e "$SSH" "$HOST:$RDIR/$p" "$(dirname "$p")/"; then say "fetched $p"; else say "could not fetch $p"; rc=3; fi
  done
  return $rc
}

cmd=${1:-}; shift || true
case "$cmd" in
  check)
    remote "(cd $(printf '%q' "$RDIR") 2>/dev/null && echo repo: \$(pwd) \$(cat .rsf_revision 2>/dev/null)) || echo 'repo: not synced yet';
            source $(printf '%q' "$RVENV")/bin/activate && python -V &&
            python -c 'import importlib.util as u; print(\"isaacsim:\", \"yes\" if u.find_spec(\"isaacsim\") else \"MISSING\")';
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo 'no NVIDIA GPU visible'";;
  sync) sync_tree;;
  fetch) fetch "$@";;
  run)
    c=""; sends=(); collects=()
    while [ $# -gt 0 ]; do
      case "$1" in
        --cmd) c=$2; shift 2;;
        --send) sends+=("$2"); shift 2;;
        --collect) collects+=("$2"); shift 2;;
        *) say "unknown option $1"; exit 2;;
      esac
    done
    [ -n "$c" ] || { say "--cmd is required"; exit 2; }
    sync_tree || exit 3
    [ ${#sends[@]} -eq 0 ] || send "${sends[@]}" || exit $?
    say "running on $HOST: $c"
    t=$(date +%s)
    # the venv first: RSF_ISAAC_VENV may be relative to the remote home
    remote "source $(printf '%q' "$RVENV")/bin/activate && cd $(printf '%q' "$RDIR") &&
            export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src && $c"
    rc=$?
    say "exit $rc after $(( $(date +%s) - t )) s"
    [ ${#collects[@]} -eq 0 ] || fetch "${collects[@]}"
    exit $rc;;
  *) sed -n 2,23p "$0"; exit 2;;
esac
