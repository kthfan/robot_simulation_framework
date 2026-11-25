#!/usr/bin/env bash
# Isaac Sim on another computer (the GPU box), the rest of the work here, over SSH.
#
#   tools/isaac/remote_isaac.sh check                     # SSH, remote venv, isaacsim, GPU
#   tools/isaac/remote_isaac.sh sync                      # this checkout's files -> the GPU box
#   tools/isaac/remote_isaac.sh run --cmd "CMD" [--send PATH]... [--collect PATH]... [--view PORT]
#   tools/isaac/remote_isaac.sh fetch PATH...             # remote -> here (paths relative to the repo)
#   tools/isaac/remote_isaac.sh stop                      # end runs left on the GPU box (Ctrl-C, then kill)
#
# `run` syncs, sends the extra files (--send: git-ignored inputs such as a plan), runs CMD at the
# remote repository root in the Isaac venv (OMNI_KIT_ACCEPT_EULA=YES, PYTHONPATH=src), and
# brings back --collect paths (files or directories, relative to the repo), even when CMD fails.
# --view PORT forwards http://localhost:PORT here to the GPU box's 127.0.0.1:PORT while CMD runs
# (the live view: CMD with `sim_workflow.py ... --live PORT`; docs/live_view.md).
# From a terminal, CMD gets one too (ssh -t): Ctrl-C here reaches it, and it ends if the
# connection drops, so no run is left serving or simulating unseen.
#
# Configuration (environment):
#   RSF_ISAAC_HOST   SSH destination: user@host or a ~/.ssh/config alias (required)
#   RSF_ISAAC_DIR    the repository on the GPU box (default: rsf; relative to the remote home)
#   RSF_ISAAC_PYTHON Isaac Sim's own interpreter on the GPU box, e.g. isaacsim/python.sh (the
#                    standalone package; relative to the remote home or absolute). When set, no
#                    venv is activated and `python` in CMD runs it.
#   RSF_ISAAC_VENV   otherwise, the pip-installed Isaac venv on the GPU box (default: venvs/rsf-isaac)
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
RPY=${RSF_ISAAC_PYTHON:-}
SSH=${RSF_ISAAC_SSH:-ssh}
say() { printf '[remote] %s\n' "$*" >&2; }
[ -n "$HOST" ] || { say "set RSF_ISAAC_HOST (user@host or an ~/.ssh/config alias)"; exit 2; }
command -v rsync >/dev/null || { say "rsync is needed here and on the GPU box"; exit 2; }

py_env() {   # the remote shell prelude that makes `python` Isaac Sim's interpreter
  if [ -n "$RPY" ]; then
    # shellcheck disable=SC2016
    printf '__py=%q; case "$__py" in /*) ;; *) __py="$HOME/$__py";; esac; [ -x "$__py" ] || { echo "no $__py"; exit 2; }; python() { "$__py" "$@"; }; export -f python; export __py; ' "$RPY"
  else
    printf 'source %q/bin/activate && ' "$RVENV"
  fi
}

remote() {   # run a shell command on the GPU box (bash -lc, at the remote home); $2: extra ssh options
  # shellcheck disable=SC2086
  $SSH ${2:-} "$HOST" "bash -lc $(printf '%q' "$1")"
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
            $(py_env) python -V &&
            (cd / && python -c 'import importlib.util as u; s = u.find_spec(\"isaacsim\"); print(\"isaacsim:\", s.origin if s and s.origin else \"MISSING\")');
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo 'no NVIDIA GPU visible'";;
  sync) sync_tree;;
  fetch) fetch "$@";;
  run)
    c=""; sends=(); collects=(); view=
    while [ $# -gt 0 ]; do
      case "$1" in
        --cmd) c=$2; shift 2;;
        --send) sends+=("$2"); shift 2;;
        --collect) collects+=("$2"); shift 2;;
        --view) view=$2; shift 2;;
        *) say "unknown option $1"; exit 2;;
      esac
    done
    [ -n "$c" ] || { say "--cmd is required"; exit 2; }
    for p in ${sends[@]+"${sends[@]}"}; do       # before anything is copied
      if [ ! -e "$p" ]; then
        say "--send $p: no such file here"
        if git check-ignore -q "$p"; then
          say "  it is git-ignored, so not in the repository: make it on this computer first"
          case "$p" in
            cells/*/plans/*) say "  (a plan: tools/plan_workflow.py assemble ... --out $p; docs/local_setup.md;"
                             say "   to try the round trip now: cells/tm_tray/plans/strict_cycle1.json.gz, in git, needs no --send)";;
          esac
        fi
        exit 2
      fi
    done
    sync_tree || exit 3
    [ ${#sends[@]} -eq 0 ] || send "${sends[@]}" || exit $?
    say "running on $HOST: $c"
    fwd=""
    if [ -n "$view" ]; then
      case "$view" in *[!0-9]*) say "--view takes a port number"; exit 2;; esac
      fwd="-L $view:127.0.0.1:$view -o ExitOnForwardFailure=yes"
      say "live view: http://localhost:$view/ (once the run serves it: --live $view in CMD)"
    fi
    # a terminal on the remote side: Ctrl-C reaches CMD, a dropped connection hangs it up
    [ -t 0 ] && [ -t 1 ] && fwd="-t $fwd"
    t=$(date +%s)
    # the interpreter first: RSF_ISAAC_PYTHON / RSF_ISAAC_VENV may be relative to the remote home
    remote "$(py_env) cd $(printf '%q' "$RDIR") &&
            export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src && $c" "$fwd"
    rc=$?
    say "exit $rc after $(( $(date +%s) - t )) s"
    [ ${#collects[@]} -eq 0 ] || fetch "${collects[@]}"
    exit $rc;;
  stop)
    # Ctrl-C (SIGINT) to this user's runs on the GPU box, then SIGKILL to what is left after 15 s
    remote "pat='tools/[s]im_workflow.py|tools/isaac/[l]ive_probe.py|python -m [p]ytest';   # [x]: no match on this shell's own line
            pids=\$(pgrep -u \$(id -u) -f \"\$pat\"); [ -n \"\$pids\" ] || { echo 'no runs'; exit 0; };
            ps -o pid=,etime=,args= -p \$pids | cut -c1-160; kill -INT \$pids;
            for i in \$(seq 15); do sleep 1; pids=\$(pgrep -u \$(id -u) -f \"\$pat\") || break; done;
            [ -z \"\$pids\" ] && echo 'ended' || { kill -KILL \$pids; echo 'killed:' \$pids; }";;
  *) sed -n 2,26p "$0"; exit 2;;
esac
