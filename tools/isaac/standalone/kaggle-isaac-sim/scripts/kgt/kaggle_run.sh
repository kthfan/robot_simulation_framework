#!/usr/bin/env bash
# kaggle_run.sh - run a command from a PUBLIC git repo on a Kaggle GPU kernel and bring the results back.
#
# Subcommands:
#   check                       verify CLI, credentials, network and repo state (no GPU quota used)
#   run    --cmd "..." [opts]   push a kernel, wait for it, fetch and summarise the result
#   run    ... --detach         push only; use `wait` and `fetch` later (best for long tests)
#   env-build --name N --python X.Y --setup "..."   prebuild a Python/PyTorch environment on Kaggle (CPU, free)
#   wait  [--slug S]            poll a pushed kernel until it finishes (default: the most recent run)
#   fetch [--slug S]            download outputs of that kernel and print the summary
#   quota                       real weekly GPU/TPU balance from `kaggle quota`
#
# Exit codes of run / fetch:
#   0 tests passed          1 tests failed            2 usage or preflight problem
#   3 infrastructure issue  4 gave up waiting         5 test command hit its time limit
#
# Run it from the root of the git repository under test.
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SELF_DIR/runner_template.py"
STATE_DIR="${KGT_STATE_DIR:-.kaggle-gpu-test}"
RUNS_DIR="$STATE_DIR/runs"      # one state file per kernel slug, so parallel runs never overwrite each other
LAST_FILE="$STATE_DIR/last_slug"
STATE_FILE=""
LEDGER="$STATE_DIR/usage.tsv"
MAX_TIMEOUT=32400          # 9 h, the conservative end of the session limits we found documented
WAIT_BUFFER="${KGT_WAIT_BUFFER:-1200}"   # extra seconds allowed for queueing and kernel start-up

say()  { printf '[kgt] %s\n' "$*" >&2; }
die()  { local code="$1"; shift; printf '[kgt] ERROR: %s\n' "$*" >&2; exit "$code"; }

usage() {
  sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

run options:
  --cmd "CMD"        test command, executed with bash at the repo root on Kaggle (required)
  --setup "CMD"      optional setup command run first, e.g. "pip install -e .[test]"
  --timeout T        budget for clone+setup+tests: seconds, or 90m / 2h (default 30m, max 9h)
  --cpu              run on a CPU kernel (does not use GPU quota)
  --gpu NAME         accelerator for `kaggle kernels push --accelerator`, e.g. NvidiaTeslaT4 (default if omitted)
  --python X.Y       run setup and tests inside a fresh venv of that Python version (installed with uv)
  --env NAME         use a prebuilt environment made by `env-build --name NAME` (replaces --python)
  --ref REF         branch, tag or full commit SHA (default: current HEAD commit)
  --repo URL         public https clone URL (default: derived from the origin remote)
  --collect GLOB     file glob (relative to the repo) to bring back; repeatable
  --slug NAME        kernel slug (default: <repo>-gpu-test); use different slugs to run in parallel
  --tail N           log lines to print in the summary (default 40)
  --poll S           polling interval in seconds (default 30)
  --detach           push and return immediately

environment:
  KAGGLE_API_TOKEN (KGAT_...) or KAGGLE_USERNAME + KAGGLE_KEY, or ~/.kaggle/kaggle.json
  KAGGLE_USERNAME / KGT_USER   Kaggle username when it cannot be read from the config
  KGT_ACCELERATOR              default for --gpu
EOF
}

# ---------- helpers ----------------------------------------------------------------------------

to_seconds() {
  local v="$1"
  case "$v" in
    *h) echo $(( ${v%h} * 3600 ));;
    *m) echo $(( ${v%m} * 60 ));;
    *s) echo "${v%s}";;
    *)  echo "$v";;
  esac
}

normalize_repo() {
  local u="${1%.git}"
  if [[ "$u" =~ ^git@([^:]+):(.+)$ ]]; then
    u="https://${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
  elif [[ "$u" =~ ^ssh://git@([^/]+)/(.+)$ ]]; then
    u="https://${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
  fi
  # drop credentials such as https://user:token@host/...
  printf '%s' "$u" | sed -E 's#^(https?://)[^/@]+@#\1#'
}

is_local_url() { [[ "$1" =~ ^https?://(localhost|127\.|\[::1\]|0\.0\.0\.0) ]]; }

slugify() {
  printf '%s' "$1" | tr 'A-Z' 'a-z' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//' | cut -c1-30
}

resolve_user() {
  if [[ -n "${KGT_USER:-}" ]]; then printf '%s' "$KGT_USER"; return; fi
  if [[ -n "${KAGGLE_USERNAME:-}" ]]; then printf '%s' "$KAGGLE_USERNAME"; return; fi
  local f="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" u=""
  if [[ -f "$f" ]]; then
    u="$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("username",""))' "$f" 2>/dev/null)"
  fi
  if [[ -z "$u" ]]; then
    u="$(kaggle config view 2>/dev/null | sed -n 's/^- username: *//p' | head -n1)"
  fi
  printf '%s' "$u"
}

have_credentials() {
  [[ -n "${KAGGLE_API_TOKEN:-}" ]] && return 0
  [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]] && return 0
  [[ -f "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" ]] && return 0
  [[ -f "$HOME/.kaggle/access_token" ]] && return 0
  return 1
}

need_kaggle() {
  command -v kaggle >/dev/null 2>&1 || die 2 "kaggle CLI not found. Install it: pip install kaggle --break-system-packages"
  have_credentials || die 2 "no Kaggle credentials. Set KAGGLE_API_TOKEN (kaggle.com/settings -> API) or provide ~/.kaggle/kaggle.json"
}

# prints one of: queued running complete error cancel unknown
kernel_status() {
  local raw s
  raw="$(kaggle kernels status "$1" 2>&1)" || { echo unknown; return; }
  s="$(printf '%s' "$raw" | tr 'A-Z' 'a-z')"
  case "$s" in
    *complete*)         echo complete;;
    *cancel*)           echo cancel;;
    *error*|*fail*)     echo error;;
    *running*)          echo running;;
    *queued*|*new*)     echo queued;;
    *)                  echo unknown;;
  esac
}

state_get() {  # state_get KEY
  python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$STATE_FILE" "$1"
}

# pick_state [--slug S]: choose which run wait/fetch act on (default: the most recent run)
pick_state() {
  local s=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --slug) s="${2:-}"; shift 2;;
      *) die 2 "unknown option: $1";;
    esac
  done
  if [[ -z "$s" ]]; then
    [[ -n "$STATE_FILE" ]] && return 0       # already chosen by `run`
    [[ -f "$LAST_FILE" ]] && s="$(cat "$LAST_FILE")"
  fi
  [[ -n "$s" ]] && STATE_FILE="$RUNS_DIR/$(slugify "$s").json"
  return 0
}

need_state() { [[ -n "$STATE_FILE" && -f "$STATE_FILE" ]] || die 2 "no matching run found. Start one with: kaggle_run.sh run ... (state dir: $STATE_DIR)"; }

# ---------- check ------------------------------------------------------------------------------

cmd_check() {
  local ok=0
  echo "== Kaggle GPU test preflight =="
  if command -v kaggle >/dev/null 2>&1; then echo "[ok]   kaggle CLI: $(kaggle --version 2>&1 | head -n1)"; else echo "[FAIL] kaggle CLI missing -> pip install kaggle --break-system-packages"; ok=1; fi
  if have_credentials; then echo "[ok]   credentials present"; else echo "[FAIL] no credentials -> export KAGGLE_API_TOKEN=KGAT_... (kaggle.com/settings -> API)"; ok=1; fi
  if [[ $ok -eq 0 ]]; then
    local out
    if out="$(kaggle kernels list --mine --page-size 1 2>&1)"; then
      echo "[ok]   Kaggle API reachable and token accepted"
    else
      echo "[FAIL] Kaggle API call failed: $(printf '%s' "$out" | head -n3 | tr '\n' ' ')"
      echo "       In Claude Code cloud, use a Custom network environment that allows kaggle.com"
      ok=1
    fi
    local u; u="$(resolve_user)"
    if [[ -n "$u" ]]; then echo "[ok]   Kaggle username: $u"; else echo "[FAIL] cannot determine Kaggle username -> export KAGGLE_USERNAME=<name>"; ok=1; fi
  fi
  if git rev-parse --git-dir >/dev/null 2>&1; then
    local remote repo sha
    remote="$(git remote get-url origin 2>/dev/null || true)"
    repo="$(normalize_repo "$remote")"
    sha="$(git rev-parse HEAD 2>/dev/null || true)"
    if [[ -z "$remote" ]]; then echo "[warn] no origin remote -> pass --repo https://github.com/<owner>/<repo> to run"
    elif is_local_url "$repo"; then echo "[warn] origin is a local proxy URL ($repo) -> pass --repo https://github.com/<owner>/<repo> to run"
    else echo "[ok]   repo URL: $repo"; fi
    echo "[info] HEAD commit: ${sha:-none}"
    if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then echo "[warn] uncommitted changes exist; Kaggle only sees what is pushed to the remote"; fi
  else
    echo "[warn] not inside a git repository; run this from the repo root"
  fi
  if [[ $ok -eq 0 ]]; then echo "RESULT: ready"; else echo "RESULT: not ready"; fi
  return $ok
}

# ---------- run --------------------------------------------------------------------------------

cmd_run() {
  local CMD="" SETUP="" TIMEOUT="1800" GPU=true REF="" REPO="" SLUG="" TAIL=40 POLL=30 DETACH=false
  local ACCEL="${KGT_ACCELERATOR:-}" PYVER="" ENVNAME="" MODE="test" BUILDNAME=""
  local -a COLLECT=()
  if [[ "${1:-}" == "--build-env-mode" ]]; then MODE="build_env"; GPU=false; shift; fi
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --name)    BUILDNAME="${2:-}"; shift 2;;
      --env)     ENVNAME="${2:-}"; shift 2;;
      --cmd)     CMD="${2:-}"; shift 2;;
      --setup)   SETUP="${2:-}"; shift 2;;
      --timeout) TIMEOUT="$(to_seconds "${2:-}")"; shift 2;;
      --cpu)     GPU=false; shift;;
      --gpu)     ACCEL="${2:-}"; shift 2;;
      --python)  PYVER="${2:-}"; shift 2;;
      --ref)     REF="${2:-}"; shift 2;;
      --repo)    REPO="${2:-}"; shift 2;;
      --collect) COLLECT+=("${2:-}"); shift 2;;
      --slug)    SLUG="${2:-}"; shift 2;;
      --tail)    TAIL="${2:-}"; shift 2;;
      --poll)    POLL="${2:-}"; shift 2;;
      --detach)  DETACH=true; shift;;
      -h|--help) usage; return 0;;
      *) die 2 "unknown option for run: $1";;
    esac
  done
  if [[ "$MODE" == build_env ]]; then
    [[ -n "$BUILDNAME" ]] || die 2 "env-build needs --name NAME"
    [[ -n "$PYVER" ]] || die 2 "env-build needs --python X.Y (the interpreter is part of the environment)"
    [[ -z "$ENVNAME" ]] || die 2 "--env makes no sense with env-build"
    BUILDNAME="$(slugify "$BUILDNAME")"; [[ -n "$BUILDNAME" ]] || die 2 "--name must contain letters or digits"
    CMD="(build environment $BUILDNAME)"; REPO="-"; REF="-"
    [[ "$TIMEOUT" == 1800 ]] && TIMEOUT=2400
    [[ -n "$SLUG" ]] || SLUG="kgt-env-$BUILDNAME"
  else
    [[ -n "$CMD" ]] || die 2 "--cmd is required"
    [[ -z "$ENVNAME" || -z "$PYVER" ]] || die 2 "--env already fixes the Python version; drop --python"
    [[ -z "$ENVNAME" ]] || { ENVNAME="$(slugify "$ENVNAME")"; [[ -n "$ENVNAME" ]] || die 2 "--env must contain letters or digits"; }
  fi
  [[ -z "$PYVER" || "$PYVER" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || die 2 "--python must look like 3.11 or 3.11.9"
  [[ -z "$ACCEL" || "$ACCEL" =~ ^[A-Za-z0-9_]+$ ]] || die 2 "--gpu must be an accelerator name such as NvidiaTeslaT4"
  if [[ "$GPU" == false && -n "$ACCEL" ]]; then say "note: --cpu given, ignoring accelerator $ACCEL"; ACCEL=""; fi
  [[ "$TIMEOUT" =~ ^[0-9]+$ ]] || die 2 "--timeout must be seconds or use an m/h suffix"
  (( TIMEOUT >= 60 )) || die 2 "--timeout must be at least 60 seconds"
  (( TIMEOUT <= MAX_TIMEOUT )) || die 2 "--timeout $TIMEOUT s exceeds the 9 h cap; split the tests into several runs"
  need_kaggle
  [[ -f "$TEMPLATE" ]] || die 2 "runner template missing: $TEMPLATE"

  # --- repo and ref (not needed to build an environment) ---
  if [[ "$MODE" == build_env ]]; then
    :
  elif [[ -z "$REPO" ]]; then
    git rev-parse --git-dir >/dev/null 2>&1 || die 2 "not inside a git repo; run from the repo root or pass --repo"
    REPO="$(normalize_repo "$(git remote get-url origin 2>/dev/null || true)")"
  else
    REPO="$(normalize_repo "$REPO")"
  fi
  if [[ "$MODE" != build_env ]]; then
    [[ -n "$REPO" ]] || die 2 "cannot determine the repo URL; pass --repo https://github.com/<owner>/<repo>"
    if is_local_url "$REPO"; then
      die 2 "origin points at a local proxy ($REPO) which Kaggle cannot reach; pass --repo https://github.com/<owner>/<repo>"
    fi
    [[ "$REPO" =~ ^https:// ]] || die 2 "repo URL must be a public https URL, got: $REPO"
  fi
  if [[ "$MODE" == build_env ]]; then
    :
  elif [[ -z "$REF" ]]; then
    REF="$(git rev-parse HEAD 2>/dev/null)" || die 2 "cannot read HEAD; pass --ref"
    if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
      say "warning: uncommitted changes are NOT part of this test (Kaggle clones the pushed commit $REF)"
    fi
    if refs="$(git ls-remote "$REPO" 2>/dev/null)" && [[ -n "$refs" ]]; then
      grep -q "^$REF" <<<"$refs" || say "warning: HEAD $REF is not the tip of any remote branch; if it was never pushed, the clone on Kaggle will fail. Push first."
    fi
  fi

  # --- identity, slug ---
  local USER_NAME; USER_NAME="$(resolve_user)"
  [[ -n "$USER_NAME" ]] || die 2 "cannot determine the Kaggle username; export KAGGLE_USERNAME=<name>"
  if [[ -z "$SLUG" ]]; then
    local base; base="$(slugify "$(basename "$REPO")")"
    SLUG="${base:-project}-gpu-test"
  fi
  SLUG="$(slugify "$SLUG")"
  local KERNEL_ID="$USER_NAME/$SLUG"

  # --- render runner and metadata ---
  local STAMP; STAMP="$(date +%Y%m%d-%H%M%S)"
  local PUSH_DIR="$STATE_DIR/push/$SLUG"
  rm -rf "$PUSH_DIR"; mkdir -p "$PUSH_DIR"
  local COLLECT_TXT=""
  (( ${#COLLECT[@]} )) && COLLECT_TXT="$(printf '%s\n' "${COLLECT[@]}")"
  python3 -I - "$TEMPLATE" "$PUSH_DIR" "$REPO" "$REF" "$SETUP" "$CMD" "$TIMEOUT" "$GPU" "$COLLECT_TXT" "$KERNEL_ID" "$SLUG" "$PYVER" "$MODE" "$ENVNAME" "$BUILDNAME" "$USER_NAME" <<'PY' || die 2 "failed to render the runner"
import base64, json, sys
tpl, out, repo, ref, setup, cmd, timeout, gpu, collect, kid, slug, pyver, mode, envname, buildname, user = sys.argv[1:17]
cfg = {"repo": repo, "ref": ref, "setup": setup, "cmd": cmd, "timeout": int(timeout), "python": pyver,
       "mode": mode, "env_name": envname, "env_name_built": buildname,
       "gpu": gpu == "true", "collect": [c for c in collect.split("\n") if c.strip()]}
src = open(tpl).read().replace("@@CONFIG_B64@@", base64.b64encode(json.dumps(cfg).encode()).decode())
open(out + "/runner.py", "w").write(src)
meta = {"id": kid, "title": slug.replace("-", " "), "code_file": "runner.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu == "true",
        "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ([user + "/kgt-env-" + envname] if envname else []), "model_sources": []}
json.dump(meta, open(out + "/kernel-metadata.json", "w"), indent=2)
PY

  # --- push ---
  local -a push=(kernels push -p "$PUSH_DIR")
  [[ -n "$ACCEL" ]] && push+=(--accelerator "$ACCEL")
  say "pushing $KERNEL_ID (gpu=$GPU${ACCEL:+ accel=$ACCEL}${PYVER:+ python=$PYVER}, ref=$REF, timeout=${TIMEOUT}s)"
  local out rc
  out="$(kaggle "${push[@]}" 2>&1)"; rc=$?
  printf '%s\n' "$out" >&2
  local pushed_ok=false
  grep -qi 'successfully pushed' <<<"$out" && pushed_ok=true
  if [[ $rc -ne 0 ]] || ! $pushed_ok; then
    die 3 "kaggle kernels push failed (see output above). Common causes: account not phone-verified, bad token, kaggle.com blocked by the sandbox network policy"
  fi

  local STARTED; STARTED="$(date +%s)"
  mkdir -p "$RUNS_DIR"; STATE_FILE="$RUNS_DIR/$SLUG.json"; printf '%s' "$SLUG" > "$LAST_FILE"
  python3 -I - "$STATE_FILE" "$KERNEL_ID" "$SLUG" "$STARTED" "$TIMEOUT" "$GPU" "$REF" "$CMD" "$TAIL" "$POLL" "$STAMP" "$ACCEL" "$MODE" <<'PY'
import json, sys
f, kid, slug, started, timeout, gpu, ref, cmd, tail, poll, stamp, accel, mode = sys.argv[1:14]
json.dump({"kernel_id": kid, "slug": slug, "started": int(started), "timeout": int(timeout),
           "gpu": gpu == "true", "ref": ref, "cmd": cmd, "tail": int(tail), "poll": int(poll),
           "stamp": stamp, "accel": accel, "mode": mode}, open(f, "w"), indent=2)
PY
  say "pushed. https://www.kaggle.com/code/$KERNEL_ID"
  if $DETACH; then
    say "detached. Next: kaggle_run.sh wait --slug $SLUG && kaggle_run.sh fetch --slug $SLUG"
    return 0
  fi
  cmd_wait || { local w=$?; [[ $w -eq 4 ]] && return 4; say "kernel did not finish cleanly; fetching whatever it produced"; }
  cmd_fetch
}

# ---------- wait -------------------------------------------------------------------------------

cmd_wait() {
  pick_state "$@"; need_kaggle; need_state
  local id started timeout poll
  id="$(state_get kernel_id)"; started="$(state_get started)"; timeout="$(state_get timeout)"; poll="$(state_get poll)"
  local deadline=$(( started + timeout + WAIT_BUFFER ))
  local last="" fails=0 n=0 st
  while :; do
    st="$(kernel_status "$id")"
    if [[ "$st" == "unknown" ]]; then
      fails=$((fails + 1))
      (( fails >= 6 )) && die 3 "cannot read the kernel status after $fails attempts (network or auth problem?)"
    else
      fails=0
    fi
    n=$((n + 1))
    if [[ "$st" != "$last" ]] || (( n % 10 == 0 )); then
      say "status=$st elapsed=$(( $(date +%s) - started ))s"
      last="$st"
    fi
    case "$st" in
      complete) return 0;;
      error|cancel) say "kernel ended with status $st"; return 3;;
    esac
    if (( $(date +%s) >= deadline )); then
      say "gave up waiting after $(( $(date +%s) - started ))s; the kernel may still be running."
      say "Kaggle's API cannot cancel it: stop it in the web UI (https://www.kaggle.com/code/$id) if needed, or call 'wait' again."
      return 4
    fi
    sleep "$poll"
  done
}

# ---------- fetch ------------------------------------------------------------------------------

cmd_fetch() {
  pick_state "$@"; need_kaggle; need_state
  local id slug stamp tail gpu started accel mode
  id="$(state_get kernel_id)"; slug="$(state_get slug)"; stamp="$(state_get stamp)"
  tail="$(state_get tail)"; gpu="$(state_get gpu)"; started="$(state_get started)"; accel="$(state_get accel)"
  mode="$(state_get mode)"
  local dir="$STATE_DIR/out/$slug-$stamp"
  mkdir -p "$dir"
  say "downloading outputs of $id into $dir"
  local -a outargs=(kernels output "$id" -p "$dir")
  # an environment kernel's output is a multi-GB tar that only other kernels need: never pull it here
  [[ "$mode" == build_env ]] && outargs+=(--file-pattern 'result\.json|test_output\.log|env\.json|env-freeze\.txt')
  if ! kaggle "${outargs[@]}" >/dev/null 2>&1; then
    say "warning: output download reported an error; summarising what is present"
  fi
  python3 -I - "$dir" "$tail" "$LEDGER" "$slug" "$gpu" "$accel" <<'PY'
import json, os, sys, time
d, tail, ledger, slug, gpu, accel = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
rp, lp = os.path.join(d, "result.json"), os.path.join(d, "test_output.log")

def show_tail(path):
    if os.path.exists(path):
        lines = open(path, errors="replace").read().splitlines()
        print("--- last %d of %d log lines (full log: %s) ---" % (min(tail, len(lines)), len(lines), path))
        print("\n".join(lines[-tail:]))
    else:
        print("(no test_output.log in the download)")

if not os.path.exists(rp):
    print("RESULT: infrastructure problem - the runner produced no result.json")
    for f in sorted(os.listdir(d)):
        if f.endswith(".log"):
            show_tail(os.path.join(d, f)); break
    sys.exit(3)

r = json.load(open(rp))
print("RESULT: %s" % r.get("status"))
print("repo:     %s @ %s" % (r.get("repo"), r.get("commit") or r.get("ref")))
if r.get("env_name"):
    print("env:      prebuilt '%s' restored from a previous kernel" % r["env_name"])
if r.get("timings"):
    print("timings:  " + "  ".join("%s=%ss" % kv for kv in r["timings"].items()))
print("gpu:      %s" % (r.get("gpu") or "none detected"))
if accel:
    print("requested accelerator: %s  <- compare with the gpu line; Kaggle can substitute silently" % accel)
print("torch:    %s" % (r.get("torch") or "n/a"))
print("python:   %s" % (r.get("python") or "n/a"))
print("command:  %s" % r.get("cmd"))
print("exit:     %s   timed_out=%s   duration=%ss (budget %ss)" % (
    r.get("exit_code"), r.get("timed_out"), r.get("duration_s"), r.get("timeout_s")))
if r.get("artifacts"):
    print("artifacts (%d): %s" % (len(r["artifacts"]), ", ".join(r["artifacts"][:10])))
if r.get("error"):
    print(r["error"])
show_tail(lp)

if r.get("status") == "env_built":
    print("VERDICT: environment built (%.2f GB), python %s, torch %s. Use it with: run --env %s ..." % (
        (r.get("env_bytes") or 0) / 1e9, r.get("python"), r.get("torch") or "n/a", slug.replace("kgt-env-", "", 1)))
    sys.exit(0)

if gpu == "True" and r.get("duration_s"):
    os.makedirs(os.path.dirname(ledger) or ".", exist_ok=True)
    with open(ledger, "a") as fh:
        fh.write("%d\t%s\t%s\n" % (time.time(), r["duration_s"], slug))

st = r.get("status")
if st in ("runner_error", "clone_failed", "setup_failed", "started"):
    print("VERDICT: infrastructure/setup failure (%s), the tests did not run to completion" % st); sys.exit(3)
if r.get("gpu_requested") and not r.get("gpu_available"):
    print("VERDICT: a GPU was requested but none was allocated, so this run proves nothing about GPU code"); sys.exit(3)
if r.get("timed_out"):
    print("VERDICT: the test command hit its time limit"); sys.exit(5)
if r.get("exit_code") == 0:
    print("VERDICT: tests passed"); sys.exit(0)
print("VERDICT: tests failed (exit code %s)" % r.get("exit_code")); sys.exit(1)
PY
}

# ---------- quota ------------------------------------------------------------------------------

cmd_quota() {
  # Kaggle CLI >= 2.x reports the real weekly balance; the local ledger is only a fallback for older CLIs.
  if command -v kaggle >/dev/null 2>&1 && have_credentials && kaggle quota 2>/dev/null; then
    return 0
  fi
  [[ -f "$LEDGER" ]] || { echo "'kaggle quota' unavailable and no local GPU runs recorded yet"; return 0; }
  python3 -I - "$LEDGER" <<'PY'
import sys, time
cut = time.time() - 7 * 86400
tot, n = 0.0, 0
for line in open(sys.argv[1]):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2 and float(p[0]) >= cut:
        tot += float(p[1]); n += 1
print("GPU runs in the last 7 days: %d, about %.2f h of the ~30 h weekly quota" % (n, tot / 3600))
print("(lower bound: it counts test time recorded by this tool, not Kaggle's own start-up time or runs made elsewhere)")
PY
}

# ---------- dispatch ---------------------------------------------------------------------------

sub="${1:-}"; [[ $# -gt 0 ]] && shift
case "$sub" in
  check) cmd_check;;
  run)   cmd_run "$@";;
  env-build) cmd_run --build-env-mode "$@";;
  wait)  cmd_wait "$@";;
  fetch) cmd_fetch "$@";;
  quota) cmd_quota;;
  -h|--help|help|"") usage;;
  *) usage; exit 2;;
esac