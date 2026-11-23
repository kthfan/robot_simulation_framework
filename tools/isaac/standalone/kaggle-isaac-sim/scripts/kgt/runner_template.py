#!/usr/bin/env python3
"""Runs ON Kaggle: clone a public repo at a given ref, run setup + test command, record results.

This file is a template. kaggle_run.sh renders it into the kernel's code file by filling in the
configuration blob below, then pushes it with `kaggle kernels push`.

Contract with the caller (kaggle_run.sh fetch):
  * /kaggle/working/result.json     machine-readable outcome (always written, even on failure)
  * /kaggle/working/test_output.log everything the setup and test commands printed
  * /kaggle/working/artifacts/...   files matched by the "collect" globs (optional)
  * The script exits 0 whenever result.json was written for a test outcome, so a kernel status of
    ERROR means "the runner itself broke", never "the tests failed". Test failures are reported
    through result.json instead.
"""
import base64
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

CONFIG = json.loads(base64.b64decode("@@CONFIG_B64@@").decode())

# Overridable so the runner can be exercised locally without a Kaggle machine.
OUT = os.environ.get("KGT_OUT", "/kaggle/working")
WORK = os.environ.get("KGT_WORK", "/tmp/kgt-repo")

# A venv embeds absolute paths in its scripts, so it is only relocatable if it lives at the same path when it
# is built and when it is restored. Both locations are fixed constants for that reason.
VENV_PATH = os.environ.get("KGT_VENV", "/tmp/kgt-venv")
UV_PY_DIR = os.environ.get("KGT_UV_PY", "/tmp/kgt-uv-python")
ENV_TAR = "env.tar"

MAX_LOG_BYTES = 50 * 1024 * 1024        # stop mirroring into the log file beyond this
MAX_ARTIFACT_BYTES = 200 * 1024 * 1024  # total size of collected artifacts


def quick(cmd, timeout=120):
    """Run a short diagnostic command and return its stripped stdout ('' on any failure)."""
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return res.stdout.strip()
    except Exception:
        return ""


VENV = None  # set when --python is used; prepended to PATH for setup and test commands


def venv_env():
    """Environment variables that make a command use the venv, or {} when none is active."""
    if not VENV:
        return {}
    return {"VIRTUAL_ENV": VENV, "PATH": VENV + "/bin:" + os.environ.get("PATH", "")}


def run_streamed(cmd, cwd, log, deadline):
    """Run `cmd` in bash, mirror output to stdout and `log`, kill the process group at `deadline`.

    Returns (returncode, timed_out). Kaggle's API cannot cancel a running kernel, so enforcing the
    time budget from inside the kernel is the only reliable way to stop wasting GPU quota.
    """
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", PYTHONUNBUFFERED="1", **venv_env())
    env.pop("PYTHONHOME", None)
    if VENV:
        # Kaggle sets UV_SYSTEM_PYTHON, which makes `uv pip install` ignore the venv and modify the system
        # interpreter instead. Inside our venv the installer must target the venv.
        env.pop("UV_SYSTEM_PYTHON", None)
    proc = subprocess.Popen(
        ["bash", "-c", "set -o pipefail; " + cmd],
        cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", bufsize=1, start_new_session=True,
    )
    timed_out = threading.Event()

    def kill():
        timed_out.set()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(max(deadline - time.time(), 1), kill)
    timer.start()
    written = 0
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if written < MAX_LOG_BYTES:
                log.write(line)
                written += len(line)
        proc.wait()
    finally:
        timer.cancel()
    return proc.returncode, timed_out.is_set()


def create_venv(python, log, deadline):
    """Create the fixed-path venv with a uv-managed interpreter. Returns (rc, timed_out).

    --managed-python matters: CPU and GPU kernels ship different system Pythons (3.12 vs 3.13 when tested), and
    an environment that borrowed the system interpreter would not be self-contained or portable between them.
    """
    mk = ("(command -v uv >/dev/null 2>&1 || pip install -q uv) && "
          "uv venv --seed --managed-python --python '%s' '%s'" % (python, VENV_PATH))
    line = "\n[kgt] $ %s\n" % mk
    log.write(line)
    sys.stdout.write(line)
    return run_streamed(mk, "/tmp", log, deadline)


def probe_env(cwd):
    """Python and torch versions as the test command will see them."""
    probe = ("python -c \"import sys;print('python', sys.version.split()[0])\"; "
             "python -c \"import torch;print('torch', torch.__version__, 'cuda', torch.version.cuda, "
             "'available', torch.cuda.is_available())\" 2>/dev/null || true")
    res = subprocess.run(["bash", "-c", probe], cwd=cwd, capture_output=True, text=True,
                         env=dict(os.environ, **venv_env()))
    out = {}
    for ln in res.stdout.splitlines():
        if ln.startswith("python "):
            out["python"] = ln[7:]
        elif ln.startswith("torch "):
            out["torch"] = ln[6:]
    return out


def build_env(cfg, result, log, deadline, save, timings):
    """Build mode: create the venv, run setup, pack venv + interpreter into OUT/env.tar. No repo needed."""
    global VENV
    cwd = "/tmp/kgt-build"
    shutil.rmtree(cwd, ignore_errors=True)
    os.makedirs(cwd)
    t = time.time()
    rc, to = create_venv(cfg["python"], log, deadline)
    timings["venv"] = round(time.time() - t, 1)
    if rc != 0 or to:
        result.update(status="setup_failed", exit_code=None if to else rc, timed_out=to)
        save()
        return
    VENV = VENV_PATH
    if cfg["setup"]:
        line = "\n[kgt] $ %s\n" % cfg["setup"]
        log.write(line)
        sys.stdout.write(line)
        t = time.time()
        rc, to = run_streamed(cfg["setup"], cwd, log, deadline)
        timings["setup"] = round(time.time() - t, 1)
        if rc != 0 or to:
            result.update(status="setup_failed", exit_code=None if to else rc, timed_out=to)
            save()
            return
    seen = probe_env(cwd)
    # overwrite, not merge: a failed probe must not leave the pre-venv values behind
    result['python'] = seen.get('python', '')
    result['torch'] = seen.get('torch', '')
    freeze = subprocess.run(["bash", "-c", "pip freeze 2>/dev/null"], capture_output=True, text=True,
                            env=dict(os.environ, **venv_env()))
    with open(os.path.join(OUT, "env-freeze.txt"), "w") as fh:
        fh.write(freeze.stdout)
    tar_path = os.path.join(OUT, ENV_TAR)
    pack = "tar -cf '%s' -C / '%s' '%s'" % (tar_path, VENV_PATH.lstrip("/"), UV_PY_DIR.lstrip("/"))
    line = "\n[kgt] $ %s\n" % pack
    log.write(line)
    sys.stdout.write(line)
    t = time.time()
    rc, to = run_streamed(pack, cwd, log, deadline)
    timings["pack"] = round(time.time() - t, 1)
    if rc != 0 or to:
        result.update(status="setup_failed", exit_code=None if to else rc, timed_out=to)
        save()
        return
    size = os.path.getsize(tar_path)
    with open(os.path.join(OUT, "env.json"), "w") as fh:
        json.dump({"name": cfg.get("env_name_built"), "python": result.get("python"), "torch": result.get("torch"),
                   "setup": cfg["setup"], "venv_path": VENV_PATH, "uv_python_dir": UV_PY_DIR,
                   "bytes": size, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh, indent=2)
    result.update(status="env_built", exit_code=0, env_bytes=size)
    save()
    line = "\n[kgt] ENV BUILT %s (%.2f GB) python=%s torch=%s\n" % (
        ENV_TAR, size / 1e9, result.get("python"), result.get("torch"))
    log.write(line)
    sys.stdout.write(line)


def find_env_tar():
    """The env kernel's output is mounted somewhere below /kaggle/input; the exact layout has changed over
    time, so search instead of hard-coding it."""
    hits = sorted(glob.glob("/kaggle/input/**/" + ENV_TAR, recursive=True))
    return hits[0] if hits else None


def collect_artifacts(patterns):
    """Copy files matching the globs (relative to the repo) into OUT/artifacts."""
    copied, total = [], 0
    for pattern in patterns:
        for path in glob.glob(os.path.join(WORK, pattern), recursive=True):
            if not os.path.isfile(path):
                continue
            size = os.path.getsize(path)
            if total + size > MAX_ARTIFACT_BYTES:
                continue
            rel = os.path.relpath(path, WORK)
            dest = os.path.join(OUT, "artifacts", rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(path, dest)
            copied.append(rel)
            total += size
    return copied


def main():
    global VENV
    started = time.time()
    cfg = CONFIG
    deadline = started + cfg["timeout"]
    os.makedirs(OUT, exist_ok=True)
    os.environ["UV_PYTHON_INSTALL_DIR"] = UV_PY_DIR
    timings = {}

    result = {
        "schema": 2,
        "mode": cfg.get("mode", "test"),
        "env_name": cfg.get("env_name") or "",
        "timings": timings,
        "status": "started",
        "repo": cfg["repo"],
        "ref": cfg["ref"],
        "commit": None,
        "setup": cfg["setup"],
        "cmd": cfg["cmd"],
        "timeout_s": cfg["timeout"],
        "gpu_requested": cfg["gpu"],
        "gpu_available": False,
        "gpu": "",
        "torch": "",
        "python_requested": cfg.get("python") or "",
        "exit_code": None,
        "timed_out": False,
        "duration_s": None,
        "artifacts": [],
    }

    def save():
        result["duration_s"] = round(time.time() - started, 1)
        with open(os.path.join(OUT, "result.json"), "w") as fh:
            json.dump(result, fh, indent=2)

    save()  # an early result.json means a crash still leaves a trace
    log_path = os.path.join(OUT, "test_output.log")

    try:
        with open(log_path, "w", errors="replace") as log:
            # --- environment facts that explain most "works on my machine" surprises ---
            result["gpu"] = quick("nvidia-smi --query-gpu=name,memory.total,driver_version "
                                  "--format=csv,noheader").replace("\n", " | ")
            result["gpu_available"] = bool(result["gpu"])
            result["torch"] = quick(
                "python -c \"import torch;print(torch.__version__, 'cuda', torch.version.cuda, "
                "'available', torch.cuda.is_available())\" 2>/dev/null")
            result["python"] = quick("python --version 2>&1")
            banner = "[kgt] gpu=%r torch=%r python=%r\n" % (
                result["gpu"], result["torch"], result["python"])
            sys.stdout.write(banner)
            log.write(banner)
            if cfg["gpu"] and not result["gpu_available"]:
                warn = ("[kgt] WARNING: a GPU was requested but nvidia-smi found none. Is the Kaggle "
                        "account phone-verified and the GPU quota not exhausted?\n")
                sys.stdout.write(warn)
                log.write(warn)

            if cfg.get("mode") == "build_env":
                build_env(cfg, result, log, deadline, save, timings)
                return

            # --- fetch the exact ref (works for branches, tags and full commit SHAs) ---
            shutil.rmtree(WORK, ignore_errors=True)
            os.makedirs(WORK, exist_ok=True)
            clone = ("git init -q && git remote add origin '{repo}' && "
                     "git fetch -q --depth 1 origin '{ref}' && git checkout -q FETCH_HEAD"
                     ).format(repo=cfg["repo"], ref=cfg["ref"])
            rc, to = run_streamed(clone, WORK, log, deadline)
            if rc != 0 or to:
                result.update(status="clone_failed", timed_out=to)
                save()
                return
            result["commit"] = quick("git -C '%s' rev-parse HEAD" % WORK)

            if cfg.get("env_name"):
                # --- restore a prebuilt environment (mounted from another kernel's output) ---
                tar = find_env_tar()
                if not tar:
                    msg = ("[kgt] prebuilt environment %r not found under /kaggle/input. Was the env kernel "
                           "built, finished, and listed in kernel_sources?\n" % cfg["env_name"])
                    log.write(msg)
                    sys.stdout.write(msg)
                    result.update(status="setup_failed", error=msg)
                    save()
                    return
                line = "\n[kgt] restoring environment from %s\n" % tar
                log.write(line)
                sys.stdout.write(line)
                t = time.time()
                rc, to = run_streamed("tar -xf '%s' -C /" % tar, "/tmp", log, deadline)
                timings["env_restore"] = round(time.time() - t, 1)
                if rc != 0 or to:
                    result.update(status="setup_failed", exit_code=None if to else rc, timed_out=to)
                    save()
                    return
                VENV = VENV_PATH
            elif cfg.get("python"):
                # --- isolated interpreter of a chosen Python version (via uv) ---
                t = time.time()
                rc, to = create_venv(cfg["python"], log, deadline)
                timings["venv"] = round(time.time() - t, 1)
                if rc != 0 or to:
                    result.update(status="setup_failed", exit_code=None if to else rc, timed_out=to)
                    save()
                    return
                VENV = VENV_PATH

            # --- optional setup (with a prebuilt env this is just the project's own install) ---
            if cfg["setup"]:
                log.write("\n[kgt] $ %s\n" % cfg["setup"])
                sys.stdout.write("\n[kgt] $ %s\n" % cfg["setup"])
                t = time.time()
                rc, to = run_streamed(cfg["setup"], WORK, log, deadline)
                timings["setup"] = round(time.time() - t, 1)
                if rc != 0 or to:
                    result.update(status="setup_failed", exit_code=None if to else rc, timed_out=to)
                    save()
                    return

            # --- what the test command will really see (differs from the first banner after --python/setup) ---
            seen = probe_env(WORK)
            # overwrite, not merge: a failed probe must not leave the pre-venv values behind
            result['python'] = seen.get('python', '')
            result['torch'] = seen.get('torch', '')
            banner2 = "[kgt] effective env: python=%r torch=%r\n" % (result.get("python"), result.get("torch"))
            sys.stdout.write(banner2)
            log.write(banner2)

            # --- the actual test command ---
            log.write("\n[kgt] $ %s\n" % cfg["cmd"])
            sys.stdout.write("\n[kgt] $ %s\n" % cfg["cmd"])
            t = time.time()
            rc, to = run_streamed(cfg["cmd"], WORK, log, deadline)
            timings["cmd"] = round(time.time() - t, 1)
            result.update(status="finished", exit_code=None if to else rc, timed_out=to)
            result["artifacts"] = collect_artifacts(cfg["collect"])
            save()
            line = "\n[kgt] RESULT exit_code=%s timed_out=%s duration=%ss\n" % (
                result["exit_code"], result["timed_out"], result["duration_s"])
            sys.stdout.write(line)
    except Exception:
        result.update(status="runner_error", error=traceback.format_exc())
        save()
        sys.stdout.write(result["error"])
        sys.exit(1)  # make the kernel status ERROR so the caller knows the runner itself broke


if __name__ == "__main__":
    main()