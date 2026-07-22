#!/usr/bin/env python3
"""JAX-on-NSG, take 2 — the tool's system venv is READ-ONLY.

v1 revealed: `pip install jax[cuda12]` failed with OSError [Errno 30] Read-only file system on
/usr/local/python/venv/.../site-packages. So runtime pip cannot write to the system venv. The fix
is to install into a writable target dir in the job cwd and prepend it to sys.path.

This probe:
  1. reports which key packages are ALREADY importable in the image (to explain why `mne` "installed"
     in 11 s — it was likely pre-present),
  2. `pip install --target=./pylibs jax[cuda12]` (writable), adds it to sys.path,
  3. checks jax.devices() / gpu_visible and runs an on-device matmul.
"""
import importlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
import time

RESULT = {"schema": "nsg-agent-kit/jax/v2", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
TARGET = os.path.abspath("./pylibs")


def sh(cmd, timeout=900):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def present(mod):
    try:
        v = md.version(mod)
    except Exception:
        v = None
    try:
        importlib.import_module(mod)
        imp = True
    except Exception:
        imp = False
    return {"version": v, "importable": imp}


def main():
    # driver
    try:
        smi = sh(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"])
        line = smi.stdout.strip().splitlines()[0]
        RESULT["driver_version"] = line.split(",")[0].strip()
        RESULT["gpu_name"] = line.split(",")[1].strip()
    except Exception as e:
        RESULT["smi_error"] = repr(e)

    # what's already in the image? (explains the mne "quick win")
    RESULT["preinstalled"] = {p: present(p) for p in
                              ("mne", "torch", "jax", "peft", "transformers", "braindecode", "numpy")}

    # writable-target install
    t0 = time.time()
    p = sh([sys.executable, "-m", "pip", "install", "--target", TARGET, "-U", "jax[cuda12]"])
    RESULT["pip_ok"] = p.returncode == 0
    RESULT["pip_seconds"] = round(time.time() - t0, 1)
    if p.returncode != 0:
        RESULT["pip_stderr_tail"] = p.stderr[-800:]
        return _write()

    sys.path.insert(0, TARGET)
    try:
        import jax
        import jax.numpy as jnp
        RESULT["jax_version"] = jax.__version__
        RESULT["jax_devices"] = [str(d) for d in jax.devices()]
        RESULT["jax_default_backend"] = jax.default_backend()
        RESULT["gpu_visible"] = any(d.platform == "gpu" for d in jax.devices())
        x = jax.random.normal(jax.random.PRNGKey(0), (1024, 1024))
        RESULT["matmul_trace"] = float(jnp.trace(x @ x.T))
        RESULT["jax_ok"] = True
    except Exception as e:
        RESULT["jax_ok"] = False
        RESULT["jax_error"] = repr(e)
    _write()


def _write():
    RESULT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open("metrics.json", "w") as f:
        json.dump(RESULT, f, indent=2)
    print(json.dumps(RESULT, indent=2))


if __name__ == "__main__":
    main()
