#!/usr/bin/env python3
"""JAX-on-NSG, take 4 — install to a writable target, then import in a CLEAN subprocess.

Lessons baked in from v1-v3:
  - the tool's system venv is READ-ONLY  -> install with `pip install --target ./pylibs`
  - `--target` alone skips deps the system already has (numpy 1.24.3) -> add `--ignore-installed`
    so the target is self-contained with a modern numpy
  - you must import jax in a FRESH interpreter with PYTHONPATH=./pylibs; importing numpy/torch in
    the same process first pins numpy 1.24.3 in sys.modules and shadows the target (this is exactly
    why entry.sh sets PYTHONPATH then runs a new `python`).

Reports driver, what's pre-installed, install status, and jax.devices() from the clean subprocess.
"""
import importlib.metadata as md
import importlib.util as iu
import json
import os
import subprocess
import sys
import time

RESULT = {"schema": "nsg-agent-kit/jax/v4", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
TARGET = os.path.abspath("./pylibs")

JAX_CHECK = r"""
import json
try:
    import jax, jax.numpy as jnp
    import numpy
    devs = jax.devices()
    x = jax.random.normal(jax.random.PRNGKey(0), (1024, 1024))
    out = {"jax_ok": True, "jax_version": jax.__version__, "numpy_version": numpy.__version__,
           "jax_devices": [str(d) for d in devs], "jax_default_backend": jax.default_backend(),
           "gpu_visible": any(d.platform == "gpu" for d in devs),
           "matmul_trace": float(jnp.trace(x @ x.T))}
except Exception as e:
    out = {"jax_ok": False, "jax_error": repr(e)}
print("JAXJSON:" + json.dumps(out))
"""


def sh(cmd, timeout=900, env=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)


def main():
    try:
        smi = sh(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"])
        line = smi.stdout.strip().splitlines()[0]
        RESULT["driver_version"] = line.split(",")[0].strip()
        RESULT["gpu_name"] = line.split(",")[1].strip()
    except Exception as e:
        RESULT["smi_error"] = repr(e)

    # inspect WITHOUT importing (find_spec + metadata only, so we don't poison sys.modules)
    def present(mod):
        try:
            v = md.version(mod)
        except Exception:
            v = None
        return {"version": v, "installed": iu.find_spec(mod) is not None}
    RESULT["preinstalled"] = {p: present(p) for p in
                              ("mne", "torch", "jax", "peft", "transformers", "braindecode", "numpy")}

    t0 = time.time()
    p = sh([sys.executable, "-m", "pip", "install", "--target", TARGET,
            "--ignore-installed", "-U", "jax[cuda12]"])
    RESULT["pip_ok"] = p.returncode == 0
    RESULT["pip_seconds"] = round(time.time() - t0, 1)
    if p.returncode != 0:
        RESULT["pip_stderr_tail"] = p.stderr[-800:]
        return _write()

    # import jax in a FRESH interpreter with the target ahead on PYTHONPATH
    env = dict(os.environ, PYTHONPATH=TARGET)
    r = sh([sys.executable, "-c", JAX_CHECK], env=env, timeout=300)
    out = {}
    for ln in r.stdout.splitlines():
        if ln.startswith("JAXJSON:"):
            out = json.loads(ln[len("JAXJSON:"):])
    if not out:
        out = {"jax_ok": False, "jax_error": "no JAXJSON line", "stderr_tail": r.stderr[-500:]}
    RESULT.update(out)
    _write()


def _write():
    RESULT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open("metrics.json", "w") as f:
        json.dump(RESULT, f, indent=2)
    print(json.dumps(RESULT, indent=2))


if __name__ == "__main__":
    main()
