#!/usr/bin/env python3
"""Can JAX use the GPU on NSG? — the gate for running neurojax on Expanse.

The PyTorch tool env is CUDA 11.7, but modern JAX ships CUDA-12 pip wheels that bundle their own
CUDA and only need a recent NVIDIA driver. This probe records the driver version, installs
`jax[cuda12]` at runtime (egress confirmed), and checks whether jax.devices() sees the V100 —
running a tiny on-device matmul to prove real compute. If the driver is too old for CUDA 12, it
reports that cleanly (a real finding: JAX-on-NSG would then need an Apptainer image).
"""
import json
import re
import subprocess
import sys
import time

RESULT = {"schema": "nsg-agent-kit/jax/v1", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}


def sh(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def main():
    # 1) driver / CUDA from nvidia-smi
    try:
        smi = sh(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"])
        line = smi.stdout.strip().splitlines()[0]
        RESULT["driver_version"] = line.split(",")[0].strip()
        RESULT["gpu_name"] = line.split(",")[1].strip()
    except Exception as e:
        RESULT["smi_error"] = repr(e)

    # 2) install jax with CUDA 12 wheels
    t0 = time.time()
    p = sh([sys.executable, "-m", "pip", "install", "--quiet", "-U", "jax[cuda12]"], timeout=900)
    RESULT["pip_ok"] = p.returncode == 0
    RESULT["pip_seconds"] = round(time.time() - t0, 1)
    if p.returncode != 0:
        RESULT["pip_stderr_tail"] = p.stderr[-800:]
        return _write()

    # 3) import + device check + tiny compute
    try:
        import jax
        import jax.numpy as jnp
        RESULT["jax_version"] = jax.__version__
        devs = jax.devices()
        RESULT["jax_devices"] = [str(d) for d in devs]
        RESULT["jax_default_backend"] = jax.default_backend()
        RESULT["gpu_visible"] = any(d.platform == "gpu" for d in devs)

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (1024, 1024))
        y = (x @ x.T)
        RESULT["matmul_trace"] = float(jnp.trace(y))
        RESULT["compute_device"] = str(y.devices()) if hasattr(y, "devices") else str(devs[0])
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
