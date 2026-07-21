#!/usr/bin/env python3
"""NSG/Expanse platform probe: cheapest real end-to-end GPU job.

A *frozen* random-init 1D-conv encoder + a trainable linear probe, on synthetic multichannel
EEG. Torch + stdlib only — NO peft/braindecode/transformers — so it isolates the platform
(env, GPU, filesystem, I/O) from any dependency or data question. Runtime is a couple of
minutes on one GPU.

It writes ``metrics.json`` to the working directory (which NSG returns to you). That file is the
contract the TDD test in ``probes/tests`` checks. Every field also answers a FINDINGS "[VERIFY]":

  python_version / torch_version  -> exact tool env
  cuda_available / gpu_name / gpu_count -> the GPU crux (§3b)
  network_egress                  -> the runtime-internet crux (§3a): can we pip at runtime?
  nemarpath / nemarpath_exists    -> is NEMAR data reachable from the PyTorch tool (§ nemar-data)
  probe_ok / probe_accuracy       -> did GPU compute actually run end to end
"""
import json
import os
import platform
import socket
import sys
import time
import urllib.request

RESULT = {"schema": "nsg-agent-kit/probe/v1", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}


def check_network_egress(timeout=5):
    """Return True iff the compute node can reach the public internet.

    This is the single most decision-relevant fact: if False, runtime `pip install` from PyPI
    will not work and deps must be vendored (see docs/dependencies.md).
    """
    for host, port in (("pypi.org", 443), ("8.8.8.8", 53)):
        try:
            socket.create_connection((host, port), timeout=timeout).close()
            return True
        except OSError:
            continue
    # HTTP fallback in case raw sockets are filtered but a proxy exists
    try:
        urllib.request.urlopen("https://pypi.org/simple/", timeout=timeout)
        return True
    except Exception:
        return False


def main():
    RESULT["python_version"] = sys.version.split()[0]
    RESULT["platform"] = platform.platform()
    RESULT["hostname"] = socket.gethostname()
    RESULT["nemarpath"] = os.environ.get("NEMARPATH")
    RESULT["nemarpath_exists"] = bool(
        RESULT["nemarpath"] and os.path.isdir(RESULT["nemarpath"])
    )
    RESULT["network_egress"] = check_network_egress()

    # --- torch / GPU ---
    try:
        import torch
        RESULT["torch_version"] = torch.__version__
        RESULT["cuda_available"] = torch.cuda.is_available()
        RESULT["gpu_count"] = torch.cuda.device_count()
        RESULT["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception as e:  # torch missing => wrong tool selected
        RESULT["torch_import_error"] = repr(e)
        RESULT["probe_ok"] = False
        _write()
        return

    # --- the actual compute: frozen encoder + linear probe on synthetic EEG ---
    try:
        import torch
        import torch.nn as nn

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        RESULT["device_used"] = dev
        g = torch.Generator().manual_seed(0)

        n, ch, t = 512, 22, 256  # trials, EEG channels, samples
        # two linearly-separable-ish classes in a random frozen feature space
        y = torch.randint(0, 2, (n,), generator=g)
        x = torch.randn(n, ch, t, generator=g)
        x += (y.view(-1, 1, 1) * 0.5)  # class-dependent offset -> above chance is expected

        class FrozenEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(ch, 16, 7, stride=2), nn.ReLU(),
                    nn.Conv1d(16, 16, 7, stride=2), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                )
                for p in self.parameters():
                    p.requires_grad_(False)  # FROZEN

            def forward(self, z):
                return self.net(z)

        enc = FrozenEncoder().to(dev).eval()
        with torch.no_grad():
            feats = enc(x.to(dev))
        probe = nn.Linear(feats.shape[1], 2).to(dev)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
        lossf = nn.CrossEntropyLoss()
        yd = y.to(dev)
        for _ in range(200):
            opt.zero_grad()
            loss = lossf(probe(feats), yd)
            loss.backward()
            opt.step()
        acc = (probe(feats).argmax(1) == yd).float().mean().item()

        RESULT["probe_accuracy"] = round(acc, 4)
        RESULT["final_loss"] = round(loss.item(), 4)
        RESULT["probe_ok"] = acc > 0.55  # must beat chance -> real gradient flow on device
    except Exception as e:
        RESULT["probe_error"] = repr(e)
        RESULT["probe_ok"] = False

    _write()


def _write():
    RESULT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open("metrics.json", "w") as f:
        json.dump(RESULT, f, indent=2)
    print(json.dumps(RESULT, indent=2))


if __name__ == "__main__":
    main()
