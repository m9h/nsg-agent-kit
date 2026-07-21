"""Red-green TDD contract for the NSG frozen probe.

RED  (before submission): probes/reve_frozen_probe/results/metrics.json does not exist -> fail.
GREEN (after you fetch NSG output into results/): the metrics.json validates -> pass.

This is the outer TDD loop for "does our pipeline run end-to-end on NSG": the test encodes the
contract the remote job must satisfy; you make it pass by actually running the job on Expanse and
bringing the artifact back, not by editing code.

    pytest probes/tests -q            # RED first, GREEN after a successful NSG run

Set PROBE_EXPECT_GPU=0 to allow a CPU-only run to pass (e.g. a local dry run), otherwise the probe
must report it ran on a CUDA device.
"""
import json
import os
import pathlib

import pytest

RESULTS = (
    pathlib.Path(__file__).resolve().parents[1] / "reve_frozen_probe" / "results" / "metrics.json"
)
EXPECT_GPU = os.environ.get("PROBE_EXPECT_GPU", "1") == "1"


def _load():
    if not RESULTS.exists():
        pytest.fail(
            f"RED: {RESULTS} not found. Submit reve_frozen_probe.zip to NSG "
            f"(tool 'PyTorch Python on Expanse'), then fetch outputs into results/. "
            f"See docs/submit-rest.md."
        )
    return json.loads(RESULTS.read_text())


def test_schema_and_completion():
    m = _load()
    assert m.get("schema") == "nsg-agent-kit/probe/v1"
    assert m.get("finished"), "job did not finish writing metrics.json"


def test_env_facts_recorded():
    m = _load()
    # these fields are the whole point — they close the FINDINGS must-verify list
    for k in ("python_version", "torch_version", "network_egress",
              "cuda_available", "gpu_count"):
        assert k in m, f"missing platform fact: {k}"


def test_compute_actually_ran():
    m = _load()
    assert m.get("probe_ok") is True, f"probe compute failed: {m}"
    assert m.get("probe_accuracy", 0) > 0.55, "probe did not beat chance -> no real gradient flow"


def test_gpu_used_when_expected():
    m = _load()
    if EXPECT_GPU:
        assert m.get("cuda_available") is True, "expected a GPU tool but CUDA was unavailable"
        assert m.get("device_used") == "cuda"
        assert m.get("gpu_name"), "no GPU name reported"


def test_report_platform_facts(capsys):
    """Not an assertion — prints the captured facts so a run is self-documenting."""
    m = _load()
    facts = {
        k: m.get(k)
        for k in (
            "python_version", "torch_version", "gpu_name", "gpu_count",
            "network_egress", "nemarpath", "nemarpath_exists",
            "probe_accuracy", "hostname",
        )
    }
    print("\nNSG platform facts:\n" + json.dumps(facts, indent=2))
