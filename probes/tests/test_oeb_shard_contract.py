"""Red-green contract for an OpenEEGBench shard run through upstream `open_eeg_bench` on NSG.

RED  : no shard_summary.json fetched yet -> fail with the submit instructions.
GREEN: the shard installed the pinned stack, ran upstream's Experiment.run() for every cell,
       every cell completed, and every classification cell beat chance.

    pytest probes/tests/test_oeb_shard_contract.py -q
    OEB_SHARD_DIR=sweep/oeb/smoke/fetched/arithmetic_zyma2019_s0/extracted pytest ...
"""
import glob
import json
import os
import pathlib

import pytest

DEFAULT = pathlib.Path(__file__).resolve().parents[2] / "sweep" / "oeb" / "smoke" / "fetched"
ROOT = pathlib.Path(os.environ.get("OEB_SHARD_DIR", str(DEFAULT)))
# chance level = 1/n_classes for the upstream default datasets (balanced accuracy)
N_CLASSES = {"braindecode/arithmetic_zyma2019": 2, "braindecode/bcic2a": 4, "braindecode/bcic2020-3": 5,
             "braindecode/physionet": 4, "braindecode/chbmit": 2, "braindecode/faced": 9,
             "braindecode/isruc-sleep": 5, "braindecode/mdd_mumtaz2016": 2}


def _summaries():
    found = glob.glob(str(ROOT / "**" / "shard_summary.json"), recursive=True)
    if not found:
        pytest.fail(
            f"RED: no shard_summary.json under {ROOT}. Run\n"
            f"  python sweep/oeb_plan.py plan --name smoke\n"
            f"  ./nsgr/nsgr.sh submit PYTORCH_PY_EXPANSE sweep/oeb/smoke/shards/arithmetic_zyma2019_s0.zip\n"
            f"then fetch output.tar.gz into sweep/oeb/smoke/fetched/<shard>/ and `oeb_plan.py merge`."
        )
    return [json.load(open(f)) for f in found]


def test_shard_installed_pinned_stack():
    for s in _summaries():
        assert s.get("pip_ok") is True, s.get("pip_err")
        assert s.get("torch", "").startswith("2.4.1"), s.get("torch")   # the Volta-safe pin (FINDINGS §6a)
        assert s.get("cuda") is True, "shard did not see the V100"


def test_shard_ran_every_cell_to_completion():
    for s in _summaries():
        assert "shard_error" not in s, s.get("trace")
        assert s.get("n_experiments", 0) > 0
        assert s.get("n_completed") == s.get("n_experiments"), [
            (r.get("dataset"), r.get("finetuning"), r.get("exception")) for r in s["rows"]
            if r.get("status") != "completed"]


def test_classification_cells_beat_chance():
    for s in _summaries():
        for r in s["rows"]:
            if r.get("dataset") in N_CLASSES and "test_balanced_accuracy" in r:
                chance = 1.0 / N_CLASSES[r["dataset"]]
                assert r["test_balanced_accuracy"] > chance, r
