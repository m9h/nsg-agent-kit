"""Red-green TDD contract for M5: REVE + LoRA fine-tune cell on NSG.

RED  (before submission): sweep/results/lora_<dataset>/metrics.json does not exist -> fail.
GREEN (after fetching NSG output): the metrics.json validates the LoRA cell ran end-to-end,
beat chance, and reports a trainable-parameter count consistent with LoRA (not full fine-tune).

    pytest probes/tests/test_lora_cell_contract.py -q

Set LORA_RESULT_DIR to point at a specific fetched result dir; defaults to sweep/results/lora_bnci2014_001.
"""
import json
import os
import pathlib

import pytest

DEFAULT_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "sweep" / "results" / "lora_bnci2014_001"
)
RESULTS = pathlib.Path(os.environ.get("LORA_RESULT_DIR", str(DEFAULT_DIR))) / "metrics.json"


def _load():
    if not RESULTS.exists():
        pytest.fail(
            f"RED: {RESULTS} not found. Submit sweep/cell.py (model=lora) to NSG "
            f"(tool 'PyTorch Python on Expanse'), then fetch outputs into this dir. "
            f"See docs/submit-rest.md."
        )
    return json.loads(RESULTS.read_text())


def test_lora_cell_ran_and_beat_chance():
    m = _load()
    assert m.get("model") == "lora"
    assert m.get("finished"), "job did not finish writing metrics.json"
    assert "model_error" not in m, m.get("trace", m.get("model_error"))
    assert "data_error" not in m, m.get("data_error")
    assert m.get("pip_ok") is True, m.get("pip_err")
    assert m["test_balanced_accuracy"] > m["chance"], (
        f"LoRA cell did not beat chance: {m['test_balanced_accuracy']} vs {m['chance']}"
    )


def test_lora_is_actually_peft_not_full_finetune():
    m = _load()
    trainable, total = m["trainable_params"], m["total_params"]
    frac = trainable / total
    assert frac < 0.5, (
        f"trainable_params/total_params = {frac:.3f} — too high for LoRA, "
        f"looks like a full-finetune-sized parameter count instead"
    )
    assert trainable > 0, "zero trainable params — LoRA adapter targeted nothing"
