#!/usr/bin/env python3
"""Your real PyTorch workload goes here (e.g. a LoRA fine-tune of a frozen REVE encoder).

Skeleton that respects the NSG constraints:
  - no runtime internet  -> load models/data from local dirs (vendored into the zip or $NEMARPATH)
  - one GPU (V100 32GB expected) -> keep batch/precision in budget
  - <=48h wallclock       -> checkpoint to the working dir; NSG returns the whole working dir
Write results (metrics.json / checkpoints) into the CWD so they come back in the output zip.
"""
import json
import os
import time


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- data ---
    # NEMAR (OpenNeuro ds######) if present, else data vendored into the zip under ./data:
    nemar = os.environ.get("NEMARPATH")
    # ds = os.path.join(nemar, "ds002718") if nemar else "data/<your-split>"
    # ... load EEG, build loaders ...

    # --- model ---
    # Load a frozen encoder from local files (HF is unreachable at runtime):
    # from transformers import AutoModel
    # enc = AutoModel.from_pretrained("vendor/reve-base", local_files_only=True).to(dev).eval()
    # from peft import LoraConfig, get_peft_model
    # model = get_peft_model(head, LoraConfig(...)).to(dev)

    # --- train / eval (checkpoint to CWD) ---
    out = {"schema": "nsg-agent-kit/job/v1", "device": dev,
           "torch": torch.__version__, "nemarpath": nemar,
           "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open("metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
