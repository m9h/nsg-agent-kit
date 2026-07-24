#!/usr/bin/env python3
"""Aggregate per-subject REVE reproduction results into a cross-subject mean vs the paper.

Usage: python aggregate_repro.py sweep/results/repro
Reads subject_*.json, computes mean ± std of test_balanced_accuracy, compares to the paper's
REVE-Base BCI-IV-2a number (0.6396 ± 0.0095), writes leaderboard.md + summary.json.
"""
import glob
import json
import os
import sys

PAPER = 0.6396
PAPER_STD = 0.0095


def main(outdir):
    rows = []
    for f in sorted(glob.glob(os.path.join(outdir, "subject_*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        rows.append(d)
    scored = [d for d in rows if isinstance(d.get("test_balanced_accuracy"), (int, float))]
    accs = [d["test_balanced_accuracy"] for d in scored]
    n = len(accs)
    mean = sum(accs) / n if n else float("nan")
    std = (sum((a - mean) ** 2 for a in accs) / n) ** 0.5 if n else float("nan")

    summary = {
        "n_subjects_scored": n,
        "n_subjects_total": len(rows),
        "reve_mean_balanced_accuracy": round(mean, 4) if n else None,
        "reve_std": round(std, 4) if n else None,
        "paper_reve_base": PAPER, "paper_std": PAPER_STD,
        "delta_vs_paper": round(mean - PAPER, 4) if n else None,
        "failed_subjects": [d.get("subject") for d in rows if not isinstance(
            d.get("test_balanced_accuracy"), (int, float))],
    }
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=2)

    lines = ["# REVE reproduction on BCI IV 2a (vs paper 0.6396 ± 0.0095)", "",
             f"**REVE-Base (our NSG run): {mean:.4f} ± {std:.4f}** over {n} subjects  ",
             f"Paper (Table 2): 0.6396 ± 0.0095  ·  Δ = {mean - PAPER:+.4f}", "",
             "| subject | bal. acc | acc | trainable | note |", "|--|--|--|--|--|"]
    for d in sorted(rows, key=lambda x: x.get("subject", 0)):
        s = d.get("subject", "?")
        ba = d.get("test_balanced_accuracy")
        ba = f"{ba:.4f}" if isinstance(ba, (int, float)) else "—"
        ac = d.get("test_accuracy")
        ac = f"{ac:.4f}" if isinstance(ac, (int, float)) else "—"
        tp = d.get("lora_trainable_params", "—")
        note = d.get("model_error") or d.get("data_error") or d.get("pip_err") or ""
        lines.append(f"| {s} | {ba} | {ac} | {tp} | {str(note)[:60]} |")
    open(os.path.join(outdir, "leaderboard.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote", os.path.join(outdir, "summary.json"), "and leaderboard.md")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sweep/results/repro")
