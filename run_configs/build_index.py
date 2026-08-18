#!/usr/bin/env python3
"""Build runs_index.csv from the per-run argument dumps in run_configs/.

Each `<dataset>/<run>/<seed>.args` file is the header block that run_expt.py
writes at the top of every log.txt (a dump of `vars(args)`). This script parses
them into one row per run and derives the canonical experiment name described in
NAMING.md, so tables can be grouped by axis instead of by ad-hoc folder name.

Usage:  python3 run_configs/build_index.py
"""
import csv
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args_file(path):
    args = {}
    with open(path, errors="ignore") as f:
        for line in f:
            if ": " in line:
                k, v = line.rstrip("\n").split(": ", 1)
                args[k] = v
    return args


def is_true(v):
    return str(v).strip().lower() == "true"


def head_loss(a):
    """Objective the (last) trainable layer is optimised under."""
    lt = a.get("Loss type", "erm")
    if lt == "group_dro":
        return "gdro"
    if lt == "reweight":
        return "rw"
    return "erm"


def feature_source(a):
    """Which run produced the frozen features, read off the checkpoint name."""
    p = a.get("Pretrained path", "") or ""
    if not p:
        return None
    base = os.path.basename(p)
    for prefix in ("erm_gdro", "erm_rw", "erm", "gdro", "rw"):
        if base.startswith(prefix + "_"):
            return prefix.replace("erm_gdro", "ermgdro").replace("erm_rw", "ermrw")
    return "unk"


def method(a, run):
    if run.startswith("jtt") or "_jtt" in run or run.startswith("ermjtt"):
        return "e2e.jtt"
    src = feature_source(a)
    if src is None:
        return f"e2e.{head_loss(a)}"
    return f"ft.{src}.{head_loss(a)}"


# Metadata CSVs whose split 0 is a group-balanced subsample rather than the
# original training split. Verified by counting groups in each file.
BALANCED_METADATA = ("metadata_dfr.csv", "list_attr_celeba_dfr.csv", "dfr")


def data_source(a):
    """Where the head's training data comes from: original train split or val."""
    md = a.get("Metadata csv name", "") or ""
    return "val" if "val_as_train" in md else "train"


def balancing(a):
    """How group balance is imposed: subsampled CSV, sampler reweighting, both."""
    md = a.get("Metadata csv name", "") or ""
    parts = []
    if any(md == b or md.startswith(b) for b in BALANCED_METADATA):
        parts.append("sub")
    if is_true(a.get("Reweight groups")):
        parts.append("rw")
    return ".".join(parts) if parts else "none"


def correlation(a, dataset):
    if dataset == "CUB":
        m = re.search(r"waterbird_complete([\d.]+)", a.get("Target name", ""))
        return m.group(1) if m else "?"
    if dataset == "MNISTCIFAR":
        m = re.search(r"[\d.]+", a.get("Confounder names", ""))
        return m.group(0) if m else "?"
    return "std"


def fmt(v):
    """Compact, filename-safe number formatting (1e-05 -> 1e-5, 1.0 -> 1.0)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == 0:
        return "0"
    s = f"{f:g}"
    if "e" not in s and "." not in s:
        s += ".0"  # keep frac-1.0 / wd-1.0 rather than frac-1 / wd-1
    return s.replace("e-0", "e-").replace("e+0", "e")


def bn_mode(a):
    """BatchNorm mode of the frozen part during last-layer retraining.

    run_epoch always calls model.train(), so every run made before --train_bn
    existed kept adapting BatchNorm statistics; those dumps have no "Train bn"
    key at all. End-to-end runs train BN legitimately, so the axis is n/a.
    """
    if not is_true(a.get("Finetune")):
        return "n/a"
    if "Train bn" not in a:
        return "train"
    return "train" if is_true(a["Train bn"]) else "eval"


def canonical_name(a, dataset, run):
    fields = [method(a, run), f"c-{correlation(a, dataset)}"]
    if feature_source(a) is not None:
        fields.append(f"src-{data_source(a)}")
    fields.append(f"bal-{balancing(a)}")
    fields.append(f"frac-{fmt(a.get('Fraction', 1.0))}")
    fields.append(f"wd-{fmt(a.get('Weight decay', 0))}")
    fields.append(f"lr-{fmt(a.get('Lr', 0))}")
    fields.append(f"ep-{a.get('N epochs', '?')}")
    if bn_mode(a) != "n/a":
        fields.append(f"bn-{bn_mode(a)}")
    unfreeze = a.get("Unfreeze", "0") or "0"
    if unfreeze not in ("0", ""):
        fields.append(f"unfreeze-{unfreeze}")
    restart = a.get("Restart layers", "0") or "0"
    if restart not in ("0", ""):
        fields.append(f"restart-{restart}")
    return "__".join(fields)


COLUMNS = [
    "dataset", "old_run", "seed", "canonical", "method", "feat_src", "head",
    "c", "src", "bal", "frac", "wd", "lr", "n_epochs", "batch_size", "bn",
    "unfreeze", "restart", "metadata_csv", "pretrained", "code_version", "path",
]


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "*", "*", "*.args"))):
        rel = os.path.relpath(path, HERE)
        dataset, run, seed_file = rel.split(os.sep)
        seed = seed_file[:-len(".args")]
        a = parse_args_file(path)
        rows.append({
            "dataset": dataset,
            "old_run": run,
            "seed": seed,
            "canonical": canonical_name(a, dataset, run),
            "method": method(a, run),
            "feat_src": feature_source(a) or "",
            "head": head_loss(a),
            "c": correlation(a, dataset),
            "src": data_source(a) if feature_source(a) is not None else "",
            "bal": balancing(a),
            "frac": a.get("Fraction", ""),
            "wd": a.get("Weight decay", ""),
            "lr": a.get("Lr", ""),
            "n_epochs": a.get("N epochs", ""),
            "batch_size": a.get("Batch size", ""),
            "bn": bn_mode(a),
            "unfreeze": a.get("Unfreeze", ""),
            "restart": a.get("Restart layers", ""),
            "metadata_csv": a.get("Metadata csv name", ""),
            "pretrained": a.get("Pretrained path", ""),
            # Which argparse keys the dump contains dates the build that ran it.
            "code_version": "pre-normalize" if "Normalize" not in a else "current",
            "path": a.get("Log dir", ""),
        })

    out = os.path.join(HERE, "runs_index.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
