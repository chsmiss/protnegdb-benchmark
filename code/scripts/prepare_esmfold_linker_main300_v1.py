#!/usr/bin/env python3
"""Prepare ESMFold linker-dimer shards for main300.

Reuses finished pilot48 prediction JSONs, then splits remaining unique pairs
across GPUs by pair_tokens^2 so long jobs do not pile onto one card.
Does not run folding.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/interim/af3_partner_ranking_v1"
DEFAULT_GPUS = "0,2,6,7"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--gpus", default=DEFAULT_GPUS)
    p.add_argument("--pairs", type=Path, default=None)
    p.add_argument("--pilot-pred", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def parse_gpus(text: str) -> list[int]:
    gpus = [int(part) for part in text.split(",") if part.strip() != ""]
    if not gpus or any(gpu < 0 or gpu > 7 for gpu in gpus):
        raise ValueError(f"gpus must be a non-empty subset of 0-7, got {text!r}")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"duplicate GPU indices: {text!r}")
    return gpus


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def link_existing(src: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    linked = 0
    if not src.is_dir():
        return 0
    for path in src.glob("*.json"):
        target = dest / path.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(path.resolve())
        linked += 1
    return linked


def assign_shards(rows: list[dict], gpus: list[int]) -> dict[int, list[dict]]:
    shards = {gpu: [] for gpu in gpus}
    load = {gpu: 0 for gpu in gpus}
    ordered = sorted(rows, key=lambda row: (-int(row["pair_tokens"]), row["job_id"]))
    for row in ordered:
        gpu = min(load, key=lambda key: (load[key], key))
        shards[gpu].append(row)
        load[gpu] += int(row["pair_tokens"]) ** 2
    return shards


def main() -> None:
    args = parse_args()
    base = args.base
    gpus = parse_gpus(args.gpus)
    pairs_path = args.pairs or (base / "main300_unique_pairs_v1.tsv")
    pilot_pred = args.pilot_pred or (base / "esmfold_linker_pilot48_v1" / "predictions")
    out = args.output_dir or (base / "esmfold_linker_main300_v1")
    pred = out / "predictions"
    pairs = read_tsv(pairs_path)
    linked = link_existing(pilot_pred, pred)
    remaining = [row for row in pairs if not (pred / f"{row['job_id']}.json").exists()]
    shards = assign_shards(remaining, gpus)
    fields = list(pairs[0]) if pairs else []
    gpu_jobs = {}
    for gpu, rows in shards.items():
        write_tsv(out / f"shard_gpu{gpu}_pairs_v1.tsv", rows, fields)
        gpu_jobs[str(gpu)] = len(rows)
    report = {
        "n_pairs": len(pairs),
        "n_linked_from_pilot48": linked,
        "n_already_present": len(pairs) - len(remaining),
        "n_remaining": len(remaining),
        "gpus": gpus,
        "gpu_jobs": gpu_jobs,
    }
    (out / "prepare_shards_v1.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
