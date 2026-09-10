#!/usr/bin/env python3
"""Prepare reciprocal MINT inputs for every Figure 2 core triplet pair."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/interim/structure_triplet_scoring_v1"
OUT = ROOT / "data/interim/structure_core_mint_full_v1"
CORE = {"3_10", "11_24"}


def read(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=Path, default=BASE / "triplet_score_links_v1.tsv")
    parser.add_argument("--pairs", type=Path, default=BASE / "plminteract_pairs_v1.csv")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    links = [row for row in read(args.links, "\t") if row["assembly_size_stratum"] in CORE]
    pair_rows = read(args.pairs, ",")
    seqs = {row["pair_id"]: (row["query"], row["text"]) for row in pair_rows}
    pair_ids = sorted({row[key] for row in links for key in ("ac_pair_id", "ad_pair_id")})
    missing = [pair_id for pair_id in pair_ids if pair_id not in seqs]
    if missing:
        raise SystemExit(f"missing sequences for {len(missing)} selected pairs")
    directed = []
    for pair_id in pair_ids:
        query, text = seqs[pair_id]
        directed.extend([
            {"directed_pair_id": f"{pair_id}:forward", "pair_id": pair_id,
             "direction": "forward", "query": query, "text": text},
            {"directed_pair_id": f"{pair_id}:reverse", "pair_id": pair_id,
             "direction": "reverse", "query": text, "text": query},
        ])
    write(args.output_dir / "mint_directed_pairs_v1.tsv", directed,
          ["directed_pair_id", "pair_id", "direction", "query", "text"])
    write(args.output_dir / "core_triplet_links_v1.tsv", links, list(links[0]))
    summary = {
        "protocol": "all Figure 2 core triplets; reciprocal directions; MINT Bernett; five fixed crop seeds",
        "core_triplets": len(links),
        "unique_ac": len({row["ac_id"] for row in links}),
        "unique_pairs": len(pair_ids),
        "directed_rows": len(directed),
        "crop_seeds": [13, 29, 47, 71, 97],
        "missing_pairs": len(missing),
    }
    if summary["core_triplets"] != 3365 or summary["unique_ac"] != 1606:
        raise SystemExit(f"unexpected locked core counts: {summary}")
    (args.output_dir / "prepare_summary_v1.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
