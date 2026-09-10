#!/usr/bin/env python3
"""Build leakage-aware fine-tuning CSVs from structure triplets v1.

Training unit = unique scored pairs linked from triplets:
- label 1: (anchor, D) direct PDB-contact positives (ad_pair_id);
- label 2: (anchor, C) assembly-context noncontacts (ac_pair_id), restricted to
  the hard-negative core strata (3_10, 11_24) by default; the gt24 exploration
  layer is opt-in via --include-gt24.

Split is grouped by the A-C example PDB: all pairs from triplets whose A-C
evidence comes from one PDB stay in the same split, so no PDB appears in both
train and dev. Homology-level grouping is still pending (v1 audit gate), so
this split controls exact-structure leakage only, not homologous-chain leakage.

Output CSVs follow PLM-interact train_binary.py format: query,text,label
(sequences are canonical UniProt, both orientations are added by the trainer).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

CORE_STRATA = ("3_10", "11_24")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=Path, required=True,
                        help="triplet_score_links_v1.tsv")
    parser.add_argument("--pairs", type=Path, required=True,
                        help="scoring_pairs_v1.tsv")
    parser.add_argument("--ac-table", type=Path, required=True,
                        help="structure_ac_base_v1.tsv (for pdb grouping)")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-gt24", action="store_true",
                        help="also include the gt24 exploration layer")
    parser.add_argument("--dev-frac", type=float, default=0.1)
    parser.add_argument("--seed", default="20260807")
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, str]:
    seqs = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            acc, seq = row.get("accession", ""), row.get("canonical_sequence", "")
            if acc and seq:
                seqs[acc] = seq.replace(" ", "").upper()
    return seqs


def stable_dev_fraction(seed: str, group: str) -> float:
    digest = hashlib.sha256(f"{seed}\t{group}".encode()).hexdigest()
    return int(digest, 16) % 10000 / 10000.0


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequences = load_metadata(args.metadata)
    allowed = set(CORE_STRATA) | ({"gt24"} if args.include_gt24 else set())

    ac_pdb = {}
    with args.ac_table.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ac_pdb[row["ac_id"]] = row["pdb_id"]

    pairs = {r["pair_id"]: r for r in csv.DictReader(
        args.pairs.open(encoding="utf-8", newline=""), delimiter="\t")}

    # Unique labelled pairs linked from triplets, with their PDB group.
    labelled: dict[str, dict[str, str]] = {}
    skipped_stratum = Counter()
    for link in csv.DictReader(args.links.open(encoding="utf-8", newline=""), delimiter="\t"):
        if link["assembly_size_stratum"] not in allowed:
            skipped_stratum[link["assembly_size_stratum"]] += 1
            continue
        pdb = ac_pdb.get(link["ac_id"], "")
        for pid, label in ((link["ac_pair_id"], "0"), (link["ad_pair_id"], "1")):
            entry = labelled.setdefault(pid, {
                "pair_id": pid, "label": label, "pdb_group": pdb,
                "role": pairs[pid]["role"] if pid in pairs else "",
            })
            if entry["label"] != label:
                raise SystemExit(f"conflicting labels for {pid}")
    counts = Counter(r["label"] for r in labelled.values())

    # Grouped split by PDB.
    groups = sorted({r["pdb_group"] for r in labelled.values()})
    dev_groups = {g for g in groups if stable_dev_fraction(args.seed, g) < args.dev_frac}
    rows_train, rows_dev = [], []
    missing_seq = 0
    for pid in sorted(labelled):
        rec = labelled[pid]
        pair = pairs.get(pid)
        if not pair:
            continue
        sx, sy = sequences.get(pair["protein_x"], ""), sequences.get(pair["protein_y"], "")
        if not sx or not sy:
            missing_seq += 1
            continue
        row = {"query": sx, "text": sy, "label": rec["label"]}
        (rows_dev if rec["pdb_group"] in dev_groups else rows_train).append(row)

    def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["query", "text", "label"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "train_v1.csv", rows_train)
    write_csv(args.output_dir / "dev_v1.csv", rows_dev)
    # train_binary.py requires a test path; it is only used for its predict
    # metrics, so point it at dev (never used for model selection here).
    write_csv(args.output_dir / "test_v1.csv", rows_dev)

    summary = {
        "status": "exploratory_finetune_split_pdb_grouped_homology_pending",
        "include_gt24": args.include_gt24,
        "unique_labelled_pairs": len(labelled),
        "label_counts": dict(counts),
        "triplet_links_skipped_by_stratum": dict(skipped_stratum),
        "pdb_groups": len(groups),
        "dev_pdb_groups": len(dev_groups),
        "train_rows": len(rows_train),
        "dev_rows": len(rows_dev),
        "train_label_counts": dict(Counter(r["label"] for r in rows_train)),
        "dev_label_counts": dict(Counter(r["label"] for r in rows_dev)),
        "pairs_missing_sequence": missing_seq,
    }
    (args.output_dir / "summary_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
