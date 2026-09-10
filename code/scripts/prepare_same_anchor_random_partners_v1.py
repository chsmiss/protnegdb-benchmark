#!/usr/bin/env python3
"""Prepare same-anchor random partners ``A-R-`` for the 1000-triplet pilot.

The biological anchor is ``AC ∩ AD``, not the left token of a sorted pair key.
For every such ``A``, one random non-interacting candidate ``R`` is chosen
deterministically from the pilot protein universe, excluding:

* ``A`` itself;
* any known positive partner (A-D) or negative partner (A-C) of ``A``;
* proteins sharing an mmseqs-30% cluster with any known partner (so ``R`` is
  not a sequence-similar stand-in for a real partner).

Outputs an undirected partner table plus directed scoring inputs for
PLM-interact / MINT (sequence pairs) and D-SCRIPT / Topsy-Turvy (pair rows;
reuse the pilot fasta and embedding h5).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PILOT_DIR = ROOT / "data/interim/counterfactual_1000_triplet_pilot_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT_DIR)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_PILOT_DIR / "random_partner_v1")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def pair_members(pair_key: str) -> set[str]:
    left, right = pair_key.split("|")
    return {left, right}


def triplet_anchor(row: dict[str, str]) -> str:
    shared = pair_members(row["ac_pair_key"]) & pair_members(row["ad_pair_key"])
    if len(shared) != 1:
        raise ValueError(f"no unique anchor in {row.get('unified_triplet_id', row)}")
    return next(iter(shared))


def other_member(pair_key: str, anchor: str) -> str:
    members = pair_members(pair_key)
    members.discard(anchor)
    if len(members) != 1:
        raise ValueError(f"cannot take counterpart of {anchor} in {pair_key}")
    return next(iter(members))


def read_fasta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    name = None
    seq = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(seq)
                name = line[1:].split()[0]
                seq = []
            elif name is not None:
                seq.append(line)
    if name is not None:
        out[name] = "".join(seq)
    return out


def main() -> None:
    args = parse_args()
    trips = read_tsv(args.pilot_dir / "pilot_triplets_v1.tsv")
    proteins = read_fasta(args.pilot_dir / "pilot_proteins_v1.fasta")
    clusters = {}
    with (args.pilot_dir / "clusters_pilot_v1_cluster.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                clusters[parts[1]] = parts[0]

    pos_partners: dict[str, set[str]] = defaultdict(set)
    neg_partners: dict[str, set[str]] = defaultdict(set)
    for t in trips:
        a = triplet_anchor(t)
        pos_partners[a].add(other_member(t["ad_pair_key"], a))
        neg_partners[a].add(other_member(t["ac_pair_key"], a))

    anchors = sorted(set(pos_partners) | set(neg_partners))
    universe = sorted(set(proteins) - {"", None})
    rng = random.Random(args.seed)
    rows = []
    skipped = []
    for anchor in anchors:
        known = {anchor} | pos_partners.get(anchor, set()) | neg_partners.get(anchor, set())
        known_clusters = {clusters[p] for p in known if p in clusters}
        candidates = [
            p for p in universe
            if p not in known
            and (p not in clusters or clusters[p] not in known_clusters)
        ]
        if not candidates:
            skipped.append((anchor, "no_candidate"))
            continue
        partner = rng.choice(candidates)
        rows.append({"anchor": anchor, "random_partner": partner})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "random_partners_v1.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=["anchor", "random_partner"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # Directed inputs: PLM-interact CSV and MINT TSV share query/text; D-SCRIPT
    # needs pair rows only (fasta/embedding already exist).
    plm_rows = []
    mint_rows = []
    dscript_rows = []
    for row in rows:
        a, r = row["anchor"], row["random_partner"]
        for direction, (x, y) in (("forward", (a, r)), ("reverse", (r, a))):
            did = f"{a}|{r}:{direction}"
            plm_rows.append({"directed_pair_id": did, "query": proteins[x], "text": proteins[y]})
            mint_rows.append({"directed_pair_id": did, "query": proteins[x], "text": proteins[y]})
            dscript_rows.append((x, y))
    with (args.output_dir / "random_partners_plminteract_pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["directed_pair_id", "query", "text"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(plm_rows)
    with (args.output_dir / "random_partners_mint_pairs.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=["directed_pair_id", "query", "text"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(mint_rows)
    with (args.output_dir / "random_partners_dscript_pairs.tsv").open("w", encoding="utf-8", newline="") as handle:
        for x, y in dscript_rows:
            handle.write(f"{x}\t{y}\n")

    summary = {
        "anchors": len(anchors),
        "assigned": len(rows),
        "skipped": len(skipped),
        "seed": args.seed,
    }
    (args.output_dir / "random_partners_summary_v1.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
