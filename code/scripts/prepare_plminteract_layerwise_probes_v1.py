#!/usr/bin/env python3
"""Build Figure 3 family-disjoint structure triplets for layerwise probes.

Train = train_eligible, test = family_unseen. Roles stay AD / AC / AR.
AR for train reuses locked matched-random C; AR for family_unseen is a
same-split taxon permutation of C. Not literature matched-neighbor C.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FT = ROOT / "data/interim/structure_family_transfer_v4/id50"
DEFAULT_OUT = ROOT / "data/interim/plminteract_layerwise_probes_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--assignments", type=Path, default=FT / "triplet_assignments_v4.tsv")
    parser.add_argument("--eval-pairs", type=Path, default=FT / "eval_pairs_v4.csv")
    parser.add_argument("--train-random", type=Path, default=FT / "train_matched_random_v4.tsv")
    parser.add_argument(
        "--ac-table", type=Path,
        default=ROOT / "data/interim/structure_counterfactual_v1/structure_ac_base_v1.tsv",
    )
    parser.add_argument(
        "--scoring-pairs", type=Path,
        default=ROOT / "data/interim/structure_triplet_scoring_v1/scoring_pairs_v1.tsv",
    )
    return parser.parse_args()


def load_lib():
    path = ROOT / "code/scripts" / "plminteract_layerwise_probe_lib_v1.py"
    spec = importlib.util.spec_from_file_location("plminteract_layerwise_probe_lib_v1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_delim(path: Path, delim: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delim))


def main() -> None:
    args = parse_args()
    lib = load_lib()
    assignments = [
        row for row in read_delim(args.assignments, "\t")
        if row["split"] in {"train_eligible", "family_unseen"}
    ]
    train = [row for row in assignments if row["split"] == "train_eligible"]
    test = [row for row in assignments if row["split"] == "family_unseen"]
    if not lib.families_disjoint(train, test):
        raise SystemExit("train_eligible families overlap family_unseen")

    ac_tax = {row["ac_id"]: row.get("tax_id", "") for row in read_delim(args.ac_table, "\t")}
    eval_pairs = read_delim(args.eval_pairs, ",")
    pair_of: dict[tuple[str, str], dict[str, str]] = {}
    seq: dict[str, str] = {}
    asg_of = {row["triplet_id"]: row for row in assignments}
    for row in eval_pairs:
        trip = asg_of.get(row["triplet_id"])
        if trip is None:
            continue
        pair_of[(row["triplet_id"], row["role"])] = row
        seq[trip["anchor"]] = row["query"]
        if row["role"] == "ac":
            seq[trip["c"]] = row["text"]
        elif row["role"] == "ad":
            seq[trip["d"]] = row["text"]

    train_random = {
        row["triplet_id"]: row["c_random"]
        for row in read_delim(args.train_random, "\t")
        if row.get("permutation_status", "ok") == "ok"
    }
    known = set()
    for row in read_delim(args.scoring_pairs, "\t"):
        if row.get("role") == "ad_positive":
            known.add(frozenset({row["protein_x"], row["protein_y"]}))
    unseen_rows = []
    for row in test:
        rec = dict(row)
        rec["tax_c"] = ac_tax.get(row["ac_id"], "")
        unseen_rows.append(rec)
    test_random = lib.matched_random_from_rows(unseen_rows, known)
    unseen_c_pool = [row["c"] for row in test]
    for row in test:
        current = test_random.get(row["triplet_id"], "")
        if current and current != row["c"]:
            continue
        for cand in unseen_c_pool:
            if cand in {row["anchor"], row["d"], row["c"]}:
                continue
            if frozenset({row["anchor"], cand}) in known:
                continue
            test_random[row["triplet_id"]] = cand
            break

    triplets = []
    pairs: dict[str, dict[str, str]] = {}
    skipped = {"missing_ad_ac": 0, "missing_r_seq": 0, "r_unpermuted": 0}

    def add_pair(ident: str, query: str, text: str, source: str) -> None:
        if ident in pairs:
            return
        pairs[ident] = {
            "directed_pair_id": ident,
            "query": query,
            "text": text,
            "source": source,
            "cohort": "structure_family_v4",
        }

    for row in assignments:
        ad = pair_of.get((row["triplet_id"], "ad"))
        ac = pair_of.get((row["triplet_id"], "ac"))
        if ad is None or ac is None:
            skipped["missing_ad_ac"] += 1
            continue
        if row["split"] == "train_eligible":
            partner_r = train_random.get(row["triplet_id"], "")
        else:
            partner_r = test_random.get(row["triplet_id"], "")
        if not partner_r or partner_r == row["c"]:
            skipped["r_unpermuted"] += 1
            continue
        seq_r = seq.get(partner_r, "")
        if not seq_r or row["anchor"] not in seq:
            skipped["missing_r_seq"] += 1
            continue
        ar_id = f"{row['triplet_id']}|AR"
        add_pair(ad["directed_pair_id"], ad["query"], ad["text"], "ad")
        add_pair(ac["directed_pair_id"], ac["query"], ac["text"], "ac")
        add_pair(ar_id, seq[row["anchor"]], seq_r, "ar")
        triplets.append({
            "triplet_id": row["triplet_id"],
            "split": row["split"],
            "anchor": row["anchor"],
            "c": row["c"],
            "d": row["d"],
            "r": partner_r,
            "fam_a": row["fam_a"],
            "fam_c": row["fam_c"],
            "fam_d": row["fam_d"],
            "ad_id": ad["directed_pair_id"],
            "ac_id": ac["directed_pair_id"],
            "ar_id": ar_id,
        })

    train_n = sum(1 for r in triplets if r["split"] == "train_eligible")
    test_n = sum(1 for r in triplets if r["split"] == "family_unseen")
    if not lib.families_disjoint(
        [r for r in triplets if r["split"] == "train_eligible"],
        [r for r in triplets if r["split"] == "family_unseen"],
    ):
        raise SystemExit("prepared triplets leak families")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trip_path = args.output_dir / "structure_triplets_v1.tsv"
    fields = ["triplet_id", "split", "anchor", "c", "d", "r",
              "fam_a", "fam_c", "fam_d", "ad_id", "ac_id", "ar_id"]
    with trip_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(triplets)
    pair_path = args.output_dir / "unique_directed_pairs_v1.csv"
    with pair_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["directed_pair_id", "query", "text", "source", "cohort"])
        writer.writeheader()
        writer.writerows(pairs.values())
    summary = {
        "n_triplets": len(triplets),
        "n_train_eligible": train_n,
        "n_family_unseen": test_n,
        "n_directed_pairs": len(pairs),
        "family_disjoint": True,
        "skipped": skipped,
        "note": "Independent literature assay negatives (Negatome 443 + ProtNeg-104 "
                "from 480 papers) are not this cohort. 1000-pilot matched C is a "
                "neighbor-C construction, not the Codex literature-negative FPR set.",
    }
    (args.output_dir / "prepare_summary_v1.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
