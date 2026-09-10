#!/usr/bin/env python3
"""Freeze Figure 4/6 negative-semantics cohorts for MINT and SPRINT scoring.

Locked pair sets match published PLM-Interact Figure 4/6:
  Negatome experimental direct (clean) = 443
  ProtNegDB experimental direct (independent) = 104
  structural-clean core noncontact = 2480

No training. MINT reuses existing five-seed full-core pair scores where the
same SP pair_id already exists; remaining pairs are written for new inference.
SPRINT gets one joint FASTA / train / test so structural and experimental
scores share the same HSP universe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import analyze_figure6_protneg_assay_v1 as fig6
import prepare_sprint_eval_inputs_v1 as sprint


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/interim/mint_sprint_neg_semantics_v1"
LOCKED = {
    "negatome_direct": 443,
    "protneg_direct": 104,
    "structural_clean": 2480,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument(
        "--swissprot",
        type=Path,
        default=Path("/data/chs/12.codebuddy/bridge/data/uniprot_sprot.fasta.gz"),
    )
    p.add_argument(
        "--string-train",
        type=Path,
        default=ROOT / "data/external/model_training_sets/plm_interact/string_v12/protein.pairs_human_V12_train.tsv",
    )
    p.add_argument(
        "--train-map",
        type=Path,
        default=ROOT / "data/interim/train_neighbor_map_v0/dscript_human",
    )
    return p.parse_args()


def read_rows(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def acc_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def load_locked_cohorts(swissprot: Path, string_train: Path) -> list[dict[str, object]]:
    scoring = read_rows(ROOT / "data/interim/structure_triplet_scoring_v1/scoring_pairs_v1.tsv", "\t")
    ac_to_pair = {
        row["ac_id"]: row
        for row in scoring
        if row["role"] == "ac_negative" and row["ac_id"]
    }
    seqs = {
        row["pair_id"]: (row["query"], row["text"])
        for row in read_rows(ROOT / "data/interim/structure_triplet_scoring_v1/plminteract_pairs_v1.csv", ",")
    }
    mint_have = {
        row["pair_id"]: row["score_mean"]
        for row in read_rows(ROOT / "data/interim/structure_core_mint_full_v1/mint_pair_scores_v1.tsv", "\t")
    }
    features = read_rows(
        ROOT / "data/interim/figure5_structure_score_drivers_v2/figure5_ac_features_v2.tsv", "\t"
    )

    rows: list[dict[str, object]] = []
    for feat in features:
        if not (
            feat["core_flag"] == "1"
            and feat["truncated"] == "0"
            and feat["train_pair"] == "0"
            and feat["association_class"] != "direct_conflict"
        ):
            continue
        meta = ac_to_pair[feat["ac_id"]]
        pair_id = meta["pair_id"]
        seq_a, seq_b = seqs[pair_id]
        rows.append({
            "cohort": "structural_clean",
            "pair_id": pair_id,
            "protein_a": meta["protein_x"],
            "protein_b": meta["protein_y"],
            "seq_a": seq_a,
            "seq_b": seq_b,
            "plm_score": feat["score"],
            "mint_existing_score": mint_have.get(pair_id, ""),
            "mint_score_source": "existing" if pair_id in mint_have else "new",
        })

    args = argparse.Namespace(
        negatome_pairs=ROOT / "data/interim/negatome_manual_plminteract_v1/negatome_manual_pairs_v1.tsv",
        negatome_sequences=ROOT / "data/interim/negatome_manual_plminteract_v1/negatome_manual_pairs_v1.csv",
        negatome_scores=ROOT / "data/interim/negatome_manual_plminteract_v1/negatome_manual_scores_v1.csv",
        swissprot=swissprot,
        string_train=string_train,
    )
    pairs = read_rows(args.negatome_pairs, "\t")
    sequences = read_rows(args.negatome_sequences, ",")
    if len(pairs) != len(sequences):
        raise SystemExit("Negatome pair/sequence mismatch")
    by_acc = {
        acc_key(pair["protein_a"], pair["protein_b"]): seq
        for pair, seq in zip(pairs, sequences)
    }
    for item in fig6.load_negatome_clean(args):
        seq = by_acc[acc_key(item["protein_a"], item["protein_b"])]
        rows.append({
            "cohort": "negatome_direct",
            "pair_id": f"NG:{item['protein_a']}|{item['protein_b']}",
            "protein_a": item["protein_a"],
            "protein_b": item["protein_b"],
            "seq_a": seq["query"],
            "seq_b": seq["text"],
            "plm_score": item["score"],
            "mint_existing_score": "",
            "mint_score_source": "new",
        })

    score_map = fig6.load_score_map(
        ROOT / "data/interim/figure6_protneg_assay_v1/protneg_assay_negative_scores_v1.csv"
    )
    seq_map = {
        row["directed_pair_id"]: row
        for row in read_rows(
            ROOT / "data/interim/figure6_protneg_assay_v1/protneg_assay_negative_pairs_v1.csv", ","
        )
    }
    for item in read_rows(ROOT / "data/interim/figure6_protneg_assay_v1/protneg_assay_manifest_v1.tsv", "\t"):
        score_id = item.get("negative_score_id", "")
        if not score_id:
            continue
        if item.get("exact_negatome_pair_overlap") == "1":
            continue
        if item.get("direct_assay_negative") != "1":
            continue
        seq = seq_map[score_id]
        rows.append({
            "cohort": "protneg_direct",
            "pair_id": score_id.rsplit(":", 1)[0],
            "protein_a": item["accession_a"],
            "protein_b": item["accession_b"],
            "seq_a": seq["query"],
            "seq_b": seq["text"],
            "plm_score": score_map[score_id],
            "mint_existing_score": "",
            "mint_score_source": "new",
        })

    counts = {name: sum(row["cohort"] == name for row in rows) for name in LOCKED}
    if counts != LOCKED:
        raise SystemExit(f"locked cohort counts drifted: {counts}")
    return rows


def write_mint_inputs(out: Path, rows: list[dict[str, object]]) -> list[dict[str, str]]:
    directed = []
    for row in rows:
        if row["mint_score_source"] != "new":
            continue
        directed.extend([
            {
                "directed_pair_id": f"{row['pair_id']}:forward",
                "pair_id": row["pair_id"],
                "direction": "forward",
                "query": row["seq_a"],
                "text": row["seq_b"],
            },
            {
                "directed_pair_id": f"{row['pair_id']}:reverse",
                "pair_id": row["pair_id"],
                "direction": "reverse",
                "query": row["seq_b"],
                "text": row["seq_a"],
            },
        ])
    write_tsv(
        out / "mint_directed_new_v1.tsv",
        directed,
        ["directed_pair_id", "pair_id", "direction", "query", "text"],
    )
    return directed


def write_sprint_inputs(
    out: Path,
    rows: list[dict[str, object]],
    train_map: Path,
) -> dict[str, object]:
    sprint_dir = out / "sprint"
    sprint_dir.mkdir(parents=True, exist_ok=True)
    proteins: dict[str, str] = {}
    skipped: list[dict[str, str]] = []
    for row in rows:
        sprint.add_protein(proteins, skipped, str(row["protein_a"]), str(row["seq_a"]), row["cohort"])
        sprint.add_protein(proteins, skipped, str(row["protein_b"]), str(row["seq_b"]), row["cohort"])

    sha_to_name: dict[str, str] = {}
    for name, seq in proteins.items():
        sha_to_name.setdefault(sprint.seq_key(seq), name)

    n_train_seq_added = 0
    with (train_map / "proteins.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            raw = row.get("sequence") or ""
            if not raw:
                continue
            seq = sprint.sanitize_sequence(raw)
            hashed = sprint.seq_key(seq)
            original_key = row["protein_key"]
            if hashed in sha_to_name:
                sha_to_name.setdefault(original_key, sha_to_name[hashed])
                continue
            before = len(proteins)
            sprint.add_protein(proteins, skipped, original_key, raw, "dscript_human_train")
            if len(proteins) > before:
                sha_to_name[hashed] = original_key
                sha_to_name[original_key] = original_key
                n_train_seq_added += 1

    train_pairs: list[tuple[str, str]] = []
    n_train_pos = 0
    n_train_mapped = 0
    with (train_map / "pairs.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["label"] != "1" or row["split"] != "train":
                continue
            n_train_pos += 1
            name_a = sha_to_name.get(row["protein_key_a"])
            name_b = sha_to_name.get(row["protein_key_b"])
            if name_a and name_b and name_a != name_b:
                train_pairs.append((name_a, name_b))
                n_train_mapped += 1

    fasta_names = set(proteins)
    test_pairs = []
    missing = []
    for row in rows:
        left, right = str(row["protein_a"]), str(row["protein_b"])
        if left in fasta_names and right in fasta_names and left != right:
            test_pairs.append((left, right, row["cohort"], row["pair_id"]))
        else:
            missing.append({
                "pair_id": row["pair_id"],
                "cohort": row["cohort"],
                "protein_a": left,
                "protein_b": right,
                "reason": "short_or_missing_sequence" if left == right else "protein_not_in_fasta",
            })

    with (sprint_dir / "sprint_proteins_v1.fasta").open("w", encoding="utf-8") as handle:
        for name in sorted(proteins):
            handle.write(f">{name}\n{proteins[name]}\n")
    sprint.write_pair_file(sprint_dir / "train_positive_v1.txt", train_pairs)
    sprint.write_pair_file(
        sprint_dir / "test_positive_v1.txt",
        [(left, right) for left, right, _cohort, _pair_id in test_pairs],
    )
    (sprint_dir / "test_negative_v1.txt").write_text("", encoding="utf-8")
    write_tsv(
        sprint_dir / "test_pair_index_v1.tsv",
        [
            {"protein_a": left, "protein_b": right, "cohort": cohort, "pair_id": pair_id}
            for left, right, cohort, pair_id in test_pairs
        ],
        ["protein_a", "protein_b", "cohort", "pair_id"],
    )
    if skipped:
        write_tsv(sprint_dir / "skipped_proteins_v1.tsv", skipped, ["name", "source", "reason", "length"])
    if missing:
        (sprint_dir / "missing_test_pairs_v1.json").write_text(
            json.dumps(missing, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "n_proteins": len(proteins),
        "n_train_only_proteins_added": n_train_seq_added,
        "n_train_positives_source": n_train_pos,
        "n_train_positives_mapped": n_train_mapped,
        "n_test_pairs_kept": len(test_pairs),
        "n_test_pairs_missing": len(missing),
        "n_skipped_proteins": len(skipped),
        "min_sequence_length": sprint.MIN_LEN,
    }
    (sprint_dir / "prepare_summary_v1.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = load_locked_cohorts(args.swissprot, args.string_train)
    fields = [
        "cohort", "pair_id", "protein_a", "protein_b", "seq_a", "seq_b",
        "plm_score", "mint_existing_score", "mint_score_source",
    ]
    write_tsv(out / "cohort_pairs_v1.tsv", rows, fields)
    directed = write_mint_inputs(out, rows)
    sprint_summary = write_sprint_inputs(out, rows, args.train_map)
    summary = {
        "status": "mint_sprint_neg_semantics_prepare_v1",
        "locked_counts": {
            name: sum(row["cohort"] == name for row in rows) for name in LOCKED
        },
        "mint_new_pairs": sum(row["mint_score_source"] == "new" for row in rows),
        "mint_existing_pairs": sum(row["mint_score_source"] == "existing" for row in rows),
        "mint_directed_new": len(directed),
        "sprint": sprint_summary,
        "no_training": True,
    }
    (out / "prepare_summary_v1.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
